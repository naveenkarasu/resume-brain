use std::sync::Mutex;
use std::time::Duration;
use tauri::{
    AppHandle, Emitter, Manager,
    menu::{MenuBuilder, MenuItemBuilder},
    tray::TrayIconBuilder,
};
use tauri_plugin_shell::ShellExt;
use tauri_plugin_shell::process::CommandChild;

/// Holds the sidecar child process so we can kill it on exit.
struct SidecarState(Mutex<Option<CommandChild>>);

/// Check if the backend is healthy by hitting /health.
async fn check_backend_health() -> bool {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(2))
        .build()
        .unwrap_or_default();

    match client.get("http://127.0.0.1:8000/health").send().await {
        Ok(resp) => resp.status().is_success(),
        Err(_) => false,
    }
}

/// Wait for the backend to become healthy, with retries.
async fn wait_for_backend(max_retries: u32) -> bool {
    for i in 0..max_retries {
        if check_backend_health().await {
            println!("[sidecar] Backend healthy after {} attempts", i + 1);
            return true;
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
    eprintln!(
        "[sidecar] Backend failed to start after {} retries",
        max_retries
    );
    false
}

/// Read the API key from the store.
fn read_api_key(app: &AppHandle) -> String {
    let store = tauri_plugin_store::StoreBuilder::new(app, "settings.json")
        .build();
    match store {
        Ok(s) => s
            .get("gemini_api_key")
            .and_then(|v| v.as_str().map(String::from))
            .unwrap_or_default(),
        Err(_) => String::new(),
    }
}

/// Spawn the sidecar backend process, passing the API key as an env var.
async fn spawn_sidecar(app: AppHandle) {
    println!("[sidecar] Starting resume-brain backend...");

    let api_key = read_api_key(&app);

    let shell = app.shell();
    let mut sidecar_command = shell
        .sidecar("binaries/resume-brain")
        .expect("Failed to create sidecar command");

    if !api_key.is_empty() {
        sidecar_command = sidecar_command.env("GEMINI_API_KEY", &api_key);
        println!("[sidecar] API key provided ({} chars)", api_key.len());
    } else {
        println!("[sidecar] No API key set — running without LLM");
    }

    match sidecar_command.spawn() {
        Ok((mut rx, child)) => {
            // Store the child process for cleanup
            let state = app.state::<SidecarState>();
            *state.0.lock().unwrap() = Some(child);

            // Log sidecar output in background
            tauri::async_runtime::spawn(async move {
                use tauri_plugin_shell::process::CommandEvent;
                while let Some(event) = rx.recv().await {
                    match event {
                        CommandEvent::Stdout(line) => {
                            println!("[sidecar:out] {}", String::from_utf8_lossy(&line));
                        }
                        CommandEvent::Stderr(line) => {
                            eprintln!("[sidecar:err] {}", String::from_utf8_lossy(&line));
                        }
                        CommandEvent::Terminated(payload) => {
                            println!("[sidecar] Process terminated: {:?}", payload);
                            break;
                        }
                        CommandEvent::Error(err) => {
                            eprintln!("[sidecar] Error: {}", err);
                            break;
                        }
                        _ => {}
                    }
                }
            });

            // Wait for backend to become healthy (up to 60 seconds)
            if wait_for_backend(120).await {
                println!("[sidecar] Backend is ready!");
                let _ = app.emit("backend-ready", true);
            } else {
                eprintln!("[sidecar] Backend failed to start!");
                let _ = app.emit("backend-ready", false);
            }
        }
        Err(e) => {
            eprintln!("[sidecar] Failed to spawn: {}", e);
            let _ = app.emit("backend-ready", false);
        }
    }
}

/// Kill the currently-running sidecar process if any.
fn kill_sidecar(app: &AppHandle) {
    let state = app.state::<SidecarState>();
    let child = state.0.lock().unwrap().take();
    if let Some(child) = child {
        println!("[sidecar] Killing backend process...");
        let _ = child.kill();
        println!("[sidecar] Backend process killed.");
    }
}

// ── IPC Commands ──

#[tauri::command]
async fn backend_status() -> Result<String, String> {
    if check_backend_health().await {
        Ok("healthy".to_string())
    } else {
        Err("Backend is not responding".to_string())
    }
}

#[tauri::command]
fn get_api_key(app: AppHandle) -> Result<String, String> {
    Ok(read_api_key(&app))
}

#[tauri::command]
fn set_api_key(app: AppHandle, key: String) -> Result<(), String> {
    let store = tauri_plugin_store::StoreBuilder::new(&app, "settings.json")
        .build()
        .map_err(|e| e.to_string())?;
    store.set("gemini_api_key", serde_json::Value::String(key));
    store.save().map_err(|e| e.to_string())?;
    Ok(())
}

#[tauri::command]
async fn restart_sidecar(app: AppHandle) -> Result<(), String> {
    kill_sidecar(&app);
    // Small delay to ensure port is freed
    tokio::time::sleep(Duration::from_millis(500)).await;
    let _ = app.emit("backend-ready", false);
    spawn_sidecar(app).await;
    Ok(())
}

// ── App Entry ──

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_store::Builder::default().build())
        .manage(SidecarState(Mutex::new(None)))
        .invoke_handler(tauri::generate_handler![
            backend_status,
            get_api_key,
            set_api_key,
            restart_sidecar
        ])
        .setup(|app| {
            let app_handle = app.handle().clone();

            // ── System tray ──
            let show = MenuItemBuilder::with_id("show", "Show Window").build(app)?;
            let quit = MenuItemBuilder::with_id("quit", "Quit").build(app)?;
            let menu = MenuBuilder::new(app)
                .item(&show)
                .separator()
                .item(&quit)
                .build()?;

            let _tray = TrayIconBuilder::new()
                .icon(app.default_window_icon().cloned().expect("no app icon"))
                .tooltip("Resume Brain")
                .menu(&menu)
                .on_menu_event(move |app, event| match event.id().as_ref() {
                    "show" => {
                        if let Some(w) = app.get_webview_window("main") {
                            let _ = w.show();
                            let _ = w.set_focus();
                        }
                    }
                    "quit" => {
                        kill_sidecar(app);
                        app.exit(0);
                    }
                    _ => {}
                })
                .on_tray_icon_event(|tray, event| {
                    if let tauri::tray::TrayIconEvent::Click { .. } = event {
                        let app = tray.app_handle();
                        if let Some(w) = app.get_webview_window("main") {
                            let _ = w.show();
                            let _ = w.set_focus();
                        }
                    }
                })
                .build(app)?;

            // ── Spawn sidecar ──
            tauri::async_runtime::spawn(async move {
                spawn_sidecar(app_handle).await;
            });

            Ok(())
        })
        .on_window_event(|window, event| {
            // Minimize to tray on close instead of quitting
            if let tauri::WindowEvent::CloseRequested { api, .. } = event {
                let _ = window.hide();
                api.prevent_close();
            }
        })
        .run(tauri::generate_context!())
        .expect("Error while running Resume Brain");
}
