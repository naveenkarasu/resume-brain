use std::sync::Mutex;
use std::time::Duration;
use tauri::Manager;
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
    eprintln!("[sidecar] Backend failed to start after {} retries", max_retries);
    false
}

#[tauri::command]
async fn backend_status() -> Result<String, String> {
    if check_backend_health().await {
        Ok("healthy".to_string())
    } else {
        Err("Backend is not responding".to_string())
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .manage(SidecarState(Mutex::new(None)))
        .invoke_handler(tauri::generate_handler![backend_status])
        .setup(|app| {
            let app_handle = app.handle().clone();

            // Spawn sidecar in a background task
            tauri::async_runtime::spawn(async move {
                println!("[sidecar] Starting resume-brain backend...");

                // Launch the sidecar binary
                let shell = app_handle.shell();
                let sidecar_command = shell.sidecar("binaries/resume-brain")
                    .expect("Failed to create sidecar command");

                match sidecar_command.spawn() {
                    Ok((mut rx, child)) => {
                        // Store the child process for cleanup
                        let state = app_handle.state::<SidecarState>();
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
                            // Emit event to frontend that backend is ready
                            let _ = app_handle.emit("backend-ready", true);
                        } else {
                            eprintln!("[sidecar] Backend failed to start!");
                            let _ = app_handle.emit("backend-ready", false);
                        }
                    }
                    Err(e) => {
                        eprintln!("[sidecar] Failed to spawn: {}", e);
                        let _ = app_handle.emit("backend-ready", false);
                    }
                }
            });

            Ok(())
        })
        .on_window_event(|window, event| {
            // Kill sidecar when the window closes
            if let tauri::WindowEvent::Destroyed = event {
                let state = window.state::<SidecarState>();
                if let Some(child) = state.0.lock().unwrap().take() {
                    println!("[sidecar] Killing backend process...");
                    let _ = child.kill();
                    println!("[sidecar] Backend process killed.");
                }
            }
        })
        .run(tauri::generate_context!())
        .expect("Error while running Resume Brain");
}
