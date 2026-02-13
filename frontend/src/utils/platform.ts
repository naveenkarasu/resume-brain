/**
 * Detect if the app is running inside a Tauri desktop shell.
 */
export function isTauri(): boolean {
  return typeof window !== 'undefined' && '__TAURI_INTERNALS__' in window;
}

/**
 * Return the correct API base URL for the current platform.
 * Desktop always hits the local sidecar; web uses the env var or proxied /api.
 */
export function getApiBaseUrl(): string {
  if (isTauri()) {
    return 'http://localhost:8000';
  }
  return import.meta.env.VITE_API_URL || '/api';
}
