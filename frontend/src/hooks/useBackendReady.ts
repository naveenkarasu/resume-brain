import { useState, useEffect } from 'react';
import { isTauri } from '../utils/platform';

/**
 * In Tauri, listens for the `backend-ready` event emitted by the Rust sidecar
 * manager. In web mode, returns `true` immediately (backend is external).
 */
export function useBackendReady(): boolean {
  const [ready, setReady] = useState(!isTauri());

  useEffect(() => {
    if (!isTauri()) return;

    let unlisten: (() => void) | undefined;

    (async () => {
      const { listen } = await import('@tauri-apps/api/event');
      unlisten = await listen<boolean>('backend-ready', (event) => {
        setReady(event.payload);
      });
    })();

    return () => {
      unlisten?.();
    };
  }, []);

  return ready;
}
