import { create } from 'zustand';
import { isTauri } from '../utils/platform';

interface SettingsState {
  apiKey: string;
  showSettings: boolean;

  openSettings: () => void;
  closeSettings: () => void;
  loadApiKey: () => Promise<void>;
  saveApiKey: (key: string) => Promise<void>;
}

export const useSettingsStore = create<SettingsState>((set) => ({
  apiKey: '',
  showSettings: false,

  openSettings: () => set({ showSettings: true }),
  closeSettings: () => set({ showSettings: false }),

  loadApiKey: async () => {
    if (!isTauri()) return;
    try {
      const { invoke } = await import('@tauri-apps/api/core');
      const key = await invoke<string>('get_api_key');
      set({ apiKey: key });
    } catch {
      // No key stored yet — that's fine
    }
  },

  saveApiKey: async (key: string) => {
    set({ apiKey: key });
    if (!isTauri()) return;
    try {
      const { invoke } = await import('@tauri-apps/api/core');
      await invoke('set_api_key', { key });
      await invoke('restart_sidecar');
    } catch (e) {
      console.error('Failed to save API key:', e);
    }
  },
}));
