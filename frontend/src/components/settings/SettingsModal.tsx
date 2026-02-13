import { useState } from 'react';
import { useSettingsStore } from '../../store/settingsStore';

export default function SettingsModal() {
  const { apiKey, showSettings, closeSettings, saveApiKey } = useSettingsStore();
  const [localKey, setLocalKey] = useState(apiKey);
  const [showKey, setShowKey] = useState(false);
  const [saving, setSaving] = useState(false);

  if (!showSettings) return null;

  const handleSave = async () => {
    setSaving(true);
    await saveApiKey(localKey.trim());
    setSaving(false);
    closeSettings();
  };

  return (
    <div className="fixed inset-0 z-[100] flex items-center justify-center">
      {/* Backdrop */}
      <div className="absolute inset-0 bg-black/60 backdrop-blur-sm" onClick={closeSettings} />

      {/* Modal */}
      <div className="relative w-full max-w-md mx-4 bg-white/5 backdrop-blur-md border border-white/10 rounded-2xl p-6 shadow-2xl">
        <h2 className="text-xl font-semibold text-white mb-1">Settings</h2>
        <p className="text-sm text-gray-400 mb-6">
          Configure your Gemini API key for AI-powered analysis.
        </p>

        {/* API Key input */}
        <label className="block text-sm font-medium text-gray-300 mb-2">
          Gemini API Key
        </label>
        <div className="relative mb-2">
          <input
            type={showKey ? 'text' : 'password'}
            value={localKey}
            onChange={(e) => setLocalKey(e.target.value)}
            placeholder="AIza..."
            className="w-full px-4 py-2.5 rounded-lg bg-white/5 border border-white/10 text-white placeholder-gray-500 focus:outline-none focus:border-blue-500/50 focus:ring-1 focus:ring-blue-500/30 pr-16 text-sm"
          />
          <button
            type="button"
            onClick={() => setShowKey(!showKey)}
            className="absolute right-2 top-1/2 -translate-y-1/2 px-2 py-1 text-xs text-gray-400 hover:text-white transition-colors"
          >
            {showKey ? 'Hide' : 'Show'}
          </button>
        </div>

        <a
          href="https://aistudio.google.com/apikey"
          target="_blank"
          rel="noopener noreferrer"
          className="inline-block text-xs text-blue-400 hover:text-blue-300 transition-colors mb-4"
        >
          Get a free API key from Google AI Studio &rarr;
        </a>

        <p className="text-xs text-gray-500 mb-6">
          The backend will restart when you save a new key. Analysis works without a key using local keyword matching only.
        </p>

        {/* Actions */}
        <div className="flex gap-3 justify-end">
          <button
            onClick={closeSettings}
            className="px-4 py-2 text-sm rounded-lg bg-white/5 hover:bg-white/10 text-gray-300 transition-colors"
          >
            Cancel
          </button>
          <button
            onClick={handleSave}
            disabled={saving}
            className="px-4 py-2 text-sm rounded-lg bg-blue-600 hover:bg-blue-500 text-white transition-colors disabled:opacity-50"
          >
            {saving ? 'Saving...' : 'Save'}
          </button>
        </div>
      </div>
    </div>
  );
}
