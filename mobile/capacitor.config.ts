import type { CapacitorConfig } from '@capacitor/cli';

const config: CapacitorConfig = {
  appId: 'com.resumebrain.app',
  appName: 'Resume Brain',
  webDir: '../frontend/dist',
  server: {
    // In production, the frontend is served from the webDir.
    // For development, point to the Vite dev server:
    // url: 'http://10.0.2.2:5173',  // Android emulator -> host machine
    androidScheme: 'https',
  },
  plugins: {
    SplashScreen: {
      launchShowDuration: 2000,
      backgroundColor: '#0f172a',
      showSpinner: true,
      spinnerColor: '#3b82f6',
    },
  },
  // NOTE: Android cannot run the PyInstaller sidecar.
  // Options for offline ML on Android:
  //   1. Chaquopy - embed Python interpreter in Android app
  //   2. Convert models to ONNX/TFLite and use native inference
  //   3. Use a remote API endpoint as fallback
  // See README for details on each approach.
};

export default config;
