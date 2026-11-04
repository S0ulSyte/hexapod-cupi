import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';

// Serve the approved robot assets from their canonical directory.
export default defineConfig({
  plugins: [react()],
  publicDir: path.resolve(__dirname, '../robot'),
  // The Runs and Scene pages open ?run= / ?scene= URLs via /@fs; the viewer's own
  // package-lock makes Vite stop its workspace search here, so allow the whole
  // checkout, where run files live in the gitignored tmp/.
  server: { fs: { allow: [path.resolve(__dirname, '..')] } },
});
