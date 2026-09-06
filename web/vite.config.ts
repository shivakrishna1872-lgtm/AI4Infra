import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// Freebuff: HMR must stay disabled; do not enable it here.
export default defineConfig({
  plugins: [react()],
  build: {
    outDir: "../web-dist",
    emptyOutDir: true,
  },
  server: {
    hmr: false,
    proxy: {
      "/api": "http://127.0.0.1:8766",
    },
  },
});