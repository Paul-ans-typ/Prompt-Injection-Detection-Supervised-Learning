import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    proxy: {
      // Proxy all API calls to FastAPI during development
      '/ab':                  'http://localhost:8000',
      '/detect':              'http://localhost:8000',
      '/available-detectors': 'http://localhost:8000',
      '/available-llms':      'http://localhost:8000',
      '/health':              'http://localhost:8000',
      '/results':             'http://localhost:8000',
      '/sessions':            'http://localhost:8000',
      '/v1':                  'http://localhost:8000',
    },
  },
  build: {
    outDir: 'dist',
  },
})
