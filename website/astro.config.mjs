import { defineConfig } from 'astro/config';
import react from '@astrojs/react';

export default defineConfig({
  integrations: [react()],
  site: 'https://nz-value-llm.sjhl.nz',
  outDir: './dist',
  vite: {
    ssr: {
      noExternal: ['papaparse'],
    },
  },
});
