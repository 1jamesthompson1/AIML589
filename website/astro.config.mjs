import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { defineConfig } from 'astro/config';
import react from '@astrojs/react';

const artifactsRoot = fileURLToPath(new URL('../artifacts/', import.meta.url));

/** Serve the repo's artifacts mirror directly during `astro dev`.
 *
 * Public-directory symlinks are convenient but are not reliable across
 * Astro/Vite versions and worktrees. The dev source of truth is the
 * repository-level `artifacts/` directory, so expose only its ft/ and bs/
 * prefixes with an explicit, traversal-safe middleware.
 */
function localArtifactsDevPlugin() {
  return {
    name: 'aiml589-local-artifacts',
    buildStart() {
      // Older checkouts may still have public/ft or public/bs symlinks. Astro
      // would follow them during a production build and accidentally upload
      // the artifact mirror, so fail with an actionable message instead.
      for (const name of ['ft', 'bs']) {
        const publicLink = new URL(`./public/${name}`, import.meta.url);
        try {
          if (fs.lstatSync(publicLink).isSymbolicLink()) {
            throw new Error(`Remove legacy website/public/${name} symlink before building; local dev data is served directly from artifacts/.`);
          }
        } catch (error) {
          if (error?.code !== 'ENOENT') throw error;
        }
      }
    },
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        if (!req.url || !['GET', 'HEAD'].includes(req.method ?? 'GET')) return next();

        let pathname;
        try {
          pathname = decodeURIComponent(new URL(req.url, 'http://localhost').pathname);
        } catch {
          return next();
        }
        if (!/^\/(?:ft|bs)(?:\/|$)/.test(pathname)) return next();

        const relative = pathname.replace(/^\/+/, '');
        let filePath = path.resolve(artifactsRoot, relative);
        const rootPrefix = `${path.resolve(artifactsRoot)}${path.sep}`;
        if (filePath !== path.resolve(artifactsRoot) && !filePath.startsWith(rootPrefix)) {
          return next();
        }

        try {
          if (fs.statSync(filePath).isDirectory()) filePath = path.join(filePath, 'index.json');
          const stat = fs.statSync(filePath);
          if (!stat.isFile()) return next();
          const contentType = {
            '.json': 'application/json; charset=utf-8',
            '.csv': 'text/csv; charset=utf-8',
            '.txt': 'text/plain; charset=utf-8',
            '.md': 'text/markdown; charset=utf-8',
            '.png': 'image/png',
            '.pdf': 'application/pdf',
          }[path.extname(filePath).toLowerCase()] ?? 'application/octet-stream';
          res.statusCode = 200;
          res.setHeader('Content-Type', contentType);
          // Local files are intentionally fresh while iterating locally.
          res.setHeader('Cache-Control', 'no-store');
          if (req.method === 'HEAD') {
            res.end();
            return;
          }
          fs.createReadStream(filePath).pipe(res);
        } catch {
          // Let Astro produce its normal 404/logging for missing local files.
          next();
        }
      });
    },
  };
}

export default defineConfig({
  integrations: [react()],
  vite: {
    plugins: [localArtifactsDevPlugin()],
  },
  site: 'https://nz-llm.sjhl.nz',
  outDir: './dist',
});
