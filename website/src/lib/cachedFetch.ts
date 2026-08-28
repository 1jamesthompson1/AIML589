/* Cached fetches for the HF bucket.
 *
 * The bucket's CDN sends no cache-control headers, so the browser re-
 * downloads every file on each visit. We add our own small persistent cache
 * (localStorage with a TTL) to make repeat views instant. Files larger than
 * a few hundred KB are not cached (localStorage quota).
 *
 * Two robustness rules matter here:
 * - the bucket's `resolve` endpoint redirects to a CDN URL signed for ~1h;
 *   we fetch with `cache: 'no-store'` so the browser never reuses an
 *   expired signed redirect (which the CDN answers with 403),
 * - transient failures (429/5xx/network) are retried with backoff.
 */

const CACHE_TTL_MS = 60 * 60 * 1000; // 1 hour
const MAX_CACHE_BYTES = 512 * 1024;  // 512 KB per entry
const MAX_ATTEMPTS = 3;

function cacheKey(url: string): string {
  return `hf-bucket:${url}`;
}

function isRetryable(err: Error): boolean {
  const msg = err.message;
  if (/^5\d\d/.test(msg)) return true;     // server errors
  if (/^429/.test(msg)) return true;       // rate limited
  if (!/^\d{3}/.test(msg)) return true;    // network / non-HTTP errors
  return false;
}

async function fetchWithRetry(url: string): Promise<string> {
  let lastErr: Error | null = null;
  for (let attempt = 0; attempt < MAX_ATTEMPTS; attempt++) {
    try {
      // no-store: never reuse a cached redirect, so the signed CDN URL in
      // the Location header is always fresh (they expire after ~1h).
      const res = await fetch(url, { cache: 'no-store' });
      if (!res.ok) throw new Error(`${res.status} ${res.statusText} (${url})`);
      return await res.text();
    } catch (err) {
      lastErr = err as Error;
      if (!isRetryable(lastErr)) throw lastErr;
      console.warn(`[cachedFetch] attempt ${attempt + 1}/${MAX_ATTEMPTS} failed for ${url}: ${lastErr.message}; retrying…`);
      await new Promise((r) => setTimeout(r, 500 * 2 ** attempt));
    }
  }
  console.error(`[cachedFetch] giving up on ${url}: ${lastErr?.message ?? 'unknown error'}`);
  throw lastErr ?? new Error(`fetch failed for ${url}`);
}

export function cachedFetchText(url: string, ttlMs = CACHE_TTL_MS): Promise<string> {
  try {
    const raw = localStorage.getItem(cacheKey(url));
    if (raw) {
      const entry = JSON.parse(raw);
      if (entry.url === url && Date.now() - entry.at < ttlMs) {
        console.info(`[cachedFetch] cache hit: ${url}`);
        return Promise.resolve(entry.text);
      }
      localStorage.removeItem(cacheKey(url));
    }
  } catch { /* corrupted/denied storage -> treat as miss */ }

  console.info(`[cachedFetch] fetch: ${url}`);
  return fetchWithRetry(url).then((text) => {
    try {
      if (text.length <= MAX_CACHE_BYTES) {
        localStorage.setItem(cacheKey(url), JSON.stringify({ url, at: Date.now(), text }));
      } else {
        console.info(`[cachedFetch] too large to cache (${text.length}B > ${MAX_CACHE_BYTES}B): ${url}`);
      }
    } catch (e) {
      console.warn(`[cachedFetch] failed to cache ${url}:`, e);
    }
    return text;
  });
}

export async function cachedFetchJson<T>(url: string, ttlMs?: number): Promise<T> {
  return JSON.parse(await cachedFetchText(url, ttlMs)) as T;
}