/* Shared data-source configuration for the runtime result viewers.
 *
 * Development defaults to the artifacts mirror exposed by the Astro dev
 * middleware. A query parameter can override it when comparing local and
 * bucket data: `?local=0` selects the public HF bucket and `?local=1`
 * selects local files.
 */

export const DEFAULT_BUCKET = '1jamesthompson1/wvs-nz-value-alignment-evals';
export const LOCAL_FT_BASE = '/ft/evals/';
export const LOCAL_BS_BASE = '/bs/runs/';
export const LOCAL_COMPARISONS_BASE = '/bs/comparisons/';

export const urlParams = typeof window !== 'undefined'
  ? new URLSearchParams(window.location.search)
  : new URLSearchParams();

export const IS_DEV = import.meta.env.DEV;
const localParam = urlParams.get('local');
export const LOCAL_MODE = localParam === '1' || (localParam !== '0' && IS_DEV);

export function manifestUrl(localBase: string, remotePath: string): string {
  const custom = urlParams.get('manifest');
  if (custom) {
    // Older links used the directory itself (for example `/ft/evals/`).
    // Treat that as the conventional index manifest so it cannot 404.
    if (LOCAL_MODE && /^\/ft\/evals\/?$/.test(custom)) return '/ft/evals/index.json';
    if (LOCAL_MODE && /^\/bs\/runs\/?$/.test(custom)) return '/bs/runs/index.json';
    return custom;
  }
  if (LOCAL_MODE) {
    const base = localBase.endsWith('/') ? localBase : `${localBase}/`;
    return `${base}index.json`;
  }
  return `https://huggingface.co/buckets/${DEFAULT_BUCKET}/resolve/${remotePath}`;
}

export function comparisonsBaseUrl(manifestUrlValue: string): string {
  if (isLocalSourceUrl(manifestUrlValue)) return LOCAL_COMPARISONS_BASE;
  // The manifest is normally at bs/runs/index.json. Keep custom manifest
  // URLs useful by deriving the sibling comparisons directory from them.
  return manifestUrlValue.replace(/bs\/runs\/?.*$/, 'bs/comparisons/');
}

export function sourceSwitchHref(local: boolean): string {
  if (typeof window === 'undefined') return '';
  const url = new URL(window.location.href);
  if (local) {
    url.searchParams.set('local', '1');
    url.searchParams.delete('manifest');
  } else {
    url.searchParams.set('local', '0');
    url.searchParams.delete('manifest');
  }
  return `${url.pathname}${url.search}${url.hash}`;
}

export function isLocalSourceUrl(url: string): boolean {
  return url.startsWith('/') && !url.startsWith('//');
}

export function sourceLabel(manifestUrl?: string): string {
  if (manifestUrl ? isLocalSourceUrl(manifestUrl) : LOCAL_MODE) return 'Local artifacts mirror';
  if (urlParams.get('manifest')) return 'Custom manifest';
  return 'Public HF storage bucket';
}

export function bucketUrl(bucket = DEFAULT_BUCKET): string {
  return `https://huggingface.co/buckets/${bucket}`;
}
