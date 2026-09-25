import {
  DEFAULT_BUCKET,
  bucketUrl,
  isLocalSourceUrl,
  sourceLabel,
  sourceSwitchHref,
} from '../lib/dataSource';

interface Props {
  bucket?: string;
  manifestUrl: string;
  kind: string;
}

export default function DataSourceNotice({ bucket = DEFAULT_BUCKET, manifestUrl, kind }: Props) {
  const local = isLocalSourceUrl(manifestUrl);
  const switchHref = sourceSwitchHref(!local);
  const otherLabel = local ? 'bucket' : 'local mirror';

  return (
    <div className="data-source-notice" aria-label={`${kind} data source`}>
      <div className="data-source-notice__main">
        <strong>Data source:</strong>{' '}
        <span className={local ? 'data-source-notice__local' : 'data-source-notice__remote'}>
          {sourceLabel(manifestUrl)}
        </span>
        <span className="data-source-notice__hint">
          {local
            ? 'Files are fetched directly from the repository artifacts/ directory and are not browser-cached.'
            : 'Runtime data is fetched from the public storage bucket.'}
        </span>
      </div>
      <div className="data-source-notice__links">
        <a href={bucketUrl(bucket)} target="_blank" rel="noopener noreferrer">Browse bucket ↗</a>
        {switchHref && <a href={switchHref}>Use {otherLabel}</a>}
      </div>
      <code className="data-source-notice__url">{manifestUrl}</code>
    </div>
  );
}
