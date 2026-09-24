import type { Degradation } from '../api/types';
import { degradationNotice } from '../lib/degradation';
import { AlertIcon, InfoIcon } from './Icons';

/**
 * Never silent: whenever the broker degraded the answer (skipped the reranker,
 * shrank the dense beam, fell back to lexical, or lost shards) the user is told,
 * in words, what they are looking at.
 */
export function DegradationBanner({ degradation }: { degradation: Degradation }) {
  const notice = degradationNotice(degradation);
  if (!notice) return null;
  return (
    <div className={`banner banner--${notice.severity}`} data-testid="degradation-banner">
      {notice.severity === 'warning' ? <AlertIcon /> : <InfoIcon />}
      <div>
        <p className="banner__title">
          <span className="visually-hidden">{notice.severity === 'warning' ? 'Warning: ' : 'Note: '}</span>
          {notice.headline}
        </p>
        {notice.details.length > 0 && (
          <ul>
            {notice.details.map((d) => (
              <li key={d}>{d}</li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}
