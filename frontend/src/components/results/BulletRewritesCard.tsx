import type { BulletRewrite, BulletScore } from '../../api/types';

function QualityBadge({ label, active }: { label: string; active: boolean }) {
  return (
    <span
      className={`px-1.5 py-0.5 text-[10px] rounded ${
        active
          ? 'bg-green-500/15 text-green-400 border border-green-500/20'
          : 'bg-white/5 text-gray-600'
      }`}
    >
      {label}
    </span>
  );
}

function scoreColor(score: number): string {
  if (score >= 80) return 'text-green-400';
  if (score >= 50) return 'text-yellow-400';
  return 'text-red-400';
}

interface Props {
  rewrites: BulletRewrite[];
  bulletScores?: BulletScore[];
}

export default function BulletRewritesCard({ rewrites, bulletScores }: Props) {
  const hasContent = rewrites.length > 0 || (bulletScores && bulletScores.length > 0);
  if (!hasContent) return null;

  return (
    <div className="bg-slate-900/80 border border-white/10 rounded-2xl p-4 sm:p-6 space-y-4">
      {/* Bullet Quality Scores */}
      {bulletScores && bulletScores.length > 0 && (
        <div>
          <h3 className="text-lg font-semibold text-white mb-3">Bullet Quality Analysis</h3>
          <div className="space-y-3">
            {bulletScores.map((bs, i) => (
              <div key={i} className="bg-white/[0.02] rounded-lg p-3 space-y-2">
                <div className="flex items-start justify-between gap-3">
                  <span className="text-sm text-gray-300 flex-1">{bs.text}</span>
                  <span className={`text-sm font-semibold shrink-0 ${scoreColor(bs.quality_score)}`}>
                    {bs.quality_score}/100
                  </span>
                </div>
                <div className="flex flex-wrap gap-1.5">
                  <QualityBadge label="Action Verb" active={bs.has_action_verb} />
                  <QualityBadge label="Metrics" active={bs.has_metrics} />
                  <QualityBadge label="Good Length" active={bs.length_ok} />
                  {bs.keyword_count > 0 && (
                    <QualityBadge label={`${bs.keyword_count} Keywords`} active={true} />
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Bullet Rewrites */}
      {rewrites.length > 0 && (
        <div>
          <h3 className="text-lg font-semibold text-white mb-3">Bullet Rewrites</h3>
          <div className="space-y-4">
            {rewrites.map((rw, i) => (
              <div key={i} className="space-y-2 border-b border-white/5 pb-4 last:border-0 last:pb-0">
                <div className="flex items-start gap-2">
                  <span className="text-red-400 text-xs mt-1 shrink-0">BEFORE</span>
                  <span className="text-gray-500 text-sm line-through">{rw.original}</span>
                </div>
                <div className="flex items-start gap-2">
                  <span className="text-green-400 text-xs mt-1 shrink-0">AFTER</span>
                  <span className="text-gray-200 text-sm">{rw.rewritten}</span>
                </div>
                <div className="text-xs text-gray-600 italic pl-14">{rw.reason}</div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
