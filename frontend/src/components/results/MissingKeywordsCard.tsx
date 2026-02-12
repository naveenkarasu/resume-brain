interface Props {
  matched: string[];
  missing: string[];
  keywordDensity: Record<string, number>;
}

function DensityIndicator({ density }: { density: number }) {
  const isOptimal = density >= 1 && density <= 3;
  const isLow = density > 0 && density < 1;
  const color = isOptimal
    ? 'bg-green-500'
    : isLow
      ? 'bg-yellow-500'
      : density === 0
        ? 'bg-red-500/30'
        : 'bg-orange-500';
  const width = Math.min(density * 33, 100); // 3% = full bar

  return (
    <div className="flex items-center gap-2 min-w-0">
      <div className="h-1.5 flex-1 bg-white/5 rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full ${color}`}
          style={{ width: `${width}%` }}
        />
      </div>
      <span className="text-[10px] text-gray-500 w-10 text-right shrink-0">
        {density > 0 ? `${density}%` : '—'}
      </span>
    </div>
  );
}

export default function MissingKeywordsCard({ matched, missing, keywordDensity }: Props) {
  const hasDensity = Object.keys(keywordDensity).length > 0;

  return (
    <div className="bg-slate-900/80 border border-white/10 rounded-2xl p-4 sm:p-6 space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold text-white">Keywords Analysis</h3>
        {hasDensity && (
          <span className="text-[10px] text-gray-600">Optimal density: 1-3%</span>
        )}
      </div>

      {missing.length > 0 && (
        <div>
          <div className="text-sm text-red-400 mb-2 font-medium">
            Missing ({missing.length})
          </div>
          <div className="space-y-1.5">
            {missing.map((kw) => (
              <div key={kw} className="flex items-center gap-2">
                <span className="px-3 py-1 text-xs rounded-full bg-red-500/10 text-red-400 border border-red-500/20 shrink-0">
                  {kw}
                </span>
                {hasDensity && (
                  <div className="flex-1 min-w-0">
                    <DensityIndicator density={keywordDensity[kw] ?? 0} />
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {matched.length > 0 && (
        <div>
          <div className="text-sm text-green-400 mb-2 font-medium">
            Matched ({matched.length})
          </div>
          <div className="space-y-1.5">
            {matched.map((kw) => (
              <div key={kw} className="flex items-center gap-2">
                <span className="px-3 py-1 text-xs rounded-full bg-green-500/10 text-green-400 border border-green-500/20 shrink-0">
                  {kw}
                </span>
                {hasDensity && (
                  <div className="flex-1 min-w-0">
                    <DensityIndicator density={keywordDensity[kw] ?? 0} />
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
