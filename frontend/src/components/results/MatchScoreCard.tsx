import { useState, useEffect } from 'react';
import type { ScoreBreakdown, SectionAnalysis } from '../../api/types';

function scoreColor(score: number): string {
  if (score >= 85) return 'text-green-400';
  if (score >= 70) return 'text-lime-400';
  if (score >= 55) return 'text-yellow-400';
  if (score >= 40) return 'text-orange-400';
  return 'text-red-400';
}

function ScoreBar({ label, value }: { label: string; value: number }) {
  const color =
    value >= 85
      ? 'bg-green-500'
      : value >= 70
        ? 'bg-lime-500'
        : value >= 55
          ? 'bg-yellow-500'
          : value >= 40
            ? 'bg-orange-500'
            : 'bg-red-500';

  return (
    <div className="space-y-1">
      <div className="flex justify-between text-sm">
        <span className="text-gray-400">{label}</span>
        <span className={scoreColor(value)}>{value}%</span>
      </div>
      <div className="h-2 bg-white/5 rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-1000 ease-out ${color}`}
          style={{ width: `${value}%` }}
        />
      </div>
    </div>
  );
}

const METHOD_LABELS: Record<string, string> = {
  hybrid: 'Hybrid (LLM + NLP)',
  local_only: 'Local NLP Only',
  llm_only: 'LLM Only',
};

const EDUCATION_LABELS: Record<string, string> = {
  phd: 'Ph.D.',
  masters: "Master's",
  bachelors: "Bachelor's",
  associate: "Associate's",
};

interface Props {
  overallScore: number;
  breakdown: ScoreBreakdown;
  summary: string;
  tfidfScore: number;
  semanticScore: number;
  scoringMethod: string;
  sectionAnalysis: SectionAnalysis;
  experienceYears: number;
  educationLevel: string;
}

export default function MatchScoreCard({
  overallScore,
  breakdown,
  summary,
  tfidfScore,
  semanticScore,
  scoringMethod,
  sectionAnalysis,
  experienceYears,
  educationLevel,
}: Props) {
  const [displayScore, setDisplayScore] = useState(0);

  useEffect(() => {
    const duration = 1500;
    const startTime = performance.now();
    let raf: number;
    function animate(time: number) {
      const progress = Math.min((time - startTime) / duration, 1);
      const eased = 1 - Math.pow(1 - progress, 3);
      setDisplayScore(Math.round(eased * overallScore));
      if (progress < 1) raf = requestAnimationFrame(animate);
    }
    raf = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(raf);
  }, [overallScore]);

  return (
    <div className="bg-slate-900/80 border border-white/10 rounded-2xl p-4 sm:p-6 space-y-4 sm:space-y-6">
      <div className="text-center">
        <div className={`text-5xl sm:text-6xl font-bold tabular-nums ${scoreColor(overallScore)}`}>
          {displayScore}
        </div>
        <div className="text-gray-500 text-sm mt-1">Match Score</div>
        <div className="text-gray-600 text-xs mt-0.5">
          {METHOD_LABELS[scoringMethod] ?? scoringMethod}
        </div>
      </div>

      <div className="space-y-3">
        <ScoreBar label="Skills" value={breakdown.skills_match} />
        <ScoreBar label="Experience" value={breakdown.experience_match} />
        <ScoreBar label="Education" value={breakdown.education_match} />
        <ScoreBar label="Keywords" value={breakdown.keywords_match} />
      </div>

      {/* NLP Similarity Scores */}
      <div className="border-t border-white/5 pt-4 space-y-2">
        <div className="text-xs text-gray-500 uppercase tracking-wider mb-2">NLP Analysis</div>
        <div className="grid grid-cols-2 gap-3">
          <div className="bg-white/[0.02] rounded-lg p-3 text-center">
            <div className="text-lg font-semibold text-blue-400">
              {Math.round(tfidfScore * 100)}%
            </div>
            <div className="text-xs text-gray-500">TF-IDF Match</div>
          </div>
          <div className="bg-white/[0.02] rounded-lg p-3 text-center">
            <div className="text-lg font-semibold text-purple-400">
              {Math.round(semanticScore * 100)}%
            </div>
            <div className="text-xs text-gray-500">Semantic Match</div>
          </div>
        </div>
        {(experienceYears > 0 || educationLevel) && (
          <div className="grid grid-cols-2 gap-3 mt-2">
            {experienceYears > 0 && (
              <div className="bg-white/[0.02] rounded-lg p-3 text-center">
                <div className="text-lg font-semibold text-cyan-400">
                  {experienceYears}
                </div>
                <div className="text-xs text-gray-500">Years Experience</div>
              </div>
            )}
            {educationLevel && (
              <div className="bg-white/[0.02] rounded-lg p-3 text-center">
                <div className="text-lg font-semibold text-emerald-400">
                  {EDUCATION_LABELS[educationLevel] ?? educationLevel}
                </div>
                <div className="text-xs text-gray-500">Education</div>
              </div>
            )}
          </div>
        )}
        {sectionAnalysis.detected_sections.length > 0 && (
          <div className="flex items-center gap-2 text-xs text-gray-500 mt-2">
            <span>Sections:</span>
            <div className="flex flex-wrap gap-1">
              {sectionAnalysis.detected_sections.map((s) => (
                <span key={s} className="bg-white/5 px-1.5 py-0.5 rounded capitalize">
                  {s}
                </span>
              ))}
            </div>
          </div>
        )}
      </div>

      {summary && (
        <p className="text-gray-400 text-sm leading-relaxed border-t border-white/5 pt-4">
          {summary}
        </p>
      )}
    </div>
  );
}
