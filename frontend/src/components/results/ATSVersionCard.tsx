import { useState } from 'react';

interface Props {
  atsResume: string;
  strengths: string[];
  weaknesses: string[];
}

export default function ATSVersionCard({ atsResume, strengths, weaknesses }: Props) {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    await navigator.clipboard.writeText(atsResume);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="bg-slate-900/80 border border-white/10 rounded-2xl p-4 sm:p-6 space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold text-white">ATS-Optimized Resume</h3>
        {atsResume && (
          <button
            onClick={handleCopy}
            className="px-3 py-1 text-xs rounded-lg bg-white/5 hover:bg-white/10 text-gray-400 hover:text-white transition-colors"
          >
            {copied ? 'Copied!' : 'Copy'}
          </button>
        )}
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {strengths.length > 0 && (
          <div>
            <div className="text-sm text-green-400 mb-2 font-medium">Strengths</div>
            <ul className="space-y-1">
              {strengths.map((s, i) => (
                <li key={i} className="text-sm text-gray-400 flex items-start gap-2">
                  <span className="text-green-500 mt-0.5 shrink-0">+</span>
                  {s}
                </li>
              ))}
            </ul>
          </div>
        )}

        {weaknesses.length > 0 && (
          <div>
            <div className="text-sm text-red-400 mb-2 font-medium">Areas to Improve</div>
            <ul className="space-y-1">
              {weaknesses.map((w, i) => (
                <li key={i} className="text-sm text-gray-400 flex items-start gap-2">
                  <span className="text-red-500 mt-0.5 shrink-0">-</span>
                  {w}
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>

      {atsResume && (
        <details className="group">
          <summary className="cursor-pointer text-sm text-blue-400 hover:text-blue-300 transition-colors">
            View optimized resume text
          </summary>
          <pre className="mt-3 p-4 bg-black/30 rounded-xl text-xs text-gray-300 whitespace-pre-wrap overflow-auto max-h-96 border border-white/5">
            {atsResume}
          </pre>
        </details>
      )}
    </div>
  );
}
