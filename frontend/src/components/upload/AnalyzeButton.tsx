import { useAnalysisStore } from '../../store/analysisStore';

export default function AnalyzeButton() {
  const { resumeFile, jobDescription, phase, startAnalysis, loadDemo, error } =
    useAnalysisStore();

  const canAnalyze = resumeFile && jobDescription.trim() && phase !== 'loading';

  return (
    <div className="space-y-3">
      <button
        onClick={startAnalysis}
        disabled={!canAnalyze}
        className={`
          w-full py-3 px-6 rounded-xl font-medium text-base transition-all duration-200
          ${canAnalyze
            ? 'bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-500 hover:to-purple-500 text-white shadow-lg shadow-blue-500/20'
            : 'bg-white/5 text-gray-500 cursor-not-allowed'
          }
        `}
      >
        {phase === 'loading' ? (
          <span className="flex items-center justify-center gap-2">
            <span className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
            Analyzing...
          </span>
        ) : (
          'Analyze Resume'
        )}
      </button>

      <button
        onClick={loadDemo}
        className="w-full py-2 px-4 rounded-lg text-sm text-gray-400 hover:text-white hover:bg-white/5 transition-colors"
      >
        Try Demo with Sample Data
      </button>

      {error && (
        <div className="text-red-400 text-sm text-center bg-red-500/10 rounded-lg py-2 px-3">
          {error}
        </div>
      )}
    </div>
  );
}
