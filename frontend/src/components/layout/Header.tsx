import { useAnalysisStore } from '../../store/analysisStore';

export default function Header() {
  const { phase, reset } = useAnalysisStore();

  return (
    <header className="fixed top-0 left-0 right-0 z-50 bg-[#0a0a1a]/80 backdrop-blur-md border-b border-white/5">
      <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
        <button onClick={reset} className="flex items-center gap-3 hover:opacity-80 transition-opacity">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center text-white font-bold text-sm">
            RB
          </div>
          <span className="text-lg font-semibold text-white">Resume Brain</span>
        </button>

        <div className="flex items-center gap-4">
          {phase === 'results' && (
            <button
              onClick={reset}
              className="px-4 py-2 text-sm rounded-lg bg-white/5 hover:bg-white/10 text-gray-300 transition-colors"
            >
              New Analysis
            </button>
          )}
          <a
            href="https://github.com"
            target="_blank"
            rel="noopener noreferrer"
            className="text-gray-400 hover:text-white transition-colors text-sm"
          >
            GitHub
          </a>
        </div>
      </div>
    </header>
  );
}
