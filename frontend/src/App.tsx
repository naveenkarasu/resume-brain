import { useEffect, useRef } from 'react';
import Header from './components/layout/Header';
import Footer from './components/layout/Footer';
import BrainScene from './components/three/BrainScene';
import ResumeUploader from './components/upload/ResumeUploader';
import JobDescriptionInput from './components/upload/JobDescriptionInput';
import AnalyzeButton from './components/upload/AnalyzeButton';
import MatchScoreCard from './components/results/MatchScoreCard';
import MissingKeywordsCard from './components/results/MissingKeywordsCard';
import BulletRewritesCard from './components/results/BulletRewritesCard';
import ATSVersionCard from './components/results/ATSVersionCard';
import SettingsModal from './components/settings/SettingsModal';
import { useAnalysisStore } from './store/analysisStore';
import { useSettingsStore } from './store/settingsStore';
import { useBackendReady } from './hooks/useBackendReady';

function App() {
  const { phase, result, isDemo, loadingStage } = useAnalysisStore();
  const backendReady = useBackendReady();
  const loadApiKey = useSettingsStore((s) => s.loadApiKey);
  const contentRef = useRef<HTMLDivElement>(null);

  // Load API key on mount (desktop only — no-ops in web)
  useEffect(() => {
    loadApiKey();
  }, [loadApiKey]);

  // Drive content panel opacity + pointer-events from page scroll position
  useEffect(() => {
    const update = () => {
      const el = contentRef.current;
      if (!el) return;
      const progress = Math.min(window.scrollY / (window.innerHeight * 0.3), 1);
      el.style.opacity = String(progress);
      el.style.pointerEvents = progress > 0.8 ? 'auto' : 'none';
    };
    window.addEventListener('scroll', update, { passive: true });
    update();
    return () => window.removeEventListener('scroll', update);
  }, []);

  // Auto-scroll on phase transitions
  useEffect(() => {
    if (phase === 'loading' || phase === 'results') {
      window.scrollTo({ top: window.innerHeight * 0.3, behavior: 'smooth' });
    }
    if (phase === 'idle') {
      window.scrollTo({ top: 0, behavior: 'smooth' });
      if (contentRef.current) contentRef.current.scrollTop = 0;
    }
  }, [phase]);

  // Allow scrolling back to full globe view from content panel
  useEffect(() => {
    const el = contentRef.current;
    if (!el) return;
    const handleWheel = (e: WheelEvent) => {
      if (el.scrollTop <= 0 && e.deltaY < 0) {
        window.scrollBy(0, e.deltaY);
      }
    };
    el.addEventListener('wheel', handleWheel, { passive: true });
    return () => el.removeEventListener('wheel', handleWheel);
  }, []);

  return (
    <div className="relative">
      {/* ── Backend loading overlay (desktop sidecar startup) ── */}
      {!backendReady && (
        <div className="fixed inset-0 z-[200] flex items-center justify-center bg-[#0a0a1a]">
          <div className="text-center space-y-4">
            <div className="w-14 h-14 mx-auto rounded-xl bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center text-white font-bold text-xl">
              RB
            </div>
            <div className="text-white text-lg font-medium">Resume Brain</div>
            <div className="text-gray-400 text-sm animate-pulse">Starting backend...</div>
            <div className="flex justify-center gap-1.5 pt-2">
              <span className="w-2 h-2 bg-blue-400 rounded-full animate-bounce [animation-delay:0ms]" />
              <span className="w-2 h-2 bg-blue-400 rounded-full animate-bounce [animation-delay:150ms]" />
              <span className="w-2 h-2 bg-blue-400 rounded-full animate-bounce [animation-delay:300ms]" />
            </div>
          </div>
        </div>
      )}

      <SettingsModal />

      {/* ── Background layers (fixed, behind everything) ── */}
      <BrainScene />
      <div className="orb orb-1" />
      <div className="orb orb-2" />
      <div className="orb orb-3" />
      <div className="dot-grid" />
      <div className="noise-overlay" />

      <Header />

      {/* Scroll driver — provides page-scroll distance for the globe transition */}
      <div style={{ height: '130vh' }} />

      {/* ── Fixed content panel — confined to bottom 50vh, never covers the globe ── */}
      <div
        ref={contentRef}
        className="fixed inset-x-0 bottom-0 z-10 overflow-y-auto"
        style={{ top: '50vh', opacity: 0, pointerEvents: 'none' }}
      >
        <div className="glow-divider" />

        {/* Loading indicator */}
        {phase === 'loading' && (
          <div className="flex items-center justify-center py-12">
            <div className="text-center space-y-3">
              <div className="text-blue-400 text-base sm:text-lg font-medium animate-pulse">
                {loadingStage || 'Analyzing your resume...'}
              </div>
              <div className="flex justify-center gap-1.5">
                <span className="w-2 h-2 bg-blue-400 rounded-full animate-bounce [animation-delay:0ms]" />
                <span className="w-2 h-2 bg-blue-400 rounded-full animate-bounce [animation-delay:150ms]" />
                <span className="w-2 h-2 bg-blue-400 rounded-full animate-bounce [animation-delay:300ms]" />
              </div>
            </div>
          </div>
        )}

        {/* Upload Section */}
        {phase !== 'results' && phase !== 'loading' && (
          <section className="max-w-2xl mx-auto px-4 sm:px-6 py-6 sm:py-8 space-y-4 sm:space-y-6">
            <div className="text-center mb-6 sm:mb-8">
              <h1 className="text-2xl sm:text-3xl md:text-4xl font-bold text-white mb-2 sm:mb-3">
                AI Resume Analyzer
              </h1>
              <p className="text-gray-400 text-sm sm:text-base px-2">
                Upload your resume and paste a job description to get an AI-powered match analysis
              </p>
            </div>
            <ResumeUploader />
            <JobDescriptionInput />
            <AnalyzeButton />
          </section>
        )}

        {/* Results Section */}
        {phase === 'results' && result && (
          <section className="max-w-4xl mx-auto px-4 sm:px-6 py-6 sm:py-8 space-y-4 sm:space-y-6">
            {isDemo && (
              <div className="text-center text-sm text-amber-400 bg-amber-500/10 rounded-lg py-2 animate-fade-in">
                Demo mode - showing sample analysis results
              </div>
            )}
            {result.degraded && (
              <div className="text-center text-sm text-amber-400 bg-amber-500/10 rounded-lg py-2 animate-fade-in">
                Partial results - AI service unavailable, using local keyword matching
              </div>
            )}

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 sm:gap-6">
              <div className="animate-reveal [animation-delay:0ms]">
                <MatchScoreCard
                  overallScore={result.overall_score}
                  breakdown={result.score_breakdown}
                  summary={result.summary}
                  tfidfScore={result.tfidf_score}
                  semanticScore={result.semantic_score}
                  scoringMethod={result.scoring_method}
                  sectionAnalysis={result.section_analysis}
                  experienceYears={result.experience_years}
                  educationLevel={result.education_level}
                />
              </div>
              <div className="animate-reveal [animation-delay:150ms]">
                <MissingKeywordsCard
                  matched={result.matched_keywords}
                  missing={result.missing_keywords}
                  keywordDensity={result.keyword_density}
                />
              </div>
            </div>

            <div className="animate-reveal [animation-delay:300ms]">
              <BulletRewritesCard
                rewrites={result.bullet_rewrites}
                bulletScores={result.bullet_scores}
              />
            </div>

            <div className="animate-reveal [animation-delay:450ms]">
              <ATSVersionCard
                atsResume={result.ats_optimized_resume}
                strengths={result.strengths}
                weaknesses={result.weaknesses}
              />
            </div>
          </section>
        )}

        <Footer />
      </div>
    </div>
  );
}

export default App;
