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
import { useAnalysisStore } from './store/analysisStore';

function App() {
  const { phase, result, isDemo } = useAnalysisStore();

  return (
    <div className="min-h-screen flex flex-col">
      <Header />

      <main className="flex-1 pt-20">
        {/* Brain Visualization */}
        <section className="relative">
          <BrainScene />
          <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
            {phase === 'loading' && (
              <div className="text-center">
                <div className="text-blue-400 text-lg font-medium animate-pulse">
                  Analyzing your resume...
                </div>
              </div>
            )}
          </div>
        </section>

        {/* Upload Section */}
        {phase !== 'results' && (
          <section className="max-w-2xl mx-auto px-6 pb-12 space-y-6">
            <div className="text-center mb-8">
              <h1 className="text-3xl md:text-4xl font-bold text-white mb-3">
                AI Resume Analyzer
              </h1>
              <p className="text-gray-400">
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
          <section className="max-w-4xl mx-auto px-6 pb-12 space-y-6">
            {isDemo && (
              <div className="text-center text-sm text-amber-400 bg-amber-500/10 rounded-lg py-2">
                Demo mode - showing sample analysis results
              </div>
            )}
            {result.degraded && (
              <div className="text-center text-sm text-amber-400 bg-amber-500/10 rounded-lg py-2">
                Partial results - AI service unavailable, using local keyword matching
              </div>
            )}

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
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
              <MissingKeywordsCard
                matched={result.matched_keywords}
                missing={result.missing_keywords}
                keywordDensity={result.keyword_density}
              />
            </div>

            <BulletRewritesCard
              rewrites={result.bullet_rewrites}
              bulletScores={result.bullet_scores}
            />

            <ATSVersionCard
              atsResume={result.ats_optimized_resume}
              strengths={result.strengths}
              weaknesses={result.weaknesses}
            />
          </section>
        )}
      </main>

      <Footer />
    </div>
  );
}

export default App;
