import { create } from 'zustand';
import type { AnalysisResponse } from '../api/types';
import { analyzeResume, analyzeQuick } from '../api/client';
import { DEMO_RESULT } from '../utils/demo-data';

export type AppPhase = 'idle' | 'loading' | 'results';

interface AnalysisState {
  phase: AppPhase;
  result: AnalysisResponse | null;
  error: string | null;
  resumeFile: File | null;
  jobDescription: string;
  isDemo: boolean;
  loadingStage: string;

  setResumeFile: (file: File | null) => void;
  setJobDescription: (jd: string) => void;
  startAnalysis: () => Promise<void>;
  startQuickAnalysis: (resumeText: string) => Promise<void>;
  loadDemo: () => void;
  reset: () => void;
}

export const useAnalysisStore = create<AnalysisState>((set, get) => ({
  phase: 'idle',
  result: null,
  error: null,
  resumeFile: null,
  jobDescription: '',
  isDemo: false,
  loadingStage: '',

  setResumeFile: (file) => set({ resumeFile: file, error: null }),
  setJobDescription: (jd) => set({ jobDescription: jd, error: null }),

  startAnalysis: async () => {
    const { resumeFile, jobDescription } = get();
    if (!resumeFile || !jobDescription.trim()) {
      set({ error: 'Please upload a resume and enter a job description' });
      return;
    }

    set({ phase: 'loading', error: null, loadingStage: 'Parsing resume...' });
    const timers = [
      setTimeout(() => { if (get().phase === 'loading') set({ loadingStage: 'Extracting keywords...' }); }, 2000),
      setTimeout(() => { if (get().phase === 'loading') set({ loadingStage: 'Comparing skills...' }); }, 5000),
      setTimeout(() => { if (get().phase === 'loading') set({ loadingStage: 'Generating results...' }); }, 8000),
    ];
    try {
      const result = await analyzeResume(resumeFile, jobDescription);
      timers.forEach(clearTimeout);
      set({ phase: 'results', result, isDemo: false, loadingStage: '' });
    } catch (e) {
      timers.forEach(clearTimeout);
      set({
        phase: 'idle',
        error: e instanceof Error ? e.message : 'Analysis failed',
        loadingStage: '',
      });
    }
  },

  startQuickAnalysis: async (resumeText: string) => {
    const { jobDescription } = get();
    if (!resumeText.trim() || !jobDescription.trim()) {
      set({ error: 'Please provide both resume text and job description' });
      return;
    }

    set({ phase: 'loading', error: null, loadingStage: 'Parsing resume...' });
    const timers = [
      setTimeout(() => { if (get().phase === 'loading') set({ loadingStage: 'Extracting keywords...' }); }, 2000),
      setTimeout(() => { if (get().phase === 'loading') set({ loadingStage: 'Comparing skills...' }); }, 5000),
      setTimeout(() => { if (get().phase === 'loading') set({ loadingStage: 'Generating results...' }); }, 8000),
    ];
    try {
      const result = await analyzeQuick(resumeText, jobDescription);
      timers.forEach(clearTimeout);
      set({ phase: 'results', result, isDemo: false, loadingStage: '' });
    } catch (e) {
      timers.forEach(clearTimeout);
      set({
        phase: 'idle',
        error: e instanceof Error ? e.message : 'Analysis failed',
        loadingStage: '',
      });
    }
  },

  loadDemo: () => {
    set({ phase: 'results', result: DEMO_RESULT, isDemo: true, error: null });
  },

  reset: () => {
    set({
      phase: 'idle',
      result: null,
      error: null,
      resumeFile: null,
      jobDescription: '',
      isDemo: false,
      loadingStage: '',
    });
  },
}));
