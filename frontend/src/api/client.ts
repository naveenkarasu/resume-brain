import type { AnalysisResponse, HealthResponse } from './types';
import { getApiBaseUrl } from '../utils/platform';

const BASE_URL = getApiBaseUrl();

export async function checkHealth(): Promise<HealthResponse> {
  const res = await fetch(`${BASE_URL}/health`);
  if (!res.ok) throw new Error('Health check failed');
  return res.json();
}

export async function analyzeResume(
  file: File,
  jobDescription: string
): Promise<AnalysisResponse> {
  const formData = new FormData();
  formData.append('resume_file', file);
  formData.append('job_description', jobDescription);

  const res = await fetch(`${BASE_URL}/analyze`, {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Analysis failed' }));
    throw new Error(err.detail || 'Analysis failed');
  }

  return res.json();
}

export async function analyzeQuick(
  resumeText: string,
  jobDescription: string
): Promise<AnalysisResponse> {
  const res = await fetch(`${BASE_URL}/analyze/quick`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      resume_text: resumeText,
      job_description: jobDescription,
    }),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Analysis failed' }));
    throw new Error(err.detail || 'Analysis failed');
  }

  return res.json();
}
