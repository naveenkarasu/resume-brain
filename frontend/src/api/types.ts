export interface ScoreBreakdown {
  skills_match: number;
  experience_match: number;
  education_match: number;
  keywords_match: number;
}

export interface BulletRewrite {
  original: string;
  rewritten: string;
  reason: string;
}

export interface SectionAnalysis {
  detected_sections: string[];
  completeness: number;
}

export interface BulletScore {
  text: string;
  has_action_verb: boolean;
  has_metrics: boolean;
  length_ok: boolean;
  keyword_count: number;
  quality_score: number;
}

export interface AnalysisResponse {
  overall_score: number;
  score_breakdown: ScoreBreakdown;
  missing_keywords: string[];
  matched_keywords: string[];
  keyword_density: Record<string, number>;
  bullet_rewrites: BulletRewrite[];
  ats_optimized_resume: string;
  summary: string;
  strengths: string[];
  weaknesses: string[];
  degraded: boolean;
  tfidf_score: number;
  semantic_score: number;
  scoring_method: string;
  section_analysis: SectionAnalysis;
  experience_years: number;
  bullet_scores: BulletScore[];
  education_level: string;
}

export interface HealthResponse {
  status: string;
  gemini_configured: boolean;
}
