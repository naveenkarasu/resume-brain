from pydantic import BaseModel


class ScoreBreakdown(BaseModel):
    skills_match: int = 0
    experience_match: int = 0
    education_match: int = 0
    keywords_match: int = 0


class BulletRewrite(BaseModel):
    original: str
    rewritten: str
    reason: str


class SectionAnalysis(BaseModel):
    detected_sections: list[str] = []
    completeness: float = 0.0


class AnalysisResponse(BaseModel):
    overall_score: int = 0
    score_breakdown: ScoreBreakdown = ScoreBreakdown()
    missing_keywords: list[str] = []
    matched_keywords: list[str] = []
    keyword_density: dict[str, float] = {}
    bullet_rewrites: list[BulletRewrite] = []
    ats_optimized_resume: str = ""
    summary: str = ""
    strengths: list[str] = []
    weaknesses: list[str] = []
    degraded: bool = False
    # Scoring transparency fields
    tfidf_score: float = 0.0
    semantic_score: float = 0.0
    scoring_method: str = "hybrid"
    section_analysis: SectionAnalysis = SectionAnalysis()
    experience_years: float = 0.0
    bullet_scores: list[dict] = []
    education_level: str = ""
