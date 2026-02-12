"""Pipeline orchestrator: wires all 6 models together.

Flow:
    resume_text + jd_text
      ├─ M1.predict(jd_text)         → JDExtracted
      ├─ M2.predict(resume_text)     → ResumeExtracted
      │       ↓                              ↓
      ├─ M3.predict(resume_skills, jd_skills)  → SkillsComparison
      ├─ M4.predict(resume_exp/edu, jd_reqs)   → ExpEduComparison
      │               ↓                              ↓
      ├─ M5.predict(skills_comparison, exp_edu_comparison)  → JudgeResult
      │                                      ↓
      └─ M6.predict(all_outputs + raw_text)  → VerdictResult
                       ↓
         _to_analysis_response()  → AnalysisResponse (backward compatible)
"""

import logging

from models.responses import AnalysisResponse, SectionAnalysis
from models.schemas.exp_edu_comparison import ExpEduComparison
from models.schemas.jd_extracted import JDExtracted
from models.schemas.judge_result import JudgeResult
from models.schemas.resume_extracted import ResumeExtracted
from models.schemas.skills_comparison import SkillsComparison
from models.schemas.verdict_result import VerdictResult
from services.pipeline.model_registry import get_model

logger = logging.getLogger(__name__)


async def analyze_v2(resume_text: str, job_description: str) -> AnalysisResponse:
    """Run the 6-model pipeline and return a backward-compatible AnalysisResponse."""

    # --- Stage 1: Extraction (M1 + M2, independent) ---
    m1 = get_model("m1_jd_extractor")
    m2 = get_model("m2_resume_extractor")

    jd_extracted: JDExtracted = m1.predict(jd_text=job_description)
    resume_extracted: ResumeExtracted = m2.predict(resume_text=resume_text)

    # --- Stage 2: Comparison (M3 + M4, depend on Stage 1) ---
    m3 = get_model("m3_skills_comparator")
    m4 = get_model("m4_exp_edu_comparator")

    skills_comparison: SkillsComparison = m3.predict(
        resume_extracted=resume_extracted,
        jd_extracted=jd_extracted,
    )
    exp_edu_comparison: ExpEduComparison = m4.predict(
        resume_extracted=resume_extracted,
        jd_extracted=jd_extracted,
    )

    # --- Stage 3: Scoring (M5, depends on Stage 2) ---
    m5 = get_model("m5_judge")
    judge_result: JudgeResult = m5.predict(
        skills_comparison=skills_comparison,
        exp_edu_comparison=exp_edu_comparison,
    )

    # --- Stage 4: Output (M6, depends on all previous) ---
    m6 = get_model("m6_verdict")
    verdict: VerdictResult = m6.predict(
        resume_extracted=resume_extracted,
        jd_extracted=jd_extracted,
        skills_comparison=skills_comparison,
        exp_edu_comparison=exp_edu_comparison,
        judge_result=judge_result,
        resume_text=resume_text,
    )

    # --- Map to backward-compatible AnalysisResponse ---
    return _to_analysis_response(
        resume_text=resume_text,
        job_description=job_description,
        jd_extracted=jd_extracted,
        resume_extracted=resume_extracted,
        skills_comparison=skills_comparison,
        exp_edu_comparison=exp_edu_comparison,
        judge_result=judge_result,
        verdict=verdict,
    )


def _to_analysis_response(
    resume_text: str,
    job_description: str,
    jd_extracted: JDExtracted,
    resume_extracted: ResumeExtracted,
    skills_comparison: SkillsComparison,
    exp_edu_comparison: ExpEduComparison,
    judge_result: JudgeResult,
    verdict: VerdictResult,
) -> AnalysisResponse:
    """Map pipeline outputs to the existing AnalysisResponse schema."""
    from services import keyword_extractor, pdf_parser
    from services.similarity import hybrid_similarity

    # Recompute TF-IDF/semantic for transparency (reuses cached models)
    _, tfidf_score, semantic_score = hybrid_similarity(resume_text, job_description)

    # Keyword density from matched + missing skills
    all_keywords = (
        [m.jd_skill for m in skills_comparison.matched_skills]
        + skills_comparison.missing_required
        + skills_comparison.missing_preferred
    )
    keyword_density = keyword_extractor.compute_keyword_density(resume_text, all_keywords)

    # Bullet scores
    bullets = pdf_parser.extract_bullets(resume_text)
    jd_kw_set = set(jd_extracted.required_skills + jd_extracted.preferred_skills)
    bullet_scores = pdf_parser.score_bullets(bullets, jd_kw_set)

    # Section analysis
    section_analysis = SectionAnalysis(
        detected_sections=resume_extracted.sections_found,
        completeness=round(len(resume_extracted.sections_found) / 7, 2),
    )

    # Matched/missing keyword lists
    matched_keywords = sorted({m.jd_skill for m in skills_comparison.matched_skills})
    missing_keywords = sorted(
        set(skills_comparison.missing_required + skills_comparison.missing_preferred)
    )

    return AnalysisResponse(
        overall_score=judge_result.overall_score,
        score_breakdown=judge_result.score_breakdown,
        matched_keywords=matched_keywords,
        missing_keywords=missing_keywords,
        keyword_density=keyword_density,
        bullet_rewrites=verdict.bullet_rewrites,
        ats_optimized_resume=verdict.ats_optimized_resume,
        summary=verdict.summary,
        strengths=verdict.strengths,
        weaknesses=verdict.weaknesses,
        degraded=False,  # no external API dependency
        tfidf_score=round(tfidf_score, 4),
        semantic_score=round(max(0, semantic_score), 4),
        scoring_method="pipeline_v2",
        section_analysis=section_analysis,
        experience_years=resume_extracted.total_years_experience,
        bullet_scores=bullet_scores,
        education_level=resume_extracted.highest_education,
    )
