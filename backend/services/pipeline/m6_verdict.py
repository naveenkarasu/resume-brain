"""Model 6: Verdict - User-facing output generation.

Tier 1 (immediate): Template-based rules engine. No ML model needed.
    Generates strengths, weaknesses, summary, bullet rewrites, and
    recommendations from structured pipeline outputs.

Tier 2 (future): Fine-tuned FLAN-T5-base for richer natural language output.
"""

import logging
from typing import Any

from models.responses import BulletRewrite
from models.schemas.exp_edu_comparison import ExpEduComparison
from models.schemas.jd_extracted import JDExtracted
from models.schemas.judge_result import JudgeResult
from models.schemas.resume_extracted import ResumeExtracted
from models.schemas.skills_comparison import SkillsComparison
from models.schemas.verdict_result import VerdictResult
from services.pipeline.base import BaseModelService

logger = logging.getLogger(__name__)


class VerdictService(BaseModelService):
    model_name = "m6_verdict"

    def __init__(self) -> None:
        pass

    def load(self) -> None:
        # Tier 1: no model to load - pure template engine
        logger.info("M6 Verdict engine ready (Tier 1: template-based)")

    def predict(self, **kwargs: Any) -> VerdictResult:
        self.ensure_loaded()
        resume_extracted: ResumeExtracted = kwargs["resume_extracted"]
        jd_extracted: JDExtracted = kwargs["jd_extracted"]
        skills_comparison: SkillsComparison = kwargs["skills_comparison"]
        exp_edu_comparison: ExpEduComparison = kwargs["exp_edu_comparison"]
        judge_result: JudgeResult = kwargs["judge_result"]
        resume_text: str = kwargs.get("resume_text", "")

        summary = _build_summary(judge_result, skills_comparison, exp_edu_comparison)
        strengths = _build_strengths(skills_comparison, exp_edu_comparison, resume_extracted)
        weaknesses = _build_weaknesses(skills_comparison, exp_edu_comparison, resume_extracted, jd_extracted)
        recommendations = _build_recommendations(skills_comparison, exp_edu_comparison, resume_extracted, jd_extracted)
        bullet_rewrites = _build_bullet_rewrites(resume_text, jd_extracted)

        return VerdictResult(
            summary=summary,
            strengths=strengths,
            weaknesses=weaknesses,
            bullet_rewrites=bullet_rewrites,
            recommendations=recommendations,
        )


# ---------------------------------------------------------------------------
# Template builders
# ---------------------------------------------------------------------------

def _build_summary(
    judge: JudgeResult,
    skills: SkillsComparison,
    exp_edu: ExpEduComparison,
) -> str:
    """Generate a 2-3 sentence summary of the analysis."""
    score = judge.overall_score
    bd = judge.score_breakdown

    if score >= 80:
        quality = "strong"
    elif score >= 60:
        quality = "moderate"
    elif score >= 40:
        quality = "partial"
    else:
        quality = "weak"

    parts = [
        f"Overall match score: {score}/100 ({quality} fit).",
    ]

    # Skills insight
    n_matched = len(skills.matched_skills)
    n_missing = len(skills.missing_required) + len(skills.missing_preferred)
    if n_matched > 0 and n_missing == 0:
        parts.append(f"All {n_matched} required skills are covered.")
    elif n_matched > 0:
        parts.append(
            f"{n_matched} skills matched, {len(skills.missing_required)} required "
            f"and {len(skills.missing_preferred)} preferred skills missing."
        )
    else:
        parts.append("No skill matches detected.")

    # Experience insight
    if exp_edu.years_gap >= 0:
        parts.append(
            f"Experience meets requirements ({exp_edu.experience_score:.0f}/100)."
        )
    elif exp_edu.years_gap >= -2:
        parts.append(
            f"Experience is slightly below requirements "
            f"({exp_edu.years_gap:+.1f} years gap)."
        )
    else:
        parts.append(
            f"Significant experience gap ({exp_edu.years_gap:+.1f} years)."
        )

    return " ".join(parts)


def _build_strengths(
    skills: SkillsComparison,
    exp_edu: ExpEduComparison,
    resume: ResumeExtracted,
) -> list[str]:
    """Generate list of strengths from pipeline outputs."""
    strengths: list[str] = []

    # Strong skill matches
    exact_matches = [m for m in skills.matched_skills if m.match_type == "exact"]
    if exact_matches:
        top = [m.jd_skill for m in exact_matches[:5]]
        strengths.append(f"Direct skill matches: {', '.join(top)}")

    # Semantic skill matches (transferable skills)
    semantic_matches = [m for m in skills.matched_skills if m.match_type == "semantic"]
    if semantic_matches:
        top = [f"{m.resume_skill} -> {m.jd_skill}" for m in semantic_matches[:3]]
        strengths.append(f"Transferable skills: {', '.join(top)}")

    # Experience strength
    if exp_edu.experience_score >= 80:
        strengths.append(
            f"Strong experience match ({exp_edu.experience_score:.0f}/100)"
        )

    # Education strength
    if exp_edu.education_score >= 80:
        strengths.append(
            f"Education meets or exceeds requirements ({exp_edu.education_score:.0f}/100)"
        )

    # Domain alignment
    if exp_edu.domain_score >= 70:
        strengths.append(
            f"Career domain aligns well with the role ({exp_edu.domain_score:.0f}/100)"
        )

    # Leadership
    if exp_edu.career_velocity >= 0.5:
        strengths.append("Strong career progression demonstrated")

    # Extra relevant skills
    if skills.extra_skills:
        strengths.append(
            f"Additional skills beyond requirements: {', '.join(skills.extra_skills[:5])}"
        )

    return strengths or ["Resume contains relevant qualifications"]


def _build_weaknesses(
    skills: SkillsComparison,
    exp_edu: ExpEduComparison,
    resume: ResumeExtracted,
    jd: JDExtracted,
) -> list[str]:
    """Generate list of weaknesses from pipeline outputs."""
    weaknesses: list[str] = []

    # Missing required skills
    if skills.missing_required:
        top = skills.missing_required[:5]
        weaknesses.append(f"Missing required skills: {', '.join(top)}")

    # Missing preferred skills
    if skills.missing_preferred:
        top = skills.missing_preferred[:3]
        weaknesses.append(f"Missing preferred skills: {', '.join(top)}")

    # Experience gaps
    if exp_edu.experience_score < 50:
        weaknesses.append(
            f"Experience below requirements "
            f"({exp_edu.years_gap:+.1f} years gap)"
        )

    # Education gaps
    if exp_edu.edu_gap < 0:
        weaknesses.append(
            f"Education level below stated requirement"
        )

    # Domain mismatch
    if exp_edu.domain_score < 30:
        weaknesses.append(
            "Career domain does not closely align with the role"
        )

    # Missing certifications
    if jd.certifications and not resume.certifications:
        weaknesses.append(
            f"JD mentions certifications ({', '.join(jd.certifications[:3])}) "
            f"but none found on resume"
        )

    return weaknesses or ["No significant gaps identified"]


def _build_recommendations(
    skills: SkillsComparison,
    exp_edu: ExpEduComparison,
    resume: ResumeExtracted,
    jd: JDExtracted,
) -> list[str]:
    """Generate actionable recommendations."""
    recs: list[str] = []

    # Add missing required skills
    if skills.missing_required:
        top = skills.missing_required[:3]
        recs.append(
            f"Add these required skills to your resume if applicable: {', '.join(top)}"
        )

    # Quantify experience
    if exp_edu.experience_score < 70:
        recs.append(
            "Quantify your experience with specific metrics, projects, or achievements"
        )

    # Section completeness
    expected = {"experience", "education", "skills", "projects"}
    found = set(resume.sections_found)
    missing_sections = expected - found
    if missing_sections:
        recs.append(
            f"Add missing resume sections: {', '.join(sorted(missing_sections))}"
        )

    # Keyword optimization
    if skills.skill_coverage < 0.5:
        recs.append(
            "Tailor your resume to include more job-specific terminology"
        )

    # Certifications
    if jd.certifications and not resume.certifications:
        recs.append(
            f"Consider adding relevant certifications: {', '.join(jd.certifications[:2])}"
        )

    return recs


def _build_bullet_rewrites(
    resume_text: str,
    jd: JDExtracted,
) -> list[BulletRewrite]:
    """Generate bullet rewrite suggestions using template rules."""
    from services.pdf_parser import extract_bullets, score_bullet

    if not resume_text:
        return []

    bullets = extract_bullets(resume_text)
    jd_keywords = set(jd.required_skills + jd.preferred_skills + jd.tools)
    rewrites: list[BulletRewrite] = []

    for bullet in bullets[:10]:
        score_data = score_bullet(bullet, jd_keywords)
        if score_data["quality_score"] >= 70:
            continue  # Good enough, no rewrite needed

        reasons: list[str] = []
        if not score_data["has_action_verb"]:
            reasons.append("Start with a strong action verb")
        if not score_data["has_metrics"]:
            reasons.append("Add quantified metrics or results")
        if not score_data["length_ok"]:
            reasons.append("Adjust length to 10-35 words")
        if score_data["keyword_count"] == 0:
            reasons.append("Incorporate relevant JD keywords")

        if reasons:
            rewrites.append(BulletRewrite(
                original=bullet,
                rewritten="",  # Tier 1: no auto-rewrite, just guidance
                reason="; ".join(reasons),
            ))

    return rewrites[:5]
