"""Tests for Model 6: Verdict (Tier 1 template engine)."""

import pytest

from models.schemas.exp_edu_comparison import ExpEduComparison
from models.schemas.jd_extracted import JDExtracted
from models.schemas.judge_result import JudgeResult
from models.schemas.resume_extracted import ResumeExtracted
from models.schemas.skills_comparison import SkillMatch, SkillsComparison
from models.schemas.verdict_result import VerdictResult
from services.pipeline.m6_verdict import VerdictService


def _make_verdict_inputs(**overrides):
    """Create a set of pipeline outputs for testing verdict generation."""
    defaults = dict(
        resume_extracted=ResumeExtracted(
            skills=["python", "react"],
            sections_found=["experience", "skills", "education"],
        ),
        jd_extracted=JDExtracted(
            required_skills=["python", "java"],
            preferred_skills=["react"],
            certifications=["AWS"],
        ),
        skills_comparison=SkillsComparison(
            matched_skills=[
                SkillMatch(jd_skill="python", resume_skill="python", similarity=1.0,
                          match_type="exact", is_required=True),
                SkillMatch(jd_skill="react", resume_skill="react", similarity=1.0,
                          match_type="exact", is_required=False),
            ],
            missing_required=["java"],
            skill_coverage=0.67,
            required_coverage=0.5,
        ),
        exp_edu_comparison=ExpEduComparison(
            experience_score=75,
            education_score=80,
            domain_score=60,
            years_gap=1.0,
        ),
        judge_result=JudgeResult(overall_score=72),
        resume_text="- Built web applications using Python\n- Managed team projects",
    )
    defaults.update(overrides)
    return defaults


class TestVerdictService:
    def setup_method(self):
        self.svc = VerdictService()
        self.svc._loaded = True

    def test_produces_complete_output(self):
        inputs = _make_verdict_inputs()
        result = self.svc.predict(**inputs)

        assert isinstance(result, VerdictResult)
        assert result.summary != ""
        assert len(result.strengths) > 0
        assert len(result.weaknesses) > 0

    def test_summary_contains_score(self):
        inputs = _make_verdict_inputs()
        result = self.svc.predict(**inputs)
        assert "72" in result.summary

    def test_missing_skills_in_weaknesses(self):
        inputs = _make_verdict_inputs()
        result = self.svc.predict(**inputs)
        weakness_text = " ".join(result.weaknesses)
        assert "java" in weakness_text.lower()

    def test_matched_skills_in_strengths(self):
        inputs = _make_verdict_inputs()
        result = self.svc.predict(**inputs)
        strength_text = " ".join(result.strengths)
        assert "python" in strength_text.lower()

    def test_certifications_missing_warning(self):
        inputs = _make_verdict_inputs(
            resume_extracted=ResumeExtracted(certifications=[]),
        )
        result = self.svc.predict(**inputs)
        weakness_text = " ".join(result.weaknesses)
        assert "certification" in weakness_text.lower() or "aws" in weakness_text.lower()

    def test_strong_candidate(self):
        inputs = _make_verdict_inputs(
            judge_result=JudgeResult(overall_score=90),
            skills_comparison=SkillsComparison(
                matched_skills=[
                    SkillMatch(jd_skill="python", resume_skill="python",
                              similarity=1.0, match_type="exact"),
                ],
                skill_coverage=0.95,
            ),
            exp_edu_comparison=ExpEduComparison(
                experience_score=95, education_score=90, domain_score=85,
                years_gap=3.0, career_velocity=0.5,
            ),
        )
        result = self.svc.predict(**inputs)
        assert "strong" in result.summary.lower()


class TestVerdictResultSchema:
    def test_default_values(self):
        v = VerdictResult()
        assert v.summary == ""
        assert v.strengths == []
        assert v.bullet_rewrites == []
