"""Tests for Model 5: Judge."""

import pytest

from models.schemas.exp_edu_comparison import ExpEduComparison
from models.schemas.judge_result import JudgeResult
from models.schemas.skills_comparison import SkillMatch, SkillsComparison
from services.pipeline.m5_judge import JudgeService, FEATURE_NAMES


class TestFeatureExtraction:
    def setup_method(self):
        self.svc = JudgeService()
        self.svc._use_fallback = True
        self.svc._loaded = True

    def test_feature_count(self):
        skills = SkillsComparison(
            matched_skills=[
                SkillMatch(jd_skill="python", resume_skill="python", similarity=1.0),
            ],
            missing_required=["java"],
            skill_coverage=0.5,
            required_coverage=0.5,
        )
        exp_edu = ExpEduComparison(
            experience_score=80,
            education_score=70,
            domain_score=60,
            title_score=75,
            years_gap=2.0,
            career_velocity=0.3,
        )

        features = self.svc._extract_features(skills, exp_edu)
        assert len(features) == len(FEATURE_NAMES)

    def test_empty_skills(self):
        skills = SkillsComparison()
        exp_edu = ExpEduComparison()
        features = self.svc._extract_features(skills, exp_edu)
        assert features["avg_match_similarity"] == 0.0
        assert features["n_matched_skills"] == 0.0


class TestJudgeFallback:
    def test_strong_candidate(self):
        svc = JudgeService()
        svc._use_fallback = True
        svc._loaded = True

        skills = SkillsComparison(
            matched_skills=[
                SkillMatch(jd_skill="python", resume_skill="python", similarity=1.0),
                SkillMatch(jd_skill="react", resume_skill="react", similarity=1.0),
            ],
            skill_coverage=0.9,
            required_coverage=0.9,
        )
        exp_edu = ExpEduComparison(
            experience_score=90,
            education_score=85,
            domain_score=80,
        )

        result = svc.predict(skills_comparison=skills, exp_edu_comparison=exp_edu)
        assert isinstance(result, JudgeResult)
        assert result.overall_score >= 60  # strong candidate

    def test_weak_candidate(self):
        svc = JudgeService()
        svc._use_fallback = True
        svc._loaded = True

        skills = SkillsComparison(
            skill_coverage=0.1,
            required_coverage=0.1,
        )
        exp_edu = ExpEduComparison(
            experience_score=20,
            education_score=30,
            domain_score=10,
        )

        result = svc.predict(skills_comparison=skills, exp_edu_comparison=exp_edu)
        assert result.overall_score <= 40  # weak candidate

    def test_breakdown_populated(self):
        svc = JudgeService()
        svc._use_fallback = True
        svc._loaded = True

        skills = SkillsComparison(skill_coverage=0.7, required_coverage=0.8)
        exp_edu = ExpEduComparison(experience_score=75, education_score=80)

        result = svc.predict(skills_comparison=skills, exp_edu_comparison=exp_edu)
        assert result.score_breakdown.skills_match >= 0
        assert result.score_breakdown.experience_match >= 0
        assert len(result.feature_vector) == len(FEATURE_NAMES)


class TestJudgeResultSchema:
    def test_default_values(self):
        j = JudgeResult()
        assert j.overall_score == 0
        assert j.feature_vector == []
