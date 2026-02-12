"""Tests for Model 3: Skills Comparator."""

import pytest

from models.schemas.jd_extracted import JDExtracted
from models.schemas.resume_extracted import ResumeExtracted
from models.schemas.skills_comparison import SkillMatch, SkillsComparison
from services.pipeline.m3_skills_comparator import SkillsComparatorService


class TestSkillsComparatorFallback:
    def test_exact_matches(self):
        svc = SkillsComparatorService()
        svc._use_fallback = True
        svc._loaded = True

        resume = ResumeExtracted(skills=["python", "react", "docker"])
        jd = JDExtracted(required_skills=["python", "react"], preferred_skills=["kubernetes"])

        result = svc.predict(resume_extracted=resume, jd_extracted=jd)
        assert isinstance(result, SkillsComparison)
        assert len(result.matched_skills) == 2
        assert result.skill_coverage > 0
        assert "kubernetes" in result.missing_preferred

    def test_no_resume_skills(self):
        svc = SkillsComparatorService()
        svc._use_fallback = True
        svc._loaded = True

        resume = ResumeExtracted(skills=[])
        jd = JDExtracted(required_skills=["python"])

        result = svc.predict(resume_extracted=resume, jd_extracted=jd)
        assert len(result.missing_required) == 1
        assert result.skill_coverage == 0.0

    def test_no_jd_skills(self):
        svc = SkillsComparatorService()
        svc._use_fallback = True
        svc._loaded = True

        resume = ResumeExtracted(skills=["python"])
        jd = JDExtracted(required_skills=[], preferred_skills=[])

        result = svc.predict(resume_extracted=resume, jd_extracted=jd)
        assert result.skill_coverage == 0.5  # neutral
        assert result.extra_skills == ["python"]

    def test_taxonomy_match(self):
        svc = SkillsComparatorService()
        svc._use_fallback = True
        svc._loaded = True

        # React implies JavaScript via SKILL_IMPLICATIONS
        resume = ResumeExtracted(skills=["react"])
        jd = JDExtracted(required_skills=["javascript"])

        result = svc.predict(resume_extracted=resume, jd_extracted=jd)
        # Should find taxonomy match
        assert len(result.matched_skills) == 1
        assert result.matched_skills[0].match_type == "taxonomy"


class TestSkillMatchSchema:
    def test_default_values(self):
        m = SkillMatch(jd_skill="python")
        assert m.resume_skill == ""
        assert m.similarity == 0.0
        assert m.match_type == "none"
        assert m.is_required is False

    def test_full_match(self):
        m = SkillMatch(
            jd_skill="python",
            resume_skill="python",
            similarity=1.0,
            match_type="exact",
            is_required=True,
        )
        assert m.is_required is True


class TestSkillsComparisonSchema:
    def test_default_values(self):
        sc = SkillsComparison()
        assert sc.skill_coverage == 0.0
        assert sc.matched_skills == []
