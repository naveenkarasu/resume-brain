"""Tests for Model 4: Experience/Education Comparator."""

import pytest

from models.schemas.exp_edu_comparison import ExpEduComparison
from models.schemas.jd_extracted import JDExtracted
from models.schemas.resume_extracted import Education, ResumeExtracted, WorkExperience
from services.pipeline.m4_exp_edu_comparator import ExpEduComparatorService, FEATURE_NAMES


class TestFeatureExtraction:
    def setup_method(self):
        self.svc = ExpEduComparatorService()
        self.svc._use_fallback = True
        self.svc._loaded = True

    def test_feature_count(self):
        resume = ResumeExtracted(
            total_years_experience=5.0,
            highest_education="bachelors",
            skills=["python", "react"],
            work_experience=[
                WorkExperience(company="Google", title="Senior Engineer", duration_months=36),
            ],
            education=[Education(degree="bachelors", field="Computer Science")],
        )
        jd = JDExtracted(
            required_years=3.0,
            education_requirements=["bachelors"],
            domain=["software engineering"],
        )

        features = self.svc._extract_features(resume, jd)
        assert len(features) == len(FEATURE_NAMES)

    def test_years_gap_positive(self):
        resume = ResumeExtracted(total_years_experience=8.0)
        jd = JDExtracted(required_years=5.0)
        features = self.svc._extract_features(resume, jd)
        assert features["years_gap"] == 3.0

    def test_years_gap_negative(self):
        resume = ResumeExtracted(total_years_experience=2.0)
        jd = JDExtracted(required_years=5.0)
        features = self.svc._extract_features(resume, jd)
        assert features["years_gap"] == -3.0

    def test_education_gap(self):
        resume = ResumeExtracted(highest_education="masters")
        jd = JDExtracted(education_requirements=["bachelors"])
        features = self.svc._extract_features(resume, jd)
        assert features["edu_gap"] == 1  # masters(3) - bachelors(2) = 1

    def test_leadership_detection(self):
        resume = ResumeExtracted(
            work_experience=[WorkExperience(title="Lead Engineer")]
        )
        jd = JDExtracted()
        features = self.svc._extract_features(resume, jd)
        assert features["has_leadership"] == 1.0


class TestExpEduFallback:
    def test_produces_valid_output(self):
        svc = ExpEduComparatorService()
        svc._use_fallback = True
        svc._loaded = True

        resume = ResumeExtracted(
            total_years_experience=5.0,
            highest_education="bachelors",
        )
        jd = JDExtracted(required_years=3.0)

        result = svc.predict(resume_extracted=resume, jd_extracted=jd)
        assert isinstance(result, ExpEduComparison)
        assert 0 <= result.experience_score <= 100
        assert 0 <= result.education_score <= 100

    def test_empty_inputs(self):
        svc = ExpEduComparatorService()
        svc._use_fallback = True
        svc._loaded = True

        result = svc.predict(
            resume_extracted=ResumeExtracted(),
            jd_extracted=JDExtracted(),
        )
        assert isinstance(result, ExpEduComparison)


class TestExpEduSchema:
    def test_default_values(self):
        e = ExpEduComparison()
        assert e.experience_score == 0.0
        assert e.field_match is False
