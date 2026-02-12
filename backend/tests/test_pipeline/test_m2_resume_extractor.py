"""Tests for Model 2: Resume Extractor."""

import pytest

from models.schemas.resume_extracted import (
    Education,
    Project,
    ResumeExtracted,
    WorkExperience,
)
from services.pipeline.m2_resume_extractor import ResumeExtractorService


SAMPLE_RESUME = """
John Doe
john.doe@email.com | +1-555-0123 | San Francisco, CA

Experience

Senior Software Engineer, Google
Jan 2020 - Present
- Built scalable microservices using Python and Go
- Led team of 5 engineers

Software Engineer, Meta
Jun 2017 - Dec 2019
- Developed React frontend applications
- Implemented CI/CD pipelines with Jenkins

Education

Bachelor of Science in Computer Science
Stanford University, 2017

Skills

Python, Go, React, JavaScript, Docker, Kubernetes, AWS

Certifications

AWS Solutions Architect Associate
"""


class TestResumeExtractorFallback:
    def test_fallback_produces_output(self):
        svc = ResumeExtractorService()
        svc._use_fallback = True
        svc._loaded = True

        result = svc.predict(resume_text=SAMPLE_RESUME)
        assert isinstance(result, ResumeExtracted)
        assert len(result.skills) > 0
        assert result.total_years_experience > 0
        assert "experience" in result.sections_found or "skills" in result.sections_found

    def test_fallback_empty_resume(self):
        svc = ResumeExtractorService()
        svc._use_fallback = True
        svc._loaded = True

        result = svc.predict(resume_text="")
        assert isinstance(result, ResumeExtracted)
        assert result.name == ""

    def test_education_detection(self):
        svc = ResumeExtractorService()
        svc._use_fallback = True
        svc._loaded = True

        result = svc.predict(resume_text=SAMPLE_RESUME)
        assert result.highest_education in ("bachelors", "masters", "phd", "associate", "")


class TestResumeExtractedSchema:
    def test_default_values(self):
        r = ResumeExtracted()
        assert r.name == ""
        assert r.skills == []
        assert r.total_years_experience == 0.0

    def test_work_experience_model(self):
        we = WorkExperience(company="Google", title="SWE", duration_months=36)
        assert we.company == "Google"
        assert we.duration_months == 36

    def test_education_model(self):
        edu = Education(institution="Stanford", degree="bachelors", field="CS")
        assert edu.institution == "Stanford"

    def test_project_model(self):
        proj = Project(name="MyApp", technologies=["Python", "React"])
        assert len(proj.technologies) == 2
