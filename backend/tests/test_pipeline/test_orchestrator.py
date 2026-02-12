"""Tests for the pipeline orchestrator."""

import pytest

from models.responses import AnalysisResponse
from services.pipeline.model_registry import clear as clear_registry


SAMPLE_RESUME = """
John Doe
john.doe@email.com | +1-555-0123

Experience

Senior Software Engineer, Google
Jan 2020 - Present
- Built scalable microservices using Python and Go
- Led team of 5 engineers on payment platform

Software Engineer, Meta
Jun 2017 - Dec 2019
- Developed React frontend applications
- Implemented CI/CD pipelines with Jenkins

Education

Bachelor of Science in Computer Science
Stanford University, 2017

Skills

Python, Go, React, JavaScript, Docker, Kubernetes, AWS, PostgreSQL
"""

SAMPLE_JD = """
Senior Python Developer

Requirements:
- 5+ years of experience with Python
- Strong knowledge of Django or FastAPI
- Experience with PostgreSQL and Redis
- Familiarity with Docker and Kubernetes

Preferred:
- Machine learning experience
- AWS certification

Education:
- Bachelor's degree in Computer Science or related field

Responsibilities:
- Design and build scalable APIs
- Mentor junior developers
"""


@pytest.fixture(autouse=True)
def _reset_registry():
    """Clear model registry before each test."""
    clear_registry()
    yield
    clear_registry()


class TestOrchestratorV2:
    @pytest.mark.asyncio
    async def test_full_pipeline_returns_analysis_response(self):
        from services.pipeline.orchestrator import analyze_v2

        result = await analyze_v2(SAMPLE_RESUME, SAMPLE_JD)
        assert isinstance(result, AnalysisResponse)
        assert 0 <= result.overall_score <= 100
        assert result.scoring_method == "pipeline_v2"
        assert result.degraded is False

    @pytest.mark.asyncio
    async def test_score_breakdown_populated(self):
        from services.pipeline.orchestrator import analyze_v2

        result = await analyze_v2(SAMPLE_RESUME, SAMPLE_JD)
        bd = result.score_breakdown
        assert 0 <= bd.skills_match <= 100
        assert 0 <= bd.experience_match <= 100
        assert 0 <= bd.education_match <= 100
        assert 0 <= bd.keywords_match <= 100

    @pytest.mark.asyncio
    async def test_keywords_populated(self):
        from services.pipeline.orchestrator import analyze_v2

        result = await analyze_v2(SAMPLE_RESUME, SAMPLE_JD)
        # Should find at least some matched and missing keywords
        total = len(result.matched_keywords) + len(result.missing_keywords)
        assert total > 0

    @pytest.mark.asyncio
    async def test_summary_and_feedback(self):
        from services.pipeline.orchestrator import analyze_v2

        result = await analyze_v2(SAMPLE_RESUME, SAMPLE_JD)
        assert result.summary != ""
        assert len(result.strengths) > 0

    @pytest.mark.asyncio
    async def test_experience_years(self):
        from services.pipeline.orchestrator import analyze_v2

        result = await analyze_v2(SAMPLE_RESUME, SAMPLE_JD)
        assert result.experience_years > 0

    @pytest.mark.asyncio
    async def test_empty_resume(self):
        from services.pipeline.orchestrator import analyze_v2

        result = await analyze_v2("", SAMPLE_JD)
        assert isinstance(result, AnalysisResponse)
        assert result.overall_score <= 30

    @pytest.mark.asyncio
    async def test_empty_jd(self):
        from services.pipeline.orchestrator import analyze_v2

        result = await analyze_v2(SAMPLE_RESUME, "")
        assert isinstance(result, AnalysisResponse)


class TestPipelineModeSwitch:
    @pytest.mark.asyncio
    async def test_legacy_mode_default(self):
        """Legacy mode should work without pipeline models."""
        from config import settings
        original = settings.pipeline_mode
        try:
            settings.pipeline_mode = "legacy"
            from services.resume_analyzer import analyze
            # Legacy mode should not import pipeline
            result = await analyze(SAMPLE_RESUME, SAMPLE_JD)
            assert isinstance(result, AnalysisResponse)
            assert result.scoring_method in ("hybrid", "local_only")
        finally:
            settings.pipeline_mode = original

    @pytest.mark.asyncio
    async def test_v2_mode(self):
        """V2 mode should use pipeline_v2 scoring method."""
        from config import settings
        original = settings.pipeline_mode
        try:
            settings.pipeline_mode = "v2"
            from services.resume_analyzer import analyze
            result = await analyze(SAMPLE_RESUME, SAMPLE_JD)
            assert isinstance(result, AnalysisResponse)
            assert result.scoring_method == "pipeline_v2"
        finally:
            settings.pipeline_mode = original
