"""Tests for Model 1: JD Extractor."""

import pytest

from models.schemas.jd_extracted import JDExtracted, JDEntity
from services.pipeline.m1_jd_extractor import JDExtractorService, _chunk_text, _entities_to_schema


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


class TestChunkText:
    def test_short_text(self):
        chunks = _chunk_text("Hello world")
        assert len(chunks) == 1
        assert chunks[0] == "Hello world"

    def test_long_text_splits(self):
        text = "\n\n".join(["word " * 100 for _ in range(5)])
        chunks = _chunk_text(text, max_words=100)
        assert len(chunks) > 1

    def test_empty_text(self):
        chunks = _chunk_text("")
        assert chunks == [""]  # fallback


class TestEntitiesToSchema:
    def test_maps_entities_correctly(self):
        entities = [
            JDEntity(text="Python", label="SKILL", confidence=0.95),
            JDEntity(text="Django", label="SKILL", confidence=0.90),
            JDEntity(text="teamwork", label="SOFT_SKILL", confidence=0.85),
            JDEntity(text="5+ years", label="EXPERIENCE_REQ", confidence=0.88),
            JDEntity(text="Bachelor's", label="EDUCATION_REQ", confidence=0.92),
        ]
        result = _entities_to_schema(entities, "5+ years of experience")
        assert "Python" in result.required_skills
        assert "Django" in result.required_skills
        assert "teamwork" in result.soft_skills
        assert result.required_years == 5.0

    def test_deduplicates(self):
        entities = [
            JDEntity(text="Python", label="SKILL", confidence=0.95),
            JDEntity(text="python", label="SKILL", confidence=0.90),
        ]
        result = _entities_to_schema(entities, "")
        assert len(result.required_skills) == 1

    def test_empty_entities(self):
        result = _entities_to_schema([], "no years mentioned")
        assert result.required_skills == []
        assert result.required_years == 0.0


class TestJDExtractorFallback:
    def test_fallback_produces_output(self):
        svc = JDExtractorService()
        svc._use_fallback = True
        svc._loaded = True

        result = svc.predict(jd_text=SAMPLE_JD)
        assert isinstance(result, JDExtracted)
        assert result.required_years == 5.0
        # Should find some skills via keyword_extractor
        all_skills = result.required_skills + result.preferred_skills
        assert len(all_skills) > 0

    def test_fallback_empty_jd(self):
        svc = JDExtractorService()
        svc._use_fallback = True
        svc._loaded = True

        result = svc.predict(jd_text="")
        assert isinstance(result, JDExtracted)


class TestJDExtractorSchema:
    def test_schema_fields(self):
        result = JDExtracted()
        assert result.required_skills == []
        assert result.preferred_skills == []
        assert result.required_years == 0.0

    def test_entity_model(self):
        entity = JDEntity(text="Python", label="SKILL")
        assert entity.confidence == 1.0
        assert entity.start == 0
