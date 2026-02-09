"""Tests for resume NER parser."""

from unittest.mock import patch

import pytest

from services.resume_ner import _chunk_text, _empty_entities, extract_resume_entities


# ---------------------------------------------------------------------------
# Chunking tests
# ---------------------------------------------------------------------------


def test_chunk_text_short():
    """Short text should produce a single chunk."""
    text = "John Doe\njohn@example.com\n\nSoftware Engineer"
    chunks = _chunk_text(text)
    assert len(chunks) == 1
    assert "John Doe" in chunks[0]


def test_chunk_text_long():
    """Long text should be split into multiple chunks."""
    # Create text with many paragraphs exceeding 450 tokens
    paragraphs = [f"Paragraph {i} " + "word " * 80 for i in range(10)]
    text = "\n\n".join(paragraphs)
    chunks = _chunk_text(text)
    assert len(chunks) > 1
    # All original content should be preserved across chunks
    rejoined = " ".join(chunks)
    assert "Paragraph 0" in rejoined
    assert "Paragraph 9" in rejoined


def test_chunk_text_empty():
    """Empty text should return empty list."""
    assert _chunk_text("") == []
    assert _chunk_text("   ") == []


# ---------------------------------------------------------------------------
# Entity extraction tests (mocked model)
# ---------------------------------------------------------------------------


def _mock_ner_results():
    """Return mock NER pipeline results."""
    return [
        {"entity_group": "Name", "score": 0.95, "word": "John Doe"},
        {"entity_group": "Email Address", "score": 0.92, "word": "john@example.com"},
        {"entity_group": "Skills", "score": 0.88, "word": "Python"},
        {"entity_group": "Skills", "score": 0.85, "word": "JavaScript"},
        {"entity_group": "Companies Worked At", "score": 0.90, "word": "Google"},
        {"entity_group": "Designation", "score": 0.87, "word": "Senior Engineer"},
        {"entity_group": "College Name", "score": 0.91, "word": "MIT"},
        {"entity_group": "Graduation Year", "score": 0.89, "word": "2018"},
        {"entity_group": "Location", "score": 0.86, "word": "San Francisco"},
        {"entity_group": "Years of Experience", "score": 0.83, "word": "5 years"},
        {"entity_group": "Degree", "score": 0.90, "word": "Bachelor's"},
    ]


@patch("services.resume_ner._get_resume_ner")
def test_returns_empty_when_model_unavailable(mock_get):
    """When model is unavailable, return empty structure."""
    mock_get.return_value = None
    result = extract_resume_entities("Some resume text here")
    assert result == _empty_entities()
    assert all(isinstance(v, list) and len(v) == 0 for v in result.values())


@patch("services.resume_ner._get_resume_ner")
def test_parses_model_output_correctly(mock_get):
    """Mock NER returns valid entities, verify dict structure."""
    mock_pipeline = lambda text: _mock_ner_results()
    mock_get.return_value = mock_pipeline
    result = extract_resume_entities("John Doe john@example.com Senior Engineer at Google")
    assert "John Doe" in result["name"]
    assert "john@example.com" in result["email"]
    assert "Python" in result["skills"]
    assert "JavaScript" in result["skills"]
    assert "Google" in result["companies"]
    assert "Senior Engineer" in result["designations"]
    assert "MIT" in result["colleges"]
    assert "2018" in result["graduation_years"]
    assert "San Francisco" in result["locations"]
    assert "5 years" in result["years_of_experience"]
    assert "Bachelor's" in result["degrees"]


@patch("services.resume_ner._get_resume_ner")
def test_filters_low_confidence_entities(mock_get):
    """Entities with score < 0.5 should be excluded."""
    mock_pipeline = lambda text: [
        {"entity_group": "Skills", "score": 0.9, "word": "Python"},
        {"entity_group": "Skills", "score": 0.3, "word": "Cooking"},
        {"entity_group": "Name", "score": 0.1, "word": "Unknown"},
    ]
    mock_get.return_value = mock_pipeline
    result = extract_resume_entities("Some text")
    assert "Python" in result["skills"]
    assert "Cooking" not in result["skills"]
    assert "Unknown" not in result["name"]


@patch("services.resume_ner._get_resume_ner")
def test_deduplicates_entities(mock_get):
    """Duplicate entities (case-insensitive) should be removed."""
    mock_pipeline = lambda text: [
        {"entity_group": "Skills", "score": 0.9, "word": "Python"},
        {"entity_group": "Skills", "score": 0.85, "word": "python"},
        {"entity_group": "Skills", "score": 0.88, "word": "PYTHON"},
    ]
    mock_get.return_value = mock_pipeline
    result = extract_resume_entities("Python developer")
    assert len(result["skills"]) == 1


@patch("services.resume_ner._get_resume_ner")
def test_handles_model_exception(mock_get):
    """If model raises exception, return empty structure."""
    def raise_err(text):
        raise RuntimeError("Model error")
    mock_get.return_value = raise_err
    result = extract_resume_entities("Some text")
    assert result == _empty_entities()


# ---------------------------------------------------------------------------
# Section parser NER integration tests
# ---------------------------------------------------------------------------

from services.section_parser import (
    extract_education_level,
    extract_experience_years,
    extract_contact_info,
)


def test_extract_education_level_with_ner():
    """NER degrees should enhance regex detection."""
    # Text has no degree keyword, but NER found one
    text = "Graduated from Stanford University in 2020"
    ner = {"degrees": ["Master of Science"]}
    level = extract_education_level(text, ner_entities=ner)
    assert level == "masters"


def test_extract_education_level_without_ner():
    """Without NER, existing behavior is unchanged."""
    text = "B.S. Computer Science from MIT"
    level = extract_education_level(text)
    assert level == "bachelors"
    # Explicit None also works
    level2 = extract_education_level(text, ner_entities=None)
    assert level2 == "bachelors"


def test_extract_experience_years_with_ner():
    """NER years_of_experience should enhance regex."""
    # Text has no explicit years claim or date ranges
    text = "Experienced professional in software development"
    ner = {"years_of_experience": ["8 years"]}
    years = extract_experience_years(text, ner_entities=ner)
    assert years >= 8.0


def test_extract_contact_info_with_ner():
    """NER should fill gaps when regex misses contact info."""
    # Text with no standard email format
    text = "Contact me for details"
    ner = {"email": ["john@example.com"], "phone": ["+1-555-0123"]}
    contact = extract_contact_info(text, ner_entities=ner)
    assert contact["email"] == "john@example.com"
    assert contact["phone"] == "+1-555-0123"


# ---------------------------------------------------------------------------
# Integration test (loads real model — skip by default)
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_real_model_produces_entities():
    """Load real resume NER model and verify it produces output."""
    text = (
        "John Smith\n"
        "john.smith@email.com\n"
        "(555) 123-4567\n\n"
        "Senior Software Engineer with 8 years of experience\n\n"
        "Education\n"
        "Bachelor of Science in Computer Science, MIT, 2015\n\n"
        "Experience\n"
        "Google, Senior Software Engineer, 2019 - Present\n"
        "Amazon, Software Engineer, 2015 - 2019\n\n"
        "Skills\n"
        "Python, Java, Kubernetes, AWS, Docker"
    )
    result = extract_resume_entities(text)
    # Model should find at least some entities
    all_entities = sum(len(v) for v in result.values())
    assert all_entities > 0, f"Model produced no entities: {result}"
    # Should have the right keys
    assert set(result.keys()) == set(_empty_entities().keys())
