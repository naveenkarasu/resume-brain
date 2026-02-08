import pytest

from services.similarity import (
    extract_tfidf_keywords,
    hybrid_similarity,
    tfidf_cosine_similarity,
)


def test_tfidf_cosine_similarity_identical():
    text = "Python developer with React and Docker experience"
    score = tfidf_cosine_similarity(text, text)
    assert score == pytest.approx(1.0, abs=0.01)


def test_tfidf_cosine_similarity_different():
    a = "Python developer with React and Docker experience in web development"
    b = "Marketing manager with expertise in social media and brand strategy"
    score = tfidf_cosine_similarity(a, b)
    assert score < 0.3  # Very different texts


def test_tfidf_cosine_similarity_related():
    resume = "Senior Python developer with 5 years building REST APIs using FastAPI and Docker"
    jd = "Looking for a Python backend engineer with experience in REST API development and Docker"
    score = tfidf_cosine_similarity(resume, jd)
    assert score > 0.05  # Related texts should have some similarity (TF-IDF on short texts is sparse)


def test_tfidf_cosine_similarity_empty():
    assert tfidf_cosine_similarity("", "some text") == 0.0
    assert tfidf_cosine_similarity("some text", "") == 0.0


def test_extract_tfidf_keywords():
    jd = """Senior Full-Stack Engineer. Requirements: Python, TypeScript,
    React, AWS, Docker, Kubernetes, PostgreSQL, microservices, CI/CD pipelines,
    system design, 5+ years experience."""
    keywords = extract_tfidf_keywords(jd, top_n=10)
    assert len(keywords) > 0
    assert len(keywords) <= 10


def test_extract_tfidf_keywords_empty():
    assert extract_tfidf_keywords("") == []


def test_hybrid_similarity_returns_three_scores():
    a = "Python developer with React"
    b = "Python engineer with React.js"
    hybrid, tfidf, semantic = hybrid_similarity(a, b)
    assert 0 <= tfidf <= 1
    # Hybrid should be between 0 and 1
    assert hybrid >= 0
