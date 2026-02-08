from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"


def test_analyze_quick():
    response = client.post(
        "/analyze/quick",
        json={
            "resume_text": "Experienced Python developer with React and Docker skills. Built REST APIs.",
            "job_description": "Looking for a Python developer with React experience and Docker.",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "overall_score" in data
    assert "missing_keywords" in data
    assert "matched_keywords" in data
    # New hybrid fields
    assert "keyword_density" in data
    assert isinstance(data["keyword_density"], dict)
    assert "tfidf_score" in data
    assert "semantic_score" in data
    assert "scoring_method" in data
    assert "section_analysis" in data
    assert data["tfidf_score"] >= 0
    assert data["scoring_method"] in ("hybrid", "local_only", "llm_only")
    # Phase 2-3 fields
    assert "experience_years" in data
    assert isinstance(data["experience_years"], (int, float))
    assert "bullet_scores" in data
    assert isinstance(data["bullet_scores"], list)
    assert "education_level" in data
    assert isinstance(data["education_level"], str)


def test_analyze_rejects_non_pdf():
    response = client.post(
        "/analyze",
        files={"resume_file": ("resume.txt", b"not a pdf", "text/plain")},
        data={"job_description": "test"},
    )
    assert response.status_code == 400
