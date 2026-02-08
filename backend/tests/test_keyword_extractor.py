import pytest

from services.keyword_extractor import (
    _canonicalize,
    _extract_relevant_jd_sections,
    _is_technical_term,
    compute_keyword_density,
    compute_keyword_overlap,
    compute_local_score,
    extract_keywords,
    extract_keywords_combined,
    extract_keywords_tfidf,
    match_keywords,
)


def test_extract_keywords():
    jd = "We need a Python developer with experience in React and Docker."
    keywords = extract_keywords(jd)
    assert "python" in keywords
    assert "react" in keywords
    assert "docker" in keywords


def test_extract_keywords_tfidf():
    jd = """Senior Full-Stack Engineer
    Requirements: 5+ years Python, TypeScript, React, AWS, Docker,
    Kubernetes, PostgreSQL, CI/CD, microservices architecture"""
    keywords = extract_keywords_tfidf(jd, top_n=10)
    assert len(keywords) > 0
    assert any("python" in kw or "react" in kw or "kubernetes" in kw for kw in keywords)


def test_extract_keywords_combined():
    jd = "Looking for a Python developer with React and Docker experience."
    combined = extract_keywords_combined(jd, top_n=15)
    assert len(combined) > 0
    assert any("python" in kw for kw in combined)


def test_match_keywords():
    resume = "Experienced Python and React developer with 5 years of experience."
    job_keywords = {"python", "react", "docker", "kubernetes"}
    matched, missing = match_keywords(resume, job_keywords)
    assert "python" in matched
    assert "react" in matched
    assert "docker" in missing
    assert "kubernetes" in missing


def test_match_keywords_with_list():
    resume = "Python developer with AWS and Docker skills."
    job_keywords = ["python", "aws", "docker", "kubernetes"]
    matched, missing = match_keywords(resume, job_keywords)
    assert "python" in matched
    assert "aws" in matched
    assert "kubernetes" in missing


# --- Synonym matching tests ---

def test_canonicalize():
    assert _canonicalize("K8s") == "kubernetes"
    assert _canonicalize("JS") == "javascript"
    assert _canonicalize("ML") == "machine learning"
    assert _canonicalize("Postgres") == "postgresql"
    assert _canonicalize("ReactJS") == "react"
    assert _canonicalize("unknown_term") == "unknown_term"


def test_match_keywords_synonym_k8s():
    """K8s in resume should match kubernetes in JD."""
    resume = "Managed K8s clusters for microservices deployment."
    matched, missing = match_keywords(resume, {"kubernetes"})
    assert "kubernetes" in matched


def test_match_keywords_synonym_ml():
    """ML in resume should match machine learning in JD."""
    resume = "Senior ML Engineer with 5 years building ML pipelines."
    matched, missing = match_keywords(resume, {"machine learning"})
    assert "machine learning" in matched


def test_match_keywords_synonym_reactjs():
    """ReactJS in resume should match react in JD."""
    resume = "Built responsive UIs with ReactJS and TypeScript."
    matched, missing = match_keywords(resume, {"react"})
    assert "react" in matched


def test_match_keywords_synonym_postgres():
    """Postgres in resume should match postgresql in JD."""
    resume = "Optimized Postgres queries for high-traffic applications."
    matched, missing = match_keywords(resume, {"postgresql"})
    assert "postgresql" in matched


def test_match_keywords_synonym_nodejs():
    """NodeJS in resume should match node.js in JD."""
    resume = "Backend services built with NodeJS and Express."
    matched, missing = match_keywords(resume, {"node.js"})
    assert "node.js" in matched


# --- Fuzzy matching tests ---

def test_match_keywords_fuzzy_typo():
    """Fuzzy matching should catch close misspellings."""
    resume = "Experience with Kubernates and Docker containerization."
    matched, missing = match_keywords(resume, {"kubernetes"})
    # "kubernates" is close enough to "kubernetes" (fuzzy ratio > 80)
    assert "kubernetes" in matched


# --- Existing tests ---

def test_compute_keyword_overlap():
    assert compute_keyword_overlap(["a", "b"], ["c"]) == pytest.approx(2 / 3, abs=0.01)
    assert compute_keyword_overlap([], []) == 0.5
    assert compute_keyword_overlap(["a"], ["b"]) == 0.5


def test_compute_keyword_density():
    resume = "Python Python Python Java Java Ruby developer with Python skills"
    density = compute_keyword_density(resume, ["python", "java", "ruby"])
    assert density["python"] > density["java"]
    assert density["java"] > density["ruby"]


def test_compute_local_score():
    assert compute_local_score(["a", "b"], ["c"]) == 67
    assert compute_local_score([], []) == 50
    assert compute_local_score(["a"], ["b"]) == 50


# --- JD boilerplate filtering tests ---

def test_is_technical_term_filters_boilerplate():
    """Non-technical JD filler words should be filtered out."""
    assert _is_technical_term("python") is True
    assert _is_technical_term("react") is True
    assert _is_technical_term("kubernetes") is True
    assert _is_technical_term("facing") is False
    assert _is_technical_term("opportunity") is False
    assert _is_technical_term("compensation") is False
    assert _is_technical_term("position") is False
    assert _is_technical_term("range") is False
    assert _is_technical_term("applications") is False
    assert _is_technical_term("related") is False


def test_is_technical_term_filters_boilerplate_phrases():
    """Multi-word boilerplate phrases should be filtered."""
    assert _is_technical_term("privacy notice") is False
    assert _is_technical_term("equal opportunity") is False
    assert _is_technical_term("salary range") is False
    assert _is_technical_term("full stack") is True  # Technical multi-word
    assert _is_technical_term("data structures") is True


def test_extract_relevant_jd_sections_strips_benefits():
    """Should strip benefits/EEO/privacy sections from JD."""
    jd = """Requirements:
- Python and React experience
- 3+ years

As part of our team you'll enjoy:
- Competitive salary
- Great benefits

Privacy Notice: We collect personal data..."""

    relevant = _extract_relevant_jd_sections(jd)
    assert "Python" in relevant
    assert "React" in relevant
    assert "Competitive salary" not in relevant
    assert "Privacy Notice" not in relevant


def test_tfidf_excludes_boilerplate_from_ziprecruiter_jd():
    """Real-world test: ZipRecruiter JD should not extract boilerplate words."""
    jd = """Software Engineer, Full Stack

About the Job:
Design and implement user-facing web applications at scale.
Experience in client side development using ReactJS, Javascript, AngularJS.
Experience building large-scale user facing web applications.

As part of our team you'll enjoy:
Competitive compensation
Exceptional benefits package
Flexible Vacation & Paid Time Off

ZipRecruiter is proud to be an equal opportunity employer.

Privacy Notice: For information about ZipRecruiter's collection
and processing of job applicant personal data."""

    keywords = extract_keywords_tfidf(jd, top_n=15)
    kw_lower = [kw.lower() for kw in keywords]
    # Should NOT contain boilerplate
    assert "facing" not in kw_lower
    assert "opportunity" not in kw_lower
    assert "compensation" not in kw_lower
    assert "privacy" not in kw_lower
    assert "applications" not in kw_lower
