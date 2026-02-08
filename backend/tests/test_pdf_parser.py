from services.pdf_parser import extract_bullets, score_bullet, score_bullets


def test_extract_bullets():
    text = """John Doe
Software Engineer

Experience:
• Built REST APIs serving 1M requests/day
- Led team of 5 engineers
* Improved test coverage from 40% to 90%
1. Deployed microservices on Kubernetes

Skills:
Python, JavaScript, Docker
"""
    bullets = extract_bullets(text)
    assert len(bullets) == 4
    assert "Built REST APIs serving 1M requests/day" in bullets
    assert "Led team of 5 engineers" in bullets


def test_extract_bullets_empty():
    assert extract_bullets("") == []
    assert extract_bullets("No bullets here\nJust plain text") == []


def test_extract_bullets_unicode_markers():
    text = "◆ Designed CI/CD pipeline\n■ Automated testing process\n→ Reduced deploy time"
    bullets = extract_bullets(text)
    assert len(bullets) == 3
    assert "Designed CI/CD pipeline" in bullets


def test_extract_bullets_two_digit_numbered():
    text = "10. Managed Kubernetes cluster\n12. Wrote integration tests"
    bullets = extract_bullets(text)
    assert len(bullets) == 2


# --- Bullet scoring tests ---


def test_score_bullet_high_quality():
    """Bullet with action verb + metrics + good length should score high."""
    result = score_bullet(
        "Built REST APIs serving 1M requests/day reducing latency by 40%",
        jd_keywords={"rest", "api"},
    )
    assert result["has_action_verb"] is True
    assert result["has_metrics"] is True
    assert result["quality_score"] >= 65


def test_score_bullet_action_verb_only():
    result = score_bullet("Led team meetings weekly")
    assert result["has_action_verb"] is True
    assert result["has_metrics"] is False
    assert result["quality_score"] >= 30


def test_score_bullet_no_action_verb():
    result = score_bullet("Responsible for maintaining the codebase")
    assert result["has_action_verb"] is False
    assert result["quality_score"] < 50


def test_score_bullet_with_metrics():
    result = score_bullet("Increased revenue by 25% through optimization")
    assert result["has_metrics"] is True
    assert result["quality_score"] >= 35


def test_score_bullet_keyword_match():
    result = score_bullet(
        "Developed Python microservices using Docker and Kubernetes",
        jd_keywords={"python", "docker", "kubernetes"},
    )
    assert result["keyword_count"] >= 2
    assert result["quality_score"] >= 20


def test_score_bullets_limits_to_15():
    bullets = [f"Built feature {i}" for i in range(20)]
    results = score_bullets(bullets)
    assert len(results) == 15


def test_score_bullets_returns_dicts():
    bullets = ["Led migration to cloud infrastructure"]
    results = score_bullets(bullets)
    assert len(results) == 1
    assert "text" in results[0]
    assert "quality_score" in results[0]
    assert "has_action_verb" in results[0]
