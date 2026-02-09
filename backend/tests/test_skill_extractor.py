"""Tests for ML skill extraction service."""

from services.skill_extractor import (
    _expand_skills,
    compute_skill_overlap,
    extract_skills_combined,
    extract_skills_pattern,
    get_skill_gap,
)


def test_extract_skills_pattern_finds_python():
    skills = extract_skills_pattern("Experience with Python and JavaScript")
    assert "python" in skills
    assert "javascript" in skills


def test_extract_skills_pattern_avoids_substring_false_positives():
    skills = extract_skills_pattern("Senior Software Engineer built scalable systems")
    assert "gin" not in skills  # "gin" should not match inside "engineer"
    assert "scala" not in skills  # "scala" should not match inside "scalable"


def test_extract_skills_pattern_java_not_in_javascript():
    skills = extract_skills_pattern("Proficient in JavaScript and TypeScript")
    assert "javascript" in skills
    assert "java" not in skills  # "java" should not match inside "javascript"


def test_extract_skills_pattern_multiword():
    skills = extract_skills_pattern("Knowledge of machine learning and natural language processing")
    assert "machine learning" in skills
    assert "natural language processing" in skills


def test_extract_skills_pattern_dotted_names():
    skills = extract_skills_pattern("Built APIs with Node.js and Next.js")
    assert "node.js" in skills
    assert "next.js" in skills


def test_extract_skills_combined_includes_both_sources():
    text = "Python developer with strong communication skills and teamwork"
    skills = extract_skills_combined(text)
    assert "python" in skills


def test_compute_skill_overlap_full_match():
    assert compute_skill_overlap({"python", "java"}, {"python", "java"}) == 1.0


def test_compute_skill_overlap_partial():
    overlap = compute_skill_overlap({"python"}, {"python", "java"})
    assert overlap == 0.5


def test_compute_skill_overlap_no_match():
    assert compute_skill_overlap({"ruby"}, {"python", "java"}) == 0.0


def test_compute_skill_overlap_empty_jd():
    assert compute_skill_overlap({"python"}, set()) == 0.5


def test_get_skill_gap():
    matched, missing = get_skill_gap(
        {"python", "docker", "aws"},
        {"python", "java", "docker", "kubernetes"},
    )
    assert matched == {"python", "docker"}
    assert missing == {"java", "kubernetes"}


# --- Skill hierarchy / implications tests ---


def test_expand_skills_react_implies_javascript():
    expanded = _expand_skills({"react"})
    assert "javascript" in expanded
    assert "react" in expanded


def test_expand_skills_nextjs_implies_chain():
    expanded = _expand_skills({"next.js"})
    assert "react" in expanded
    assert "javascript" in expanded


def test_expand_skills_no_implication():
    expanded = _expand_skills({"python"})
    assert expanded == {"python"}


def test_compute_skill_overlap_implied_gives_half_credit():
    """React implies JavaScript — if JD wants JavaScript, resume with React gets 0.5."""
    overlap = compute_skill_overlap({"react"}, {"javascript"})
    assert overlap == 0.5  # implied match = half credit


def test_compute_skill_overlap_direct_beats_implied():
    """Direct match should give full credit."""
    overlap = compute_skill_overlap({"javascript", "react"}, {"javascript"})
    assert overlap == 1.0  # direct match


def test_compute_skill_overlap_required_skills_weighted():
    """Required skills should get 2x weight."""
    # Resume has python (direct) but not react (missing required)
    # JD: python (regular=1.0), react (required=2.0)
    # Score = (1*1.0 + 0*2.0) / (1+2) = 1/3
    overlap = compute_skill_overlap(
        {"python"}, {"python", "react"}, required_skills={"react"}
    )
    # python: direct match, weight=1 -> 1.0
    # react: no match, weight=2 -> 0.0
    # total_weight=3, earned=1 -> 1/3
    assert abs(overlap - 1 / 3) < 0.01
