from services.section_parser import (
    compute_experience_match,
    compute_section_completeness,
    extract_contact_info,
    extract_education_level,
    extract_experience_years,
    extract_required_years,
    parse_sections,
)


SAMPLE_RESUME = """John Doe
john.doe@email.com | (555) 123-4567
linkedin.com/in/johndoe | github.com/johndoe

Summary
Experienced software engineer with 5+ years building web applications.

Experience
Senior Software Engineer | TechCorp | 2021 - Present
• Built REST APIs serving 1M requests/day
• Led team of 5 engineers

Software Engineer | StartupXYZ | 2019 - 2021
• Developed React frontend components
• Implemented CI/CD pipelines

Education
B.S. Computer Science | State University | 2019

Skills
Python, JavaScript, React, Docker, AWS, PostgreSQL, Git
"""


def test_parse_sections_detects_all():
    sections = parse_sections(SAMPLE_RESUME)
    assert "summary" in sections
    assert "experience" in sections
    assert "education" in sections
    assert "skills" in sections
    assert "header" in sections


def test_parse_sections_content():
    sections = parse_sections(SAMPLE_RESUME)
    assert "REST APIs" in sections["experience"]
    assert "Computer Science" in sections["education"]
    assert "Python" in sections["skills"]


def test_parse_sections_empty():
    sections = parse_sections("")
    assert len(sections) <= 1  # At most 'header' with empty content


def test_extract_contact_info():
    contact = extract_contact_info(SAMPLE_RESUME)
    assert contact["email"] == "john.doe@email.com"
    assert contact["linkedin"] == "linkedin.com/in/johndoe"
    assert contact["github"] == "github.com/johndoe"


def test_compute_section_completeness_full():
    sections = parse_sections(SAMPLE_RESUME)
    score = compute_section_completeness(sections)
    # Has experience + education + skills + summary = high score
    assert score >= 0.6


def test_compute_section_completeness_partial():
    text = """Summary
Some summary text

Skills
Python, Java
"""
    sections = parse_sections(text)
    score = compute_section_completeness(sections)
    assert 0.0 < score < 0.6  # Missing experience, education


# --- Experience extraction tests ---

def test_extract_experience_years_explicit():
    text = "Senior engineer with 5+ years of experience in Python."
    years = extract_experience_years(text)
    assert years >= 5.0


def test_extract_experience_years_date_ranges():
    text = """
    Software Engineer | TechCorp | Jan 2020 - Present
    Junior Developer | StartupXYZ | Mar 2018 - Dec 2019
    """
    years = extract_experience_years(text)
    assert years >= 4.0


def test_extract_required_years():
    jd = "Requirements: 5+ years of experience in software development"
    years = extract_required_years(jd)
    assert years == 5.0


def test_compute_experience_match_exceeds():
    score = compute_experience_match(7.0, 5.0)
    assert score >= 80


def test_compute_experience_match_below():
    score = compute_experience_match(2.0, 5.0)
    assert score < 60


def test_compute_experience_match_no_requirement():
    score = compute_experience_match(5.0, 0.0)
    assert score >= 70


# --- Education level tests ---

def test_extract_education_level_bachelors():
    assert extract_education_level("B.S. Computer Science") == "bachelors"
    assert extract_education_level("Bachelor's in Engineering") == "bachelors"


def test_extract_education_level_masters():
    assert extract_education_level("M.S. in Data Science") == "masters"
    assert extract_education_level("Master's degree in CS") == "masters"


def test_extract_education_level_phd():
    assert extract_education_level("Ph.D. in Machine Learning") == "phd"


def test_extract_education_level_none():
    assert extract_education_level("Some random text") == ""


def test_extract_education_level_highest():
    """Should return the highest degree found."""
    text = "B.S. from MIT, M.S. from Stanford, Ph.D. from Berkeley"
    assert extract_education_level(text) == "phd"
