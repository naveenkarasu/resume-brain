"""Resume section segmentation and contact extraction."""

import re

# Section header patterns and their canonical names
SECTION_PATTERNS: dict[str, list[str]] = {
    "experience": [
        r"(?:work|professional|employment)\s*(?:experience|history)",
        r"experience",
        r"career\s*(?:history|summary|path)",
        r"(?:positions?\s*held|roles)",
    ],
    "education": [
        r"education(?:al)?\s*(?:background|qualifications|history)?",
        r"academic\s*(?:background|qualifications)",
    ],
    "skills": [
        r"(?:technical|core|key|professional)?\s*skills",
        r"(?:technical|core)?\s*(?:competencies|proficiencies|expertise)",
        r"technologies",
        r"(?:technical\s+)?(?:stack|toolkit|tooling)",
        r"(?:programming\s+)?languages",
    ],
    "summary": [
        r"(?:professional|executive|career)?\s*summary",
        r"(?:career|professional)?\s*objective",
        r"profile",
        r"about\s*me",
    ],
    "projects": [
        r"(?:key|notable|selected|personal)?\s*projects",
        r"portfolio",
    ],
    "certifications": [
        r"certific(?:ations?|ates?)",
        r"licen[sc]es?\s*(?:&|and)?\s*certific(?:ations?|ates?)",
    ],
    "achievements": [
        r"(?:key\s+)?achievements?",
        r"(?:awards?|honors?|accomplishments)",
    ],
}

# Compile all patterns into a single regex per section
_COMPILED: dict[str, re.Pattern] = {}
for section, patterns in SECTION_PATTERNS.items():
    combined = "|".join(patterns)
    _COMPILED[section] = re.compile(
        rf"^\s*(?:{combined})\s*:?\s*$", re.IGNORECASE | re.MULTILINE
    )

# Contact info patterns
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
PHONE_RE = re.compile(r"\+?[\d\s\-().]{7,15}\d")
LINKEDIN_RE = re.compile(r"linkedin\.com/in/[\w-]+", re.IGNORECASE)
GITHUB_RE = re.compile(r"github\.com/[\w-]+", re.IGNORECASE)

# Weighted section importance for completeness scoring
SECTION_WEIGHTS: dict[str, float] = {
    "experience": 20,
    "skills": 15,
    "education": 12,
    "projects": 12,
    "summary": 10,
    "certifications": 8,
    "achievements": 5,
}
_TOTAL_WEIGHT = sum(SECTION_WEIGHTS.values())

# Keep legacy constant for backward compatibility
EXPECTED_SECTIONS = {"experience", "education", "skills"}


def parse_sections(text: str) -> dict[str, str]:
    """Split resume text into named sections.

    Returns a dict mapping section name -> section text content.
    Unmatched text at the top goes into 'header'.
    """
    lines = text.split("\n")
    sections: dict[str, str] = {}
    current_section = "header"
    current_lines: list[str] = []

    for line in lines:
        matched_section = None
        stripped = line.strip()

        # Skip empty lines for header detection, but keep them in content
        if stripped:
            for section_name, pattern in _COMPILED.items():
                if pattern.match(stripped):
                    matched_section = section_name
                    break

        if matched_section:
            # Save previous section
            if current_lines:
                sections[current_section] = "\n".join(current_lines).strip()
            current_section = matched_section
            current_lines = []
        else:
            current_lines.append(line)

    # Save last section
    if current_lines:
        sections[current_section] = "\n".join(current_lines).strip()

    return sections


def extract_contact_info(
    text: str, ner_entities: dict | None = None
) -> dict[str, str | None]:
    """Extract contact information from resume text.

    If ner_entities is provided, fills gaps from NER when regex misses them.
    """
    email_match = EMAIL_RE.search(text)
    phone_match = PHONE_RE.search(text)
    linkedin_match = LINKEDIN_RE.search(text)
    github_match = GITHUB_RE.search(text)

    email = email_match.group() if email_match else None
    phone = phone_match.group().strip() if phone_match else None

    # Fill gaps from NER entities if regex missed
    if ner_entities:
        if email is None and ner_entities.get("email"):
            email = ner_entities["email"][0]
        if phone is None and ner_entities.get("phone"):
            phone = ner_entities["phone"][0]

    return {
        "email": email,
        "phone": phone,
        "linkedin": linkedin_match.group() if linkedin_match else None,
        "github": github_match.group() if github_match else None,
    }


def compute_section_completeness(sections: dict[str, str]) -> float:
    """Score 0.0-1.0 based on weighted importance of present sections."""
    found_weight = sum(
        SECTION_WEIGHTS[s] for s in SECTION_WEIGHTS if s in sections
    )
    return round(found_weight / _TOTAL_WEIGHT, 3)


# ---------------------------------------------------------------------------
# Experience duration extraction
# ---------------------------------------------------------------------------

# "5+ years of experience" or "3 years experience in Python"
EXP_YEARS_RE = re.compile(
    r"(\d+)\+?\s*(?:years?|yrs?)\s*(?:of\s+)?(?:experience|exp\b)",
    re.IGNORECASE,
)

# Date ranges: "Jan 2019 - Present", "2020 - 2023", "March 2018 – Nov 2022"
_MONTHS = (
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
)
DATE_RANGE_RE = re.compile(
    rf"({_MONTHS}\.?\s*\d{{4}}|\d{{4}})"
    r"\s*[-–—to]+\s*"
    rf"({_MONTHS}\.?\s*\d{{4}}|\d{{4}}|[Pp]resent|[Cc]urrent)",
    re.IGNORECASE,
)

_MONTH_MAP = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6,
    "jul": 7, "july": 7, "aug": 8, "august": 8, "sep": 9, "sept": 9,
    "september": 9, "oct": 10, "october": 10, "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}


def _parse_date(date_str: str) -> tuple[int, int]:
    """Parse a date string into (year, month). Returns (year, 1) if month not found."""
    from datetime import datetime

    date_str = date_str.strip().rstrip(".")
    if date_str.lower() in ("present", "current"):
        now = datetime.now()
        return now.year, now.month

    # Try "Month Year" format
    parts = date_str.split()
    if len(parts) == 2:
        month_str = parts[0].lower().rstrip(".")
        if month_str in _MONTH_MAP:
            try:
                return int(parts[1]), _MONTH_MAP[month_str]
            except ValueError:
                pass

    # Try bare year
    try:
        year = int(date_str)
        if 1970 <= year <= 2100:
            return year, 1
    except ValueError:
        pass

    return 0, 0


def extract_experience_years(
    text: str, ner_entities: dict | None = None
) -> float:
    """Extract total years of experience from resume text.

    Uses two strategies (plus optional NER):
    1. Explicit claims: "5+ years of experience"
    2. Date range calculation: sum of all role date ranges
    3. NER years_of_experience strings (parsed for digits)

    Returns the highest of all estimates.
    """
    # Strategy 1: Explicit claims
    explicit_years = 0.0
    for match in EXP_YEARS_RE.finditer(text):
        years = int(match.group(1))
        if years > explicit_years:
            explicit_years = float(years)

    # Strategy 2: Date range summation
    total_months = 0
    for match in DATE_RANGE_RE.finditer(text):
        start_year, start_month = _parse_date(match.group(1))
        end_year, end_month = _parse_date(match.group(2))
        if start_year > 0 and end_year > 0:
            months = (end_year - start_year) * 12 + (end_month - start_month)
            if 0 < months < 600:  # Sanity: < 50 years
                total_months += months

    date_years = round(total_months / 12, 1) if total_months > 0 else 0.0

    # Strategy 3: NER years_of_experience
    ner_years = 0.0
    if ner_entities and ner_entities.get("years_of_experience"):
        for yoe_str in ner_entities["years_of_experience"]:
            digits = re.findall(r"\d+", yoe_str)
            for d in digits:
                val = float(d)
                if val > ner_years and val < 60:
                    ner_years = val

    return max(explicit_years, date_years, ner_years)


def extract_required_years(job_description: str) -> float:
    """Extract required years of experience from a job description."""
    best = 0.0
    for match in EXP_YEARS_RE.finditer(job_description):
        years = float(match.group(1))
        if years > best:
            best = years
    return best


def compute_experience_match(resume_years: float, required_years: float) -> int:
    """Score 0-100 comparing resume experience to JD requirement."""
    if required_years <= 0:
        # No explicit requirement — give a neutral score based on resume
        if resume_years >= 5:
            return 85
        elif resume_years >= 2:
            return 70
        return 55

    ratio = resume_years / required_years
    if ratio >= 1.0:
        return min(100, round(80 + ratio * 10))
    elif ratio >= 0.7:
        return round(60 + (ratio - 0.7) * 66)
    elif ratio >= 0.4:
        return round(30 + (ratio - 0.4) * 100)
    else:
        return round(ratio * 75)


# ---------------------------------------------------------------------------
# Education level detection
# ---------------------------------------------------------------------------

DEGREE_PATTERNS: dict[str, list[str]] = {
    "phd": [
        r"ph\.?d", r"doctorate", r"doctoral", r"doctor of philosophy",
    ],
    "masters": [
        r"m\.?s\.?", r"m\.?sc\.?", r"m\.?e\.?", r"m\.?tech", r"mba",
        r"m\.?a\.?(?:\s|$)", r"master(?:'?s)?",
    ],
    "bachelors": [
        r"b\.?s\.?", r"b\.?sc\.?", r"b\.?e\.?", r"b\.?tech", r"b\.?a\.?(?:\s|$)",
        r"bachelor(?:'?s)?", r"b\.?eng",
    ],
    "associate": [
        r"a\.?s\.?", r"a\.?a\.?(?:\s|$)", r"associate(?:'?s)?",
    ],
}

_DEGREE_COMPILED: dict[str, re.Pattern] = {}
for _level, _patterns in DEGREE_PATTERNS.items():
    _combined = "|".join(_patterns)
    _DEGREE_COMPILED[_level] = re.compile(
        rf"\b(?:{_combined})\b", re.IGNORECASE
    )

# Order matters: check highest first
_DEGREE_PRIORITY = ["phd", "masters", "bachelors", "associate"]


def extract_education_level(
    text: str, ner_entities: dict | None = None
) -> str:
    """Detect the highest education level mentioned in text.

    If ner_entities is provided, checks NER degree strings first using
    the same degree patterns, then falls through to full-text regex.

    Returns one of: 'phd', 'masters', 'bachelors', 'associate', or '' if none found.
    """
    # Check NER degree strings first (more targeted)
    if ner_entities and ner_entities.get("degrees"):
        for degree_str in ner_entities["degrees"]:
            for level in _DEGREE_PRIORITY:
                if _DEGREE_COMPILED[level].search(degree_str):
                    return level

    # Fall through to full-text regex
    for level in _DEGREE_PRIORITY:
        if _DEGREE_COMPILED[level].search(text):
            return level
    return ""
