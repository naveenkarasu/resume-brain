"""Keyword extraction and matching for resume-JD analysis.

Combines TF-IDF-based dynamic keyword discovery with a curated
fallback dictionary, skill synonym resolution, fuzzy matching,
and JD section segmentation for comprehensive coverage.
"""

import logging
import re
from collections import Counter

from nltk.stem import WordNetLemmatizer
from rapidfuzz import fuzz

from services.similarity import extract_tfidf_keywords

logger = logging.getLogger(__name__)

_lemmatizer = WordNetLemmatizer()

# ---------------------------------------------------------------------------
# JD boilerplate filtering: non-technical terms TF-IDF often picks up
# These are common in job descriptions but NOT job requirements/skills
# ---------------------------------------------------------------------------
JD_STOPWORDS: frozenset[str] = frozenset({
    # Company / HR boilerplate
    "opportunity", "opportunities", "position", "positions", "role", "roles",
    "candidate", "candidates", "applicant", "applicants", "application",
    "applications", "employment", "employer", "employee", "employees",
    "company", "organization", "team", "teams", "department",
    # Compensation & benefits
    "compensation", "salary", "benefits", "bonus", "bonuses", "equity",
    "insurance", "401k", "pto", "vacation", "retirement",
    "medical", "dental", "vision",
    # Legal / EEO / privacy
    "privacy", "notice", "policy", "policies", "compliance",
    "equal", "discrimination", "disability", "veteran", "race", "color",
    "religion", "sex", "gender", "orientation", "national", "origin", "age",
    "genetic", "genetics", "protected", "status", "regard",
    "eeo", "affirmative", "accommodation", "accessible",
    # Generic JD filler
    "facing", "range", "related", "including", "including",
    "based", "preferred", "required", "minimum", "maximum",
    "experience", "qualified", "qualification", "qualifications",
    "responsible", "responsibilities", "requirement", "requirements",
    "description", "overview", "summary", "mission", "vision",
    "proud", "committed", "dedicated", "passionate", "exciting",
    "thriving", "innovative", "dynamic", "diverse", "inclusive",
    "competitive", "exceptional", "flexible", "remote", "hybrid",
    "onsite", "location", "office",
    # Generic action words that aren't skills
    "deliver", "manage", "create", "build", "develop", "maintain",
    "implement", "design", "support", "ensure", "provide", "engage",
    "communicate", "collaborate", "utilize", "leverage",
    "connect", "serve", "help", "join", "apply", "submit",
    # Common words that sneak through TF-IDF
    "job", "work", "working", "workers", "career", "careers",
    "people", "person", "individual", "individuals",
    "us", "our", "we", "will", "can", "may",
    "year", "years", "day", "days", "time",
    "great", "best", "good", "strong", "key", "core",
    "new", "first", "well", "also", "part",
    "full", "level", "senior", "junior", "mid", "staff",
    "processing", "information", "data",  # only when standalone, not "data structures"
})

# ---------------------------------------------------------------------------
# JD section headers that indicate boilerplate (not requirements)
# Text under these headings is stripped before keyword extraction
# ---------------------------------------------------------------------------
_JD_BOILERPLATE_PATTERNS: list[re.Pattern] = [
    re.compile(
        r"(?:^|\n)\s*(?:"
        r"(?:as\s+part\s+of\s+(?:our|the)\s+team|what\s+we\s+offer|"
        r"(?:our|the)\s+(?:benefits|perks|compensation)|"
        r"(?:salary|pay|compensation)\s+(?:range|information)|"
        r"equal\s+(?:opportunity|employment)|"
        r"privacy\s+(?:notice|policy)|"
        r"eeo\s+statement|"
        r"about\s+(?:us|the\s+company|ziprecruiter|our\s+mission)|"
        r"(?:our|the)\s+mission|"
        r"who\s+we\s+are|"
        r"depending\s+on\s+the\s+position)"
        r")",
        re.IGNORECASE | re.MULTILINE,
    ),
]

# ---------------------------------------------------------------------------
# Skill synonym mapping: aliases -> canonical form
# Applied BEFORE matching so "K8s" and "Kubernetes" both resolve to "kubernetes"
# ---------------------------------------------------------------------------
SKILL_SYNONYMS: dict[str, str] = {
    # JavaScript ecosystem
    "js": "javascript", "es6": "javascript", "es2015": "javascript",
    "ts": "typescript",
    "react.js": "react", "reactjs": "react", "react native": "react native",
    "vue.js": "vue", "vuejs": "vue",
    "angular.js": "angular", "angularjs": "angular",
    "node": "node.js", "nodejs": "node.js",
    "next": "next.js", "nextjs": "next.js",
    "nuxt": "nuxt", "nuxtjs": "nuxt",
    "express.js": "express", "expressjs": "express",
    # Python ecosystem
    "py": "python", "python3": "python",
    "sklearn": "scikit-learn",
    "tf": "tensorflow", "tensor flow": "tensorflow",
    "torch": "pytorch",
    "fastapi": "fastapi", "fast api": "fastapi",
    # Cloud & DevOps
    "k8s": "kubernetes", "kube": "kubernetes",
    "amazon web services": "aws", "amazon aws": "aws",
    "google cloud": "gcp", "google cloud platform": "gcp",
    "microsoft azure": "azure",
    "ci/cd": "ci/cd", "cicd": "ci/cd",
    "github action": "github actions", "gh actions": "github actions",
    "docker compose": "docker",
    "tf": "terraform",
    # Databases
    "postgres": "postgresql", "pg": "postgresql",
    "mongo": "mongodb", "mongo db": "mongodb",
    "mysql": "mysql", "my sql": "mysql",
    "ms sql": "sql server", "mssql": "sql server",
    "dynamo": "dynamodb", "dynamo db": "dynamodb",
    # Languages
    "c sharp": "c#", "csharp": "c#",
    "cpp": "c++", "c plus plus": "c++",
    "golang": "go",
    "rb": "ruby",
    # AI/ML
    "ml": "machine learning", "ai/ml": "machine learning",
    "dl": "deep learning",
    "nlp": "natural language processing",
    "cv": "computer vision",
    "gen ai": "generative ai", "genai": "generative ai",
    "large language model": "llm", "large language models": "llm",
    # Tools & methodologies
    "git": "git", "github": "github", "gitlab": "gitlab",
    "jira": "jira", "confluence": "confluence",
    "vs code": "vscode", "visual studio code": "vscode",
    "vi": "vim", "neovim": "vim",
    "rest api": "rest", "restful": "rest", "rest apis": "rest",
    "graphql": "graphql", "graph ql": "graphql",
    # Soft skills
    "pm": "project management", "project mgmt": "project management",
    "agile methodology": "agile", "agile/scrum": "agile",
}

# Supplementary keyword dictionary for common tech terms that TF-IDF might miss
# due to short document length. Used as a fallback layer, not the primary source.
COMMON_KEYWORDS = {
    # Programming languages
    "python", "javascript", "typescript", "java", "c++", "c#", "go", "rust",
    "ruby", "php", "swift", "kotlin", "scala", "r", "matlab", "sql",
    # Frontend
    "react", "react native", "angular", "vue", "svelte", "next.js", "nuxt",
    "html", "css", "tailwind", "bootstrap", "sass", "webpack", "vite",
    # Backend
    "node.js", "express", "fastapi", "django", "flask", "spring", "rails",
    ".net", "graphql", "rest", "grpc",
    # Cloud & DevOps
    "aws", "azure", "gcp", "docker", "kubernetes", "terraform", "ansible",
    "jenkins", "github actions", "ci/cd", "linux", "nginx",
    # Data
    "postgresql", "mysql", "mongodb", "redis", "elasticsearch", "kafka",
    "spark", "hadoop", "snowflake", "bigquery", "pandas", "numpy",
    # ML/AI
    "machine learning", "deep learning", "tensorflow", "pytorch",
    "natural language processing", "computer vision", "scikit-learn",
    "llm", "transformers", "generative ai",
    # Soft skills & methodologies
    "agile", "scrum", "kanban", "leadership", "communication",
    "problem-solving", "teamwork", "project management",
}

# Fuzzy match threshold (0-100). 80+ catches "Postgres" -> "PostgreSQL" etc.
FUZZY_THRESHOLD = 80


def _normalize(text: str, lemmatize: bool = False) -> str:
    """Normalize text for keyword matching.

    When lemmatize=True, applies WordNet lemmatization so
    'developing', 'developed', 'developer' all reduce to 'develop'.
    """
    # Strip sentence-ending periods but keep dots in tech terms like "node.js"
    text = re.sub(r"\.(\s|$)", " ", text.lower())
    normalized = re.sub(r"[^a-z0-9.#+/ -]", " ", text)
    if lemmatize:
        words = normalized.split()
        normalized = " ".join(_lemmatizer.lemmatize(w, pos="v") for w in words)
    return normalized


def _canonicalize(term: str) -> str:
    """Resolve a term to its canonical form via synonym dictionary."""
    lower = term.lower().strip()
    return SKILL_SYNONYMS.get(lower, lower)


def _canonicalize_set(terms: set[str]) -> set[str]:
    """Canonicalize all terms in a set."""
    return {_canonicalize(t) for t in terms}


def _extract_terms(text: str) -> set[str]:
    """Extract multi-word and single-word terms from text.

    Includes both raw normalized forms and lemmatized forms for broader matching.
    """
    normalized = _normalize(text)
    lemmatized = _normalize(text, lemmatize=True)
    words: set[str] = set()

    for norm_text in (normalized, lemmatized):
        word_list = norm_text.split()
        words.update(word_list)
        for i in range(len(word_list) - 1):
            words.add(f"{word_list[i]} {word_list[i+1]}")
        for i in range(len(word_list) - 2):
            words.add(f"{word_list[i]} {word_list[i+1]} {word_list[i+2]}")
    return words


def _extract_relevant_jd_sections(job_description: str) -> str:
    """Extract only the relevant sections from a job description.

    Strips boilerplate: benefits, EEO statements, privacy notices, company info.
    Keeps: responsibilities, requirements, qualifications, technical details.
    """
    text = job_description

    # Find the earliest boilerplate section and truncate there
    # This handles the common pattern where requirements come first, then boilerplate
    earliest_boilerplate = len(text)
    for pattern in _JD_BOILERPLATE_PATTERNS:
        match = pattern.search(text)
        if match:
            earliest_boilerplate = min(earliest_boilerplate, match.start())

    # Only truncate if we found boilerplate and there's meaningful content before it
    if earliest_boilerplate > 50 and earliest_boilerplate < len(text):
        text = text[:earliest_boilerplate]

    return text.strip()


def _is_technical_term(term: str) -> bool:
    """Check if a TF-IDF-extracted term is likely a technical/job-relevant keyword.

    Filters out generic JD boilerplate that TF-IDF picks up as 'important'.
    """
    # Single words: check against stopword list
    words = term.lower().split()
    if len(words) == 1:
        return words[0] not in JD_STOPWORDS and len(words[0]) > 1

    # Multi-word: reject if ALL words are stopwords
    if all(w in JD_STOPWORDS for w in words):
        return False

    # Reject if term matches known boilerplate phrases
    boilerplate_phrases = {
        "privacy notice", "equal opportunity", "employment opportunity",
        "job applicant", "personal data", "national origin",
        "gender identity", "sexual orientation", "veteran status",
        "total compensation", "base salary", "salary range",
        "full time", "part time", "paid time",
    }
    if term.lower() in boilerplate_phrases:
        return False

    return True


def extract_keywords_tfidf(job_description: str, top_n: int = 20) -> list[str]:
    """Extract important keywords from JD using TF-IDF (dynamic, not hardcoded).

    Pre-filters JD to remove boilerplate sections and post-filters
    results to remove non-technical terms.
    """
    # Pre-filter: extract only relevant JD sections
    relevant_jd = _extract_relevant_jd_sections(job_description)
    raw_keywords = extract_tfidf_keywords(relevant_jd or job_description, top_n=top_n * 2)

    # Post-filter: remove non-technical terms
    filtered = [kw for kw in raw_keywords if _is_technical_term(kw)]
    return filtered[:top_n]


def extract_keywords(job_description: str) -> set[str]:
    """Extract relevant keywords from a job description using the curated dictionary."""
    jd_terms = _extract_terms(job_description)
    canonical_jd = _canonicalize_set(jd_terms)
    return {kw for kw in COMMON_KEYWORDS if _canonicalize(kw) in canonical_jd or kw in jd_terms}


def extract_keywords_combined(job_description: str, top_n: int = 25) -> list[str]:
    """Extract keywords using both TF-IDF and dictionary methods."""
    tfidf_kws = set(extract_keywords_tfidf(job_description, top_n=top_n))
    dict_kws = extract_keywords(job_description)
    combined = sorted(dict_kws) + sorted(tfidf_kws - dict_kws)
    return combined[:top_n]


def _fuzzy_match(keyword: str, resume_terms: set[str], resume_lower: str) -> bool:
    """Check if keyword matches any resume term using exact, synonym, or fuzzy matching."""
    canon_kw = _canonicalize(keyword)

    # 1. Exact match (fastest path)
    if keyword in resume_terms or keyword.lower() in resume_lower:
        return True

    # 2. Canonical match (synonym resolution)
    canon_terms = _canonicalize_set(resume_terms)
    if canon_kw in canon_terms:
        return True

    # 3. Check if canonical form appears as substring in resume text
    if canon_kw in resume_lower:
        return True

    # 4. Fuzzy match (Levenshtein distance) for typos and close variants
    if len(canon_kw) >= 3:  # Don't fuzzy match very short terms
        for term in resume_terms:
            if len(term) >= 3 and fuzz.ratio(canon_kw, term) >= FUZZY_THRESHOLD:
                return True

    return False


def match_keywords(
    resume_text: str, job_keywords: set[str] | list[str]
) -> tuple[list[str], list[str]]:
    """Match resume text against job keywords using synonym + fuzzy matching."""
    resume_terms = _extract_terms(resume_text)
    resume_lower = resume_text.lower()

    if isinstance(job_keywords, list):
        job_keywords = set(job_keywords)

    matched = []
    missing = []
    for kw in sorted(job_keywords):
        if _fuzzy_match(kw, resume_terms, resume_lower):
            matched.append(kw)
        else:
            missing.append(kw)

    return matched, missing


def compute_keyword_overlap(matched: list[str], missing: list[str]) -> float:
    """Compute keyword overlap ratio as 0.0-1.0."""
    total = len(matched) + len(missing)
    if total == 0:
        return 0.5
    return len(matched) / total


def compute_keyword_density(
    resume_text: str, keywords: list[str]
) -> dict[str, float]:
    """Compute keyword density (frequency / total words) for each keyword.

    Returns dict of keyword -> density percentage.
    ATS optimal range: 1-3% per primary keyword.
    """
    words = resume_text.lower().split()
    total_words = len(words)
    if total_words == 0:
        return {}

    word_counts = Counter(words)
    densities = {}
    for kw in keywords:
        kw_lower = kw.lower()
        if " " in kw_lower:
            count = resume_text.lower().count(kw_lower)
        else:
            count = word_counts.get(kw_lower, 0)
        densities[kw] = round((count / total_words) * 100, 2)

    return densities


def compute_local_score(matched: list[str], missing: list[str]) -> int:
    """Compute a simple match percentage (legacy compatibility)."""
    total = len(matched) + len(missing)
    if total == 0:
        return 50
    return round((len(matched) / total) * 100)
