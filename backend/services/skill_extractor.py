"""ML-powered skill extraction using JobBERT NER + pattern matching.

Combines:
1. NER model (jjzha/jobbert_skill_extraction) for soft skill spans
2. Pattern-based extraction for hard technical skills in structured formats
3. Existing synonym/canonical mapping for normalization
"""

import logging
import re

logger = logging.getLogger(__name__)

# Lazy-loaded NER pipeline
_skill_ner = None


def _get_skill_ner():
    """Load skill extraction NER model lazily."""
    global _skill_ner
    if _skill_ner is None:
        try:
            from transformers import pipeline

            _skill_ner = pipeline(
                "token-classification",
                model="jjzha/jobbert_skill_extraction",
                aggregation_strategy="simple",
            )
            logger.info("Skill NER model loaded successfully")
        except Exception as e:
            logger.warning("Failed to load skill NER model: %s", e)
    return _skill_ner


# ---------------------------------------------------------------------------
# Technical skill patterns: catch skills in structured formats
# Matches: "Python, JavaScript, React" / "Python/JavaScript" / "Python & React"
# ---------------------------------------------------------------------------
_TECH_SKILL_PATTERN = re.compile(
    r"(?:^|[,;|•·▪▸►◦‣–—\n])\s*"
    r"([A-Za-z][A-Za-z0-9.#+/ -]{1,30})"
    r"(?=[,;|•·▪▸►◦‣–—\n]|$)",
)

# Known technical skill vocabulary (broader than COMMON_KEYWORDS for extraction)
TECH_SKILLS: frozenset[str] = frozenset({
    # Programming languages
    "python", "javascript", "typescript", "java", "c++", "c#", "c",
    "go", "golang", "rust", "ruby", "php", "swift", "kotlin", "scala",
    "r", "matlab", "sql", "perl", "haskell", "lua", "dart", "elixir",
    "clojure", "groovy", "objective-c", "shell", "bash", "powershell",
    # Frontend
    "react", "react native", "angular", "vue", "vue.js", "svelte",
    "next.js", "nextjs", "nuxt", "nuxtjs", "gatsby",
    "html", "html5", "css", "css3", "tailwind", "tailwindcss",
    "bootstrap", "sass", "scss", "less", "webpack", "vite",
    "jquery", "backbone", "ember",
    # Backend
    "node.js", "nodejs", "express", "fastapi", "django", "flask",
    "spring", "spring boot", "rails", "ruby on rails",
    ".net", "asp.net", "asp.net core",
    "graphql", "rest", "restful", "grpc", "soap",
    "laravel", "symfony", "gin", "echo", "fiber",
    # Cloud & DevOps
    "aws", "azure", "gcp", "google cloud", "heroku", "vercel", "netlify",
    "docker", "kubernetes", "k8s", "terraform", "ansible", "puppet", "chef",
    "jenkins", "github actions", "gitlab ci", "circleci", "travis ci",
    "ci/cd", "cicd", "linux", "unix", "nginx", "apache",
    "cloudformation", "helm", "istio", "prometheus", "grafana",
    "datadog", "new relic", "splunk", "elk",
    # Databases
    "postgresql", "postgres", "mysql", "mongodb", "redis",
    "elasticsearch", "kafka", "rabbitmq",
    "sqlite", "oracle", "sql server", "mssql",
    "dynamodb", "cassandra", "couchdb", "neo4j",
    "snowflake", "bigquery", "redshift",
    "firebase", "supabase",
    # Data & ML
    "pandas", "numpy", "scipy", "matplotlib",
    "scikit-learn", "sklearn", "tensorflow", "pytorch", "keras",
    "spark", "hadoop", "airflow", "dbt",
    "tableau", "power bi", "looker",
    "machine learning", "deep learning",
    "natural language processing", "nlp",
    "computer vision", "cv",
    "llm", "transformers", "generative ai",
    "hugging face", "langchain", "openai",
    # Tools & platforms
    "git", "github", "gitlab", "bitbucket",
    "jira", "confluence", "slack",
    "vscode", "visual studio", "intellij", "eclipse",
    "postman", "swagger", "openapi",
    "figma", "sketch", "adobe xd",
    # Security
    "oauth", "oauth2", "jwt", "saml", "ldap",
    "ssl", "tls", "https", "ssh",
    "owasp", "penetration testing",
    # Methodologies
    "agile", "scrum", "kanban", "waterfall",
    "tdd", "bdd", "pair programming",
    "microservices", "serverless", "event-driven",
    "design patterns", "solid",
    # Mobile
    "android", "ios", "flutter", "xamarin",
    "react native", "ionic",
    # Testing
    "jest", "mocha", "cypress", "selenium", "playwright",
    "pytest", "unittest", "junit", "testng",
})


def _normalize_skill(skill: str) -> str:
    """Normalize a skill string for comparison."""
    return re.sub(r"\s+", " ", skill.lower().strip().rstrip(".,:;"))


def _chunk_for_ner(text: str, max_words: int = 300) -> list[str]:
    """Split text into chunks that fit within BERT's 512 token limit."""
    paragraphs = re.split(r"\n\n+", text.strip())
    chunks: list[str] = []
    current: list[str] = []
    current_words = 0
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        pw = len(para.split())
        if current_words + pw > max_words and current:
            chunks.append("\n\n".join(current))
            current = [para]
            current_words = pw
        else:
            current.append(para)
            current_words += pw
    if current:
        chunks.append("\n\n".join(current))
    return chunks or [text[:2000]]


def extract_skills_ner(text: str) -> list[str]:
    """Extract soft skill spans using the NER model."""
    ner = _get_skill_ner()
    if ner is None:
        return []

    try:
        skills = []
        for chunk in _chunk_for_ner(text):
            results = ner(chunk)
            for r in results:
                if r["score"] > 0.6:
                    word = r["word"].strip()
                    if len(word) > 2 and not word.lower().startswith("##"):
                        skills.append(word)
        return skills
    except Exception as e:
        logger.warning("Skill NER extraction failed: %s", e)
        return []


def extract_skills_pattern(text: str) -> set[str]:
    """Extract technical skills using vocabulary matching.

    Scans the text for known technical terms from TECH_SKILLS vocabulary.
    Uses word boundary matching to avoid false positives (e.g. "gin" in "engineer").
    """
    text_lower = text.lower()
    found: set[str] = set()

    for skill in TECH_SKILLS:
        escaped = re.escape(skill)
        # Word boundary matching for all skills to prevent substring matches
        # e.g. "java" should NOT match inside "javascript"
        # e.g. "gin" should NOT match inside "engineer"
        if re.search(rf"(?<![a-zA-Z0-9.#]){escaped}(?![a-zA-Z0-9])", text_lower):
            found.add(skill)

    return found


def extract_skills_combined(text: str) -> set[str]:
    """Extract all skills from text using both NER and pattern matching.

    Returns normalized, deduplicated skill set.
    """
    # Pattern-based extraction (hard technical skills)
    pattern_skills = extract_skills_pattern(text)

    # NER extraction (soft skills and competency phrases)
    ner_skills = extract_skills_ner(text)

    # Combine and normalize
    all_skills: set[str] = set()
    for s in pattern_skills:
        all_skills.add(_normalize_skill(s))
    for s in ner_skills:
        norm = _normalize_skill(s)
        # Only add NER skills that are meaningful (>2 chars, not just a number)
        if len(norm) > 2 and not norm.isdigit():
            all_skills.add(norm)

    return all_skills


# Skill hierarchy: child skill implies parent skills
SKILL_IMPLICATIONS: dict[str, list[str]] = {
    "react": ["javascript"],
    "next.js": ["react", "javascript"],
    "angular": ["javascript", "typescript"],
    "vue": ["javascript"],
    "svelte": ["javascript"],
    "gatsby": ["react", "javascript"],
    "django": ["python"],
    "flask": ["python"],
    "fastapi": ["python"],
    "pytorch": ["python"],
    "tensorflow": ["python"],
    "keras": ["python", "tensorflow"],
    "scikit-learn": ["python"],
    "pandas": ["python"],
    "numpy": ["python"],
    "kubernetes": ["docker"],
    "spring boot": ["java"],
    "spring": ["java"],
    "rails": ["ruby"],
    "ruby on rails": ["ruby"],
    "flutter": ["dart"],
    "react native": ["react", "javascript"],
    "express": ["node.js", "javascript"],
    "nuxt": ["vue", "javascript"],
    "laravel": ["php"],
    "symfony": ["php"],
}


def _expand_skills(skills: set[str]) -> set[str]:
    """Expand skills with implied parent skills from SKILL_IMPLICATIONS."""
    expanded = set(skills)
    for skill in skills:
        implied = SKILL_IMPLICATIONS.get(skill, [])
        expanded.update(implied)
    return expanded


def compute_skill_overlap(
    resume_skills: set[str],
    jd_skills: set[str],
    required_skills: set[str] | None = None,
) -> float:
    """Compute skill overlap ratio between resume and JD skills.

    Features:
    - Expands resume_skills with implied parent skills
    - Full credit (1.0) for direct matches, half credit (0.5) for implied-only matches
    - Required skills get 2x weight

    Returns 0.0-1.0 indicating weighted match quality.
    """
    if not jd_skills:
        return 0.5

    expanded_resume = _expand_skills(resume_skills)
    if required_skills is None:
        required_skills = set()

    total_weight = 0.0
    earned_weight = 0.0

    for skill in jd_skills:
        weight = 2.0 if skill in required_skills else 1.0
        total_weight += weight

        if skill in resume_skills:
            # Direct match: full credit
            earned_weight += weight
        elif skill in expanded_resume:
            # Implied match: half credit
            earned_weight += weight * 0.5

    return earned_weight / total_weight if total_weight > 0 else 0.5


def get_skill_gap(
    resume_skills: set[str], jd_skills: set[str]
) -> tuple[set[str], set[str]]:
    """Get matched and missing skills between resume and JD.

    Returns (matched_skills, missing_skills).
    """
    matched = resume_skills & jd_skills
    missing = jd_skills - resume_skills
    return matched, missing
