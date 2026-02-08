import io
import re

import pdfplumber

# Bullet markers: standard + expanded unicode set
BULLET_MARKERS = frozenset("•-–—►▪✓*○◆⚫→▸▹◇■□●")

# Strong action verbs for bullet quality scoring
ACTION_VERBS = frozenset({
    "achieved", "administered", "advanced", "analyzed", "architected",
    "automated", "built", "collaborated", "conducted", "configured",
    "consolidated", "contributed", "coordinated", "created", "decreased",
    "delivered", "deployed", "designed", "developed", "directed",
    "drove", "eliminated", "enabled", "engineered", "enhanced",
    "established", "evaluated", "executed", "expanded", "facilitated",
    "founded", "generated", "grew", "identified", "implemented",
    "improved", "increased", "influenced", "initiated", "innovated",
    "integrated", "introduced", "launched", "led", "leveraged",
    "maintained", "managed", "mentored", "migrated", "modernized",
    "negotiated", "optimized", "orchestrated", "organized", "overhauled",
    "partnered", "performed", "pioneered", "planned", "presented",
    "processed", "produced", "programmed", "proposed", "published",
    "rebuilt", "reduced", "refactored", "refined", "remodeled",
    "resolved", "restructured", "revamped", "scaled", "secured",
    "simplified", "spearheaded", "standardized", "streamlined",
    "strengthened", "supervised", "surpassed", "tested", "trained",
    "transformed", "tripled", "upgraded", "utilized",
})

# Regex for quantified metrics
_METRICS_RE = re.compile(r"\d+[%$KMBx]|\$\d+|\d+\+?\s*(?:users|clients|requests|customers|endpoints|services|teams?|members?)", re.IGNORECASE)


def extract_text(pdf_bytes: bytes) -> str:
    """Extract all text from a PDF file."""
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        pages = [page.extract_text() or "" for page in pdf.pages]
    return "\n".join(pages).strip()


def extract_text_docx(docx_bytes: bytes) -> str:
    """Extract all text from a DOCX file."""
    try:
        from docx import Document
        doc = Document(io.BytesIO(docx_bytes))
        return "\n".join(p.text for p in doc.paragraphs).strip()
    except ImportError:
        raise ImportError("python-docx is required for DOCX support")


def extract_bullets(text: str) -> list[str]:
    """Extract bullet-point lines from resume text."""
    bullets = []
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        # Match lines starting with bullet markers
        if stripped[0] in BULLET_MARKERS:
            cleaned = stripped.lstrip("".join(BULLET_MARKERS) + " ").strip()
            if cleaned:
                bullets.append(cleaned)
        # Numbered bullets: "1.", "12.", "1)", "12)"
        elif re.match(r"^\d{1,2}[.)]\s", stripped):
            cleaned = re.sub(r"^\d{1,2}[.)]\s*", "", stripped).strip()
            if cleaned:
                bullets.append(cleaned)
    return bullets


def score_bullet(bullet: str, jd_keywords: set[str] | None = None) -> dict:
    """Score a single bullet point on quality rubric.

    Returns dict with individual quality signals and overall score (0-100).
    """
    words = bullet.split()
    word_count = len(words)

    # Check action verb
    has_action_verb = words[0].lower().rstrip("ed") in ACTION_VERBS or words[0].lower() in ACTION_VERBS if words else False

    # Check quantified metrics
    has_metrics = bool(_METRICS_RE.search(bullet))

    # Check length (15-30 words is optimal)
    length_ok = 10 <= word_count <= 35

    # Check keyword relevance
    keyword_count = 0
    if jd_keywords:
        bullet_lower = bullet.lower()
        keyword_count = sum(1 for kw in jd_keywords if kw.lower() in bullet_lower)

    # Compute weighted quality score
    score = 0
    if has_action_verb:
        score += 30
    if has_metrics:
        score += 35
    if length_ok:
        score += 15
    if keyword_count >= 2:
        score += 20
    elif keyword_count == 1:
        score += 10

    return {
        "text": bullet[:100],  # Truncate for response size
        "has_action_verb": has_action_verb,
        "has_metrics": has_metrics,
        "length_ok": length_ok,
        "keyword_count": keyword_count,
        "quality_score": min(100, score),
    }


def score_bullets(bullets: list[str], jd_keywords: set[str] | None = None) -> list[dict]:
    """Score all bullet points. Returns list of score dicts."""
    return [score_bullet(b, jd_keywords) for b in bullets[:15]]  # Limit to 15
