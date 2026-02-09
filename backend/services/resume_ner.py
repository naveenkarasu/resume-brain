"""Resume NER parser using yashpwr/resume-ner-bert-v2.

Extracts structured entities (name, email, skills, companies, etc.)
from resume text using a fine-tuned BERT NER model.
"""

import logging
import re

logger = logging.getLogger(__name__)

# Lazy-loaded NER pipeline
_resume_ner = None


def _get_resume_ner():
    """Load resume NER model lazily."""
    global _resume_ner
    if _resume_ner is None:
        try:
            from transformers import pipeline

            _resume_ner = pipeline(
                "token-classification",
                model="yashpwr/resume-ner-bert-v2",
                aggregation_strategy="simple",
            )
            logger.info("Resume NER model loaded successfully")
        except Exception as e:
            logger.warning("Failed to load resume NER model: %s", e)
    return _resume_ner


# Map model entity_group labels to our canonical keys.
# The model uses labels like "Name", "Email Address", "Skills", etc.
_LABEL_MAP: dict[str, str] = {
    "name": "name",
    "email address": "email",
    "phone number": "phone",
    "skills": "skills",
    "companies worked at": "companies",
    "designation": "designations",
    "college name": "colleges",
    "graduation year": "graduation_years",
    "location": "locations",
    "years of experience": "years_of_experience",
    "degree": "degrees",
}


def _empty_entities() -> dict:
    """Return empty entity structure."""
    return {
        "name": [],
        "email": [],
        "phone": [],
        "skills": [],
        "companies": [],
        "designations": [],
        "colleges": [],
        "graduation_years": [],
        "locations": [],
        "years_of_experience": [],
        "degrees": [],
    }


def _chunk_text(text: str, max_tokens: int = 450) -> list[str]:
    """Split text on paragraph boundaries to stay within BERT's 512 token limit.

    Uses a rough 1 token ~ 0.75 words heuristic. Splits on double newlines
    and accumulates paragraphs until the chunk approaches max_tokens.
    """
    if not text or not text.strip():
        return []

    paragraphs = re.split(r"\n\n+", text.strip())
    chunks: list[str] = []
    current_chunk: list[str] = []
    current_words = 0
    # Rough heuristic: 1 token ≈ 0.75 words, so max_tokens tokens ≈ max_tokens * 0.75 words
    max_words = int(max_tokens * 0.75)

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        para_words = len(para.split())
        if current_words + para_words > max_words and current_chunk:
            chunks.append("\n\n".join(current_chunk))
            current_chunk = [para]
            current_words = para_words
        else:
            current_chunk.append(para)
            current_words += para_words

    if current_chunk:
        chunks.append("\n\n".join(current_chunk))

    return chunks


def extract_resume_entities(text: str) -> dict:
    """Extract structured entities from resume text using NER.

    Returns a dict with keys: name, email, phone, skills, companies,
    designations, colleges, graduation_years, locations,
    years_of_experience, degrees.

    Each value is a deduplicated list of strings. Entities with
    confidence < 0.5 are filtered out.
    """
    ner = _get_resume_ner()
    if ner is None:
        return _empty_entities()

    chunks = _chunk_text(text)
    if not chunks:
        return _empty_entities()

    entities = _empty_entities()

    try:
        for chunk in chunks:
            results = ner(chunk)
            for r in results:
                if r["score"] < 0.5:
                    continue
                label = r["entity_group"].lower()
                key = _LABEL_MAP.get(label)
                if key is None:
                    continue
                word = r["word"].strip()
                # Clean up subword artifacts
                word = re.sub(r"\s*##\s*", "", word).strip()
                if word and len(word) > 1:
                    entities[key].append(word)
    except Exception as e:
        logger.warning("Resume NER extraction failed: %s", e)
        return _empty_entities()

    # Deduplicate within each category (case-insensitive)
    for key in entities:
        seen: set[str] = set()
        deduped: list[str] = []
        for val in entities[key]:
            lower = val.lower()
            if lower not in seen:
                seen.add(lower)
                deduped.append(val)
        entities[key] = deduped

    return entities
