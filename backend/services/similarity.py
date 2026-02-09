"""TF-IDF and JobBERT-v2 similarity engine for resume-JD matching."""

import logging

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine

logger = logging.getLogger(__name__)

# Lazy-loaded JobBERT-v2 model (loaded on first use, ~425MB)
# TechWolf/JobBERT-v2: trained on millions of job postings, 1024-dim embeddings
_sbert_model = None


def _get_sbert_model():
    """Load JobBERT-v2 model lazily on first call."""
    global _sbert_model
    if _sbert_model is None:
        try:
            from sentence_transformers import SentenceTransformer

            _sbert_model = SentenceTransformer("TechWolf/JobBERT-v2")
            logger.info("JobBERT-v2 model loaded successfully")
        except Exception as e:
            logger.warning("Failed to load JobBERT-v2 model: %s", e)
    return _sbert_model


def tfidf_cosine_similarity(text_a: str, text_b: str) -> float:
    """Compute cosine similarity between two texts using TF-IDF vectors."""
    if not text_a.strip() or not text_b.strip():
        return 0.0

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=5000,
        sublinear_tf=True,
        ngram_range=(1, 2),
    )
    try:
        tfidf_matrix = vectorizer.fit_transform([text_a, text_b])
        score = sklearn_cosine(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return float(score)
    except ValueError:
        return 0.0


def sbert_cosine_similarity(text_a: str, text_b: str) -> float:
    """Compute cosine similarity using Sentence-BERT embeddings."""
    model = _get_sbert_model()
    if model is None:
        return -1.0  # Signal that SBERT is unavailable

    if not text_a.strip() or not text_b.strip():
        return 0.0

    try:
        embeddings = model.encode([text_a, text_b], convert_to_numpy=True)
        score = sklearn_cosine(
            embeddings[0:1], embeddings[1:2]
        )[0][0]
        return float(score)
    except Exception as e:
        logger.warning("SBERT encoding failed: %s", e)
        return -1.0


def hybrid_similarity(
    text_a: str,
    text_b: str,
    alpha: float = 0.4,
) -> tuple[float, float, float]:
    """Compute hybrid similarity score.

    Returns (hybrid_score, tfidf_score, semantic_score).
    alpha weights TF-IDF; (1-alpha) weights SBERT.
    If SBERT unavailable, falls back to TF-IDF only.
    """
    tfidf_score = tfidf_cosine_similarity(text_a, text_b)
    semantic_score = sbert_cosine_similarity(text_a, text_b)

    if semantic_score < 0:
        # SBERT unavailable, use TF-IDF only
        return tfidf_score, tfidf_score, 0.0

    hybrid = alpha * tfidf_score + (1 - alpha) * semantic_score
    return hybrid, tfidf_score, semantic_score


def section_similarities(
    resume_sections: dict[str, str],
    job_description: str,
) -> dict[str, float]:
    """Compute SBERT similarity per resume section against the full JD.

    Returns dict mapping section name -> similarity score (0.0-1.0).
    Falls back to TF-IDF if SBERT unavailable.
    """
    results: dict[str, float] = {}
    for section_name, section_text in resume_sections.items():
        if section_name == "header" or not section_text.strip():
            continue
        sem = sbert_cosine_similarity(section_text, job_description)
        if sem < 0:
            # SBERT unavailable, use TF-IDF
            sem = tfidf_cosine_similarity(section_text, job_description)
        results[section_name] = max(0.0, float(sem))
    return results


def extract_tfidf_keywords(text: str, top_n: int = 20) -> list[str]:
    """Extract top keywords from text using TF-IDF scores.

    Uses a small reference corpus of generic filler text to compute IDF,
    making distinctive terms in the input text stand out.
    """
    if not text.strip():
        return []

    # Reference corpus to provide IDF contrast
    reference = [
        "the candidate should have experience and skills in relevant areas",
        "looking for a professional with strong background and qualifications",
        "requirements include working with teams and delivering results",
    ]

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=3000,
        sublinear_tf=True,
        ngram_range=(1, 2),
    )

    try:
        tfidf_matrix = vectorizer.fit_transform([text] + reference)
        feature_names = vectorizer.get_feature_names_out()
        scores = tfidf_matrix[0].toarray().flatten()

        # Get top scoring terms
        top_indices = np.argsort(scores)[::-1][:top_n]
        keywords = [
            feature_names[i]
            for i in top_indices
            if scores[i] > 0 and len(feature_names[i]) > 1
        ]
        return keywords
    except ValueError:
        return []
