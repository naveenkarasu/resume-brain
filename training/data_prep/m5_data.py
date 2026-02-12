#!/usr/bin/env python3
"""Model 5 Data Preparation: Judge Model Training Data.

Parses labeled resume-JD pairs from three sources and builds a dataset
of 13-dimensional feature vectors with match-quality score labels for
training the Judge model (LightGBM / MLP).

Sources:
    - netsol/resume-score-details (1K pairs, multi-dim scores + justifications)
    - resume-ats-score-v1-en (6.4K pairs, continuous ATS scores 19.2-90.1)
    - resume-job-description-fit (8K pairs, 3-class fit labels)

Feature vector (13 dimensions):
    skill_coverage, skill_exact_matches, skill_partial_matches,
    missing_required_count, experience_score, education_score,
    domain_relevance, career_trajectory, certification_match,
    soft_skills_score, keyword_density, section_completeness,
    overall_text_similarity

Output: HuggingFace Dataset with 13 feature columns + ``score`` label.
"""

from __future__ import annotations

import json
import logging
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict, Features, Value

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "model5_judge"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "model5_judge" / "unified"

FEATURE_NAMES: list[str] = [
    "skill_coverage",
    "skill_exact_matches",
    "skill_partial_matches",
    "missing_required_count",
    "experience_score",
    "education_score",
    "domain_relevance",
    "career_trajectory",
    "certification_match",
    "soft_skills_score",
    "keyword_density",
    "section_completeness",
    "overall_text_similarity",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _simple_cosine(a: str, b: str) -> float:
    """Word-level cosine similarity between two strings."""
    if not a or not b:
        return 0.0
    ta = Counter(a.lower().split())
    tb = Counter(b.lower().split())
    common = set(ta.keys()) & set(tb.keys())
    if not common:
        return 0.0
    dot = sum(ta[t] * tb[t] for t in common)
    na = math.sqrt(sum(v ** 2 for v in ta.values()))
    nb = math.sqrt(sum(v ** 2 for v in tb.values()))
    return dot / (na * nb) if na > 0 and nb > 0 else 0.0


def _keyword_overlap(text: str, keywords: list[str]) -> tuple[int, int, float]:
    """Count exact, partial keyword matches and coverage ratio."""
    if not text or not keywords:
        return 0, 0, 0.0
    text_lower = text.lower()
    exact = sum(1 for kw in keywords if kw.lower() in text_lower)
    # Partial: check if any word of the keyword appears
    partial = 0
    for kw in keywords:
        kw_words = kw.lower().split()
        if any(w in text_lower for w in kw_words) and kw.lower() not in text_lower:
            partial += 1
    coverage = exact / len(keywords) if keywords else 0.0
    return exact, partial, coverage


def _extract_features_from_texts(
    resume_text: str,
    jd_text: str,
) -> dict[str, float]:
    """Extract 13 features from resume and JD text pair.

    This is a text-based heuristic extraction used when structured model
    outputs are not available (i.e., for raw labeled pairs).
    """
    features: dict[str, float] = {}

    # Extract pseudo-keywords from JD (words > 4 chars, excluding stopwords)
    stopwords = {"with", "that", "this", "from", "have", "will", "your", "they",
                 "been", "were", "being", "about", "which", "their", "would",
                 "could", "should", "other", "these", "those", "than", "into"}
    jd_words = [w.strip(".,;:()[]{}\"'") for w in jd_text.lower().split()
                if len(w) > 4 and w.lower().strip(".,;:()[]{}\"'") not in stopwords]
    jd_keywords = list(set(jd_words))[:50]

    exact, partial, coverage = _keyword_overlap(resume_text, jd_keywords)

    features["skill_coverage"] = coverage
    features["skill_exact_matches"] = min(float(exact) / max(len(jd_keywords), 1), 1.0)
    features["skill_partial_matches"] = min(float(partial) / max(len(jd_keywords), 1), 1.0)
    features["missing_required_count"] = float(max(len(jd_keywords) - exact - partial, 0))

    # Experience indicators
    import re
    exp_pattern = re.compile(r"(\d+)\+?\s*years?", re.IGNORECASE)
    resume_years = [int(m.group(1)) for m in exp_pattern.finditer(resume_text)]
    jd_years = [int(m.group(1)) for m in exp_pattern.finditer(jd_text)]
    resume_max_years = max(resume_years) if resume_years else 0
    jd_min_years = min(jd_years) if jd_years else 0
    features["experience_score"] = min(float(resume_max_years) / max(float(jd_min_years), 1.0), 2.0)

    # Education indicators
    edu_keywords = ["phd", "doctorate", "master", "mba", "bachelor", "degree", "bs", "ms", "ba", "ma"]
    resume_edu = sum(1 for kw in edu_keywords if kw in resume_text.lower())
    jd_edu = sum(1 for kw in edu_keywords if kw in jd_text.lower())
    features["education_score"] = min(float(resume_edu) / max(float(jd_edu), 1.0), 2.0)

    # Domain relevance via text similarity
    features["domain_relevance"] = _simple_cosine(resume_text[:500], jd_text[:500])

    # Career trajectory: number of distinct roles mentioned
    role_indicators = ["manager", "engineer", "developer", "analyst", "director",
                       "lead", "senior", "specialist", "coordinator", "consultant"]
    features["career_trajectory"] = min(
        sum(1 for r in role_indicators if r in resume_text.lower()) / 5.0, 1.0
    )

    # Certification match
    cert_keywords = ["certified", "certification", "certificate", "aws", "pmp",
                     "scrum", "cisco", "google", "microsoft", "oracle"]
    resume_certs = sum(1 for kw in cert_keywords if kw in resume_text.lower())
    jd_certs = sum(1 for kw in cert_keywords if kw in jd_text.lower())
    features["certification_match"] = min(float(resume_certs) / max(float(jd_certs), 1.0), 2.0)

    # Soft skills score
    soft_keywords = ["communication", "leadership", "teamwork", "problem-solving",
                     "analytical", "creative", "adaptable", "collaborative"]
    features["soft_skills_score"] = min(
        sum(1 for kw in soft_keywords if kw in resume_text.lower()) / 4.0, 1.0
    )

    # Keyword density (ratio of JD keywords found in resume)
    features["keyword_density"] = coverage

    # Section completeness (does resume have standard sections?)
    sections = ["education", "experience", "skills", "summary", "projects",
                "certifications", "work history", "objective"]
    features["section_completeness"] = sum(
        1 for s in sections if s in resume_text.lower()
    ) / len(sections)

    # Overall text similarity
    features["overall_text_similarity"] = _simple_cosine(resume_text, jd_text)

    return features


# ---------------------------------------------------------------------------
# Dataset Loaders
# ---------------------------------------------------------------------------


def load_netsol() -> list[dict[str, Any]]:
    """Load netsol/resume-score-details (1K pairs with multi-dim scores).

    Each JSON file contains: input (resume, jd) and output (scores, justification).
    """
    netsol_dir = DATA_DIR / "netsol_score_details"
    if not netsol_dir.exists():
        logger.warning("Netsol directory not found at %s -- skipping.", netsol_dir)
        return []

    logger.info("Loading netsol from %s", netsol_dir)
    records: list[dict[str, Any]] = []
    json_files = sorted(netsol_dir.glob("*.json"))

    if not json_files:
        logger.warning("No JSON files in %s.", netsol_dir)
        return []

    for jf in json_files:
        # Skip files that are clearly invalid
        if "invalid" in jf.name or "gibberish" in jf.name:
            continue
        try:
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError):
            logger.debug("Netsol: skipping malformed file %s.", jf.name)
            continue

        inp = data.get("input", {})
        out = data.get("output", {})

        resume_text = inp.get("resume", inp.get("resume_text", ""))
        jd_text = inp.get("job_description", inp.get("jd_text", ""))

        if not resume_text or not jd_text:
            continue

        # Extract score from output
        scores = out.get("scores", {})
        agg = scores.get("aggregated_scores", {})
        macro = float(agg.get("macro_scores", 0))
        micro = float(agg.get("micro_scores", 0))
        score = (macro + micro) / 2.0 if macro or micro else 0.0

        # Normalize to 0-1 range (scores appear to be 0-10)
        score = min(score / 10.0, 1.0)

        features = _extract_features_from_texts(str(resume_text), str(jd_text))
        features["score"] = score
        features["source"] = "netsol"
        records.append(features)

    logger.info("Netsol: loaded %d valid pairs.", len(records))
    return records


def load_ats_score() -> list[dict[str, Any]]:
    """Load resume-ats-score-v1-en (6.4K pairs with continuous ATS scores).

    Parquet with resume text, JD text, and ATS score (19.2 - 90.1 range).
    """
    records: list[dict[str, Any]] = []
    ats_dir = DATA_DIR / "ats_score"

    if not ats_dir.exists():
        logger.warning("ATS score directory not found at %s -- skipping.", ats_dir)
        return []

    logger.info("Loading ATS score from %s", ats_dir)
    parquet_files = sorted(ats_dir.glob("*.parquet"))

    if not parquet_files:
        logger.warning("No parquet files in %s.", ats_dir)
        return []

    for pf in parquet_files:
        try:
            df = pd.read_parquet(pf)
        except Exception:
            logger.warning("Failed to read %s -- skipping.", pf)
            continue

        # Identify columns
        resume_col = None
        jd_col = None
        score_col = None

        for col in df.columns:
            cl = col.lower()
            if "resume" in cl and "score" not in cl:
                resume_col = col
            elif "job" in cl or "jd" in cl or "description" in cl:
                jd_col = col
            elif "score" in cl or "ats" in cl:
                score_col = col

        if resume_col is None or score_col is None:
            # Try positional fallback
            cols = df.columns.tolist()
            if len(cols) >= 3:
                resume_col = cols[0]
                jd_col = cols[1]
                score_col = cols[2]
            else:
                logger.warning("ATS: cannot identify columns in %s. Got: %s", pf.name, df.columns.tolist())
                continue

        for _, row in df.iterrows():
            resume_text = str(row.get(resume_col, ""))
            jd_text = str(row.get(jd_col, "")) if jd_col else ""
            raw_score = row.get(score_col, 0)

            if not resume_text or len(resume_text) < 20:
                continue

            try:
                score = float(raw_score)
            except (ValueError, TypeError):
                continue

            # Normalize score to 0-1 (original range ~19-90)
            score = max(0.0, min((score - 10) / 90.0, 1.0))

            features = _extract_features_from_texts(resume_text, jd_text)
            features["score"] = score
            features["source"] = "ats_score"
            records.append(features)

    logger.info("ATS score: loaded %d pairs.", len(records))
    return records


def load_resume_jd_fit() -> list[dict[str, Any]]:
    """Load resume-job-description-fit (8K pairs with 3-class labels).

    Labels: Fit, Partial Fit, No Fit -> mapped to scores 1.0, 0.5, 0.0.
    """
    records: list[dict[str, Any]] = []
    fit_dir = DATA_DIR / "resume_jd_fit"

    if not fit_dir.exists():
        logger.warning("Resume-JD-fit directory not found at %s -- skipping.", fit_dir)
        return []

    logger.info("Loading resume-JD-fit from %s", fit_dir)
    parquet_files = sorted(fit_dir.glob("*.parquet"))
    csv_files = sorted(fit_dir.glob("*.csv"))
    data_files = parquet_files + csv_files

    if not data_files:
        logger.warning("No data files in %s.", fit_dir)
        return []

    fit_label_map = {
        "fit": 1.0,
        "good fit": 1.0,
        "strong fit": 1.0,
        "partial fit": 0.5,
        "partial": 0.5,
        "no fit": 0.0,
        "not fit": 0.0,
        "poor fit": 0.0,
    }

    for df_path in data_files:
        try:
            if df_path.suffix == ".parquet":
                df = pd.read_parquet(df_path)
            else:
                df = pd.read_csv(df_path)
        except Exception:
            logger.warning("Failed to read %s -- skipping.", df_path)
            continue

        # Identify columns
        resume_col = None
        jd_col = None
        label_col = None

        for col in df.columns:
            cl = col.lower()
            if "resume" in cl and "score" not in cl and "fit" not in cl:
                resume_col = col
            elif "job" in cl or "jd" in cl or "description" in cl:
                jd_col = col
            elif "fit" in cl or "label" in cl or "class" in cl or "match" in cl:
                label_col = col

        if resume_col is None or label_col is None:
            cols = df.columns.tolist()
            logger.warning("Resume-JD-fit: cannot identify columns in %s. Got: %s", df_path.name, cols)
            continue

        for _, row in df.iterrows():
            resume_text = str(row.get(resume_col, ""))
            jd_text = str(row.get(jd_col, "")) if jd_col else ""
            raw_label = str(row.get(label_col, "")).strip().lower()

            if not resume_text or len(resume_text) < 20:
                continue

            # Map label to score
            score = fit_label_map.get(raw_label)
            if score is None:
                # Try numeric
                try:
                    score = float(raw_label)
                    if score > 1:
                        score = score / 100.0  # percentage to fraction
                except ValueError:
                    continue

            features = _extract_features_from_texts(resume_text, jd_text)
            features["score"] = max(0.0, min(float(score), 1.0))
            features["source"] = "resume_jd_fit"
            records.append(features)

    logger.info("Resume-JD-fit: loaded %d pairs.", len(records))
    return records


# ---------------------------------------------------------------------------
# Score Normalization
# ---------------------------------------------------------------------------


def normalize_scores(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize scores to [0, 1] range across all sources.

    Per-source z-score normalization followed by min-max scaling to [0, 1].
    """
    if not records:
        return records

    # Group by source
    by_source: dict[str, list[int]] = {}
    for i, rec in enumerate(records):
        src = rec.get("source", "unknown")
        by_source.setdefault(src, []).append(i)

    for src, indices in by_source.items():
        scores = np.array([records[i]["score"] for i in indices], dtype=np.float64)
        mean = np.mean(scores)
        std = np.std(scores)

        if std > 0:
            z_scores = (scores - mean) / std
            # Min-max scale to [0, 1]
            z_min, z_max = z_scores.min(), z_scores.max()
            if z_max > z_min:
                normalized = (z_scores - z_min) / (z_max - z_min)
            else:
                normalized = np.full_like(z_scores, 0.5)
        else:
            normalized = np.full_like(scores, 0.5)

        for idx, norm_score in zip(indices, normalized):
            records[idx]["score"] = float(norm_score)

        logger.info(
            "Normalized '%s': %d records, raw mean=%.3f, raw std=%.3f -> [0, 1].",
            src, len(indices), mean, std,
        )

    return records


# ---------------------------------------------------------------------------
# Dataset Construction
# ---------------------------------------------------------------------------


def build_dataset(records: list[dict[str, Any]], seed: int = 42) -> DatasetDict:
    """Convert records into train/val/test DatasetDict."""
    rng = random.Random(seed)
    rng.shuffle(records)

    n = len(records)
    n_train = int(n * 0.80)
    n_val = int(n * 0.10)

    splits = {
        "train": records[:n_train],
        "validation": records[n_train : n_train + n_val],
        "test": records[n_train + n_val :],
    }

    feature_dict: dict[str, Any] = {name: Value("float64") for name in FEATURE_NAMES}
    feature_dict["score"] = Value("float64")
    feature_dict["source"] = Value("string")
    features = Features(feature_dict)

    dd = DatasetDict()
    for name, data in splits.items():
        col_data: dict[str, list] = {fname: [] for fname in FEATURE_NAMES}
        col_data["score"] = []
        col_data["source"] = []
        for row in data:
            for fname in FEATURE_NAMES:
                col_data[fname].append(float(row.get(fname, 0.0)))
            col_data["score"].append(float(row.get("score", 0.0)))
            col_data["source"].append(str(row.get("source", "unknown")))

        dd[name] = Dataset.from_dict(col_data, features=features)
        logger.info("Split '%s': %d examples.", name, len(data))

    return dd


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run full M5 data pipeline: load, normalize, build dataset, save."""
    logger.info("=" * 60)
    logger.info("M5 Data Preparation: Judge Model Training Data")
    logger.info("DATA_DIR: %s", DATA_DIR)
    logger.info("OUTPUT_DIR: %s", OUTPUT_DIR)
    logger.info("Features (%d): %s", len(FEATURE_NAMES), FEATURE_NAMES)
    logger.info("=" * 60)

    all_records: list[dict[str, Any]] = []

    for name, loader in [
        ("netsol", load_netsol),
        ("ats_score", load_ats_score),
        ("resume_jd_fit", load_resume_jd_fit),
    ]:
        try:
            records = loader()
            all_records.extend(records)
            logger.info("[%s] contributed %d records (total: %d).", name, len(records), len(all_records))
        except Exception:
            logger.exception("Loader '%s' raised an unexpected error -- skipping.", name)

    if not all_records:
        logger.error("No records produced -- aborting.")
        return

    all_records = normalize_scores(all_records)
    logger.info("After normalization: %d records.", len(all_records))

    dataset = build_dataset(all_records)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(OUTPUT_DIR))
    logger.info("Dataset saved to %s", OUTPUT_DIR)

    # Save feature metadata
    meta_path = OUTPUT_DIR / "feature_metadata.json"
    with open(meta_path, "w") as f:
        json.dump({
            "feature_names": FEATURE_NAMES,
            "num_features": len(FEATURE_NAMES),
            "score_range": [0.0, 1.0],
            "sources": list(set(r.get("source", "unknown") for r in all_records)),
        }, f, indent=2)
    logger.info("Feature metadata saved to %s", meta_path)

    # Summary statistics
    for split_name in dataset:
        ds = dataset[split_name]
        scores = [row["score"] for row in ds]
        source_counts = Counter(row["source"] for row in ds)
        logger.info(
            "  %s: %d examples, score mean=%.3f, std=%.3f, sources=%s",
            split_name,
            len(ds),
            np.mean(scores),
            np.std(scores),
            dict(source_counts),
        )


if __name__ == "__main__":
    main()
