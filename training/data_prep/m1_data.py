#!/usr/bin/env python3
"""Model 1 Data Preparation: JD NER Training Data.

Unifies 8 job-description datasets (SkillSpan, Green, JD2Skills, Google Job
Skills, Djinni, Sayfullina, Skill-Extraction-Benchmark, JOBS-Information-
Extraction) into BIO-tagged sequences for token-level NER.

Entity types (9):
    SKILL, SOFT_SKILL, QUALIFICATION, EXPERIENCE_REQ, EDUCATION_REQ,
    CERTIFICATION, RESPONSIBILITY, TOOL, DOMAIN

Output: HuggingFace Dataset with columns ``tokens`` (List[str]) and
``ner_tags`` (List[int]) using the BIO tagging scheme.
"""

from __future__ import annotations

import json
import logging
import re
import tarfile
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import Dataset, DatasetDict, Features, Sequence, Value, load_from_disk

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "model1_jd_extractor"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "model1_jd_extractor" / "unified"

ENTITY_TYPES: list[str] = [
    "SKILL",
    "SOFT_SKILL",
    "QUALIFICATION",
    "EXPERIENCE_REQ",
    "EDUCATION_REQ",
    "CERTIFICATION",
    "RESPONSIBILITY",
    "TOOL",
    "DOMAIN",
]

# Build label list: O, B-SKILL, I-SKILL, B-SOFT_SKILL, ...
LABELS: list[str] = ["O"]
for etype in ENTITY_TYPES:
    LABELS.append(f"B-{etype}")
    LABELS.append(f"I-{etype}")

LABEL2ID: dict[str, int] = {label: idx for idx, label in enumerate(LABELS)}
ID2LABEL: dict[int, str] = {idx: label for idx, label in enumerate(LABELS)}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bio_tags_from_spans(
    tokens: list[str],
    char_offsets: list[tuple[int, int]],
    spans: list[dict[str, Any]],
) -> list[str]:
    """Convert character-level spans to BIO tags aligned to token offsets.

    Parameters
    ----------
    tokens:
        Whitespace-split tokens.
    char_offsets:
        List of (start, end) character positions for each token.
    spans:
        List of dicts with keys ``start``, ``end``, ``label``.

    Returns
    -------
    list of BIO tag strings, one per token.
    """
    tags = ["O"] * len(tokens)
    for span in spans:
        s_start, s_end, label = span["start"], span["end"], span["label"]
        first = True
        for idx, (t_start, t_end) in enumerate(char_offsets):
            if t_end <= s_start or t_start >= s_end:
                continue
            prefix = "B" if first else "I"
            tag = f"{prefix}-{label}"
            if tag in LABEL2ID:
                tags[idx] = tag
                first = False
    return tags


def _tokenize_with_offsets(text: str) -> tuple[list[str], list[tuple[int, int]]]:
    """Whitespace-tokenize text and return tokens with char offsets."""
    tokens: list[str] = []
    offsets: list[tuple[int, int]] = []
    for m in re.finditer(r"\S+", text):
        tokens.append(m.group())
        offsets.append((m.start(), m.end()))
    return tokens, offsets


def _label_to_id(tags: list[str]) -> list[int]:
    """Convert string BIO tags to integer ids."""
    return [LABEL2ID.get(t, 0) for t in tags]


# ---------------------------------------------------------------------------
# Dataset Loaders
# ---------------------------------------------------------------------------


def load_skillspan() -> list[dict[str, Any]]:
    """Load SkillSpan (jjzha/skillspan) from local HF arrow files.

    SkillSpan has columns: tokens, tags_skill, tags_knowledge, source.
    We map tags_skill B-Skill/I-Skill -> B-SKILL/I-SKILL and
    tags_knowledge B-Knowledge/I-Knowledge -> B-QUALIFICATION/I-QUALIFICATION.
    """
    path = DATA_DIR / "skillspan"
    if not path.exists():
        logger.warning("SkillSpan directory not found at %s -- skipping.", path)
        return []

    logger.info("Loading SkillSpan from %s", path)
    try:
        ds = load_from_disk(str(path))
    except Exception:
        logger.exception("Failed to load SkillSpan dataset.")
        return []

    tag_map = {
        "B-Skill": "B-SKILL",
        "I-Skill": "I-SKILL",
        "B-Knowledge": "B-QUALIFICATION",
        "I-Knowledge": "I-QUALIFICATION",
        "O": "O",
    }

    records: list[dict[str, Any]] = []
    for split in ds:
        for row in ds[split]:
            tokens = row["tokens"]
            # Merge skill and knowledge tags -- skill takes priority
            merged: list[str] = []
            for sk, kn in zip(row["tags_skill"], row["tags_knowledge"]):
                if sk != "O":
                    merged.append(tag_map.get(sk, "O"))
                elif kn != "O":
                    merged.append(tag_map.get(kn, "O"))
                else:
                    merged.append("O")
            records.append({
                "tokens": tokens,
                "ner_tags": _label_to_id(merged),
                "source": "skillspan",
            })

    logger.info("SkillSpan: loaded %d sequences.", len(records))
    return records


def load_green() -> list[dict[str, Any]]:
    """Load Green Skill Extraction Benchmark.

    Format: CSV with columns sentence_id, word, pos, tag.
    Tags use BIO with types like Skill, Qualification, etc.
    """
    csv_path = DATA_DIR / "green_skill_extraction" / "preprocessed_data" / "df_answers.csv"
    if not csv_path.exists():
        logger.warning("Green dataset not found at %s -- skipping.", csv_path)
        return []

    logger.info("Loading Green from %s", csv_path)
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        logger.exception("Failed to read Green CSV.")
        return []

    green_tag_map = {
        "B-Skill": "B-SKILL",
        "I-Skill": "I-SKILL",
        "B-Qualification": "B-QUALIFICATION",
        "I-Qualification": "I-QUALIFICATION",
        "B-Experience": "B-EXPERIENCE_REQ",
        "I-Experience": "I-EXPERIENCE_REQ",
        "B-Occupation": "B-RESPONSIBILITY",
        "I-Occupation": "I-RESPONSIBILITY",
        "B-Domain": "B-DOMAIN",
        "I-Domain": "I-DOMAIN",
        "O": "O",
    }

    records: list[dict[str, Any]] = []
    for sid, group in df.groupby("sentence_id"):
        tokens = group["word"].tolist()
        tags = [green_tag_map.get(str(t).strip(), "O") for t in group["tag"]]
        records.append({
            "tokens": tokens,
            "ner_tags": _label_to_id(tags),
            "source": "green",
        })

    logger.info("Green: loaded %d sequences.", len(records))
    return records


def load_jd2skills() -> list[dict[str, Any]]:
    """Load JD2Skills structured postings.

    The dataset is a tar.gz of JSON records with structured fields. We extract
    description text and create weakly-supervised BIO tags from the structured
    fields (skills list, experience, education).
    """
    tar_path = DATA_DIR / "jd2skills" / "data" / "mycareersfuture.tar.gz"
    if not tar_path.exists():
        logger.warning("JD2Skills tar not found at %s -- skipping.", tar_path)
        return []

    logger.info("Loading JD2Skills from %s", tar_path)
    records: list[dict[str, Any]] = []
    try:
        with tarfile.open(str(tar_path), "r:gz") as tf:
            for member in tf.getmembers():
                if not member.name.endswith(".json"):
                    continue
                fobj = tf.extractfile(member)
                if fobj is None:
                    continue
                try:
                    data = json.loads(fobj.read().decode("utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue

                description = data.get("description", "")
                if not description or len(description) < 20:
                    continue

                tokens, offsets = _tokenize_with_offsets(description)
                if not tokens:
                    continue

                # Build spans from structured fields
                spans: list[dict[str, Any]] = []
                for skill in data.get("skills", []):
                    if isinstance(skill, str) and skill:
                        for m in re.finditer(re.escape(skill), description, re.IGNORECASE):
                            spans.append({"start": m.start(), "end": m.end(), "label": "SKILL"})

                min_exp = data.get("minimumYearsExperience")
                if min_exp is not None:
                    pattern = rf"\b{re.escape(str(min_exp))}\s*(?:\+\s*)?years?"
                    for m in re.finditer(pattern, description, re.IGNORECASE):
                        spans.append({"start": m.start(), "end": m.end(), "label": "EXPERIENCE_REQ"})

                tags = _bio_tags_from_spans(tokens, offsets, spans)
                # Only keep if at least one entity was found
                if any(t != "O" for t in tags):
                    records.append({
                        "tokens": tokens,
                        "ner_tags": _label_to_id(tags),
                        "source": "jd2skills",
                    })
    except Exception:
        logger.exception("Failed to process JD2Skills tar.")

    logger.info("JD2Skills: loaded %d sequences.", len(records))
    return records


def load_google_job_skills() -> list[dict[str, Any]]:
    """Load Google Job Skills dataset.

    CSV with columns: Company, Title, Category, Location, Responsibilities,
    Minimum Qualifications, Preferred Qualifications.

    We tokenize each section and assign entity tags based on column type.
    """
    csv_path = DATA_DIR / "google_job_skills" / "job_skills.csv"
    if not csv_path.exists():
        logger.warning("Google Job Skills not found at %s -- skipping.", csv_path)
        return []

    logger.info("Loading Google Job Skills from %s", csv_path)
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        logger.exception("Failed to read Google Job Skills CSV.")
        return []

    section_label_map = {
        "Responsibilities": "RESPONSIBILITY",
        "Minimum Qualifications": "QUALIFICATION",
        "Preferred Qualifications": "SKILL",
    }

    records: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        for col, label in section_label_map.items():
            text = row.get(col)
            if not isinstance(text, str) or len(text.strip()) < 10:
                continue
            # Split into bullet points / lines
            lines = [line.strip() for line in text.split("\n") if line.strip()]
            for line in lines:
                tokens, offsets = _tokenize_with_offsets(line)
                if len(tokens) < 3:
                    continue
                # Tag entire line as the section entity type
                tags = [f"B-{label}"] + [f"I-{label}"] * (len(tokens) - 1)
                records.append({
                    "tokens": tokens,
                    "ner_tags": _label_to_id(tags),
                    "source": "google_job_skills",
                })

    logger.info("Google Job Skills: loaded %d sequences.", len(records))
    return records


def load_djinni() -> list[dict[str, Any]]:
    """Load Djinni JDs (parquet with experience_years and primary_keyword).

    We extract keyword mentions from JD text and tag them as SKILL, plus
    create EXPERIENCE_REQ tags from experience-related patterns.
    """
    parquet_path = DATA_DIR / "djinni_jds" / "train.parquet"
    if not parquet_path.exists():
        logger.warning("Djinni JDs parquet not found at %s -- skipping.", parquet_path)
        return []

    logger.info("Loading Djinni JDs from %s", parquet_path)
    try:
        df = pd.read_parquet(parquet_path)
    except Exception:
        logger.exception("Failed to read Djinni parquet.")
        return []

    # Limit to a manageable sample for training
    if len(df) > 20000:
        df = df.sample(n=20000, random_state=42)
        logger.info("Djinni: sampled 20,000 of %d total rows.", len(df))

    records: list[dict[str, Any]] = []
    text_col = None
    for candidate in ["description", "text", "job_description", "content"]:
        if candidate in df.columns:
            text_col = candidate
            break

    if text_col is None:
        logger.warning("Djinni: no text column found. Columns: %s", df.columns.tolist())
        return []

    exp_pattern = re.compile(r"\b(\d+)\+?\s*years?\b", re.IGNORECASE)

    for _, row in df.iterrows():
        text = row.get(text_col, "")
        if not isinstance(text, str) or len(text) < 30:
            continue

        # Truncate very long descriptions for sentence-level NER
        if len(text) > 1500:
            text = text[:1500]

        tokens, offsets = _tokenize_with_offsets(text)
        if len(tokens) < 5:
            continue

        spans: list[dict[str, Any]] = []

        # Tag experience mentions
        for m in exp_pattern.finditer(text):
            spans.append({"start": m.start(), "end": m.end(), "label": "EXPERIENCE_REQ"})

        # Tag primary keyword if present in text
        keyword = row.get("primary_keyword", "")
        if isinstance(keyword, str) and keyword:
            for m in re.finditer(re.escape(keyword), text, re.IGNORECASE):
                spans.append({"start": m.start(), "end": m.end(), "label": "SKILL"})

        tags = _bio_tags_from_spans(tokens, offsets, spans)
        if any(t != "O" for t in tags):
            records.append({
                "tokens": tokens,
                "ner_tags": _label_to_id(tags),
                "source": "djinni",
            })

    logger.info("Djinni: loaded %d sequences.", len(records))
    return records


def load_sayfullina() -> list[dict[str, Any]]:
    """Load Sayfullina Soft Skills dataset.

    Parquet with BIO tags for soft skill spans. We map their tags to
    B-SOFT_SKILL / I-SOFT_SKILL.
    """
    records: list[dict[str, Any]] = []
    sayfullina_dir = DATA_DIR / "sayfullina_soft_skills"
    if not sayfullina_dir.exists():
        logger.warning("Sayfullina directory not found at %s -- skipping.", sayfullina_dir)
        return []

    logger.info("Loading Sayfullina Soft Skills from %s", sayfullina_dir)

    parquet_files = list(sayfullina_dir.glob("*.parquet"))
    if not parquet_files:
        logger.warning("No parquet files found in %s -- skipping.", sayfullina_dir)
        return []

    tag_map = {
        "B-Skill": "B-SOFT_SKILL",
        "I-Skill": "I-SOFT_SKILL",
        "B-SoftSkill": "B-SOFT_SKILL",
        "I-SoftSkill": "I-SOFT_SKILL",
        "O": "O",
    }

    for pf in parquet_files:
        try:
            df = pd.read_parquet(pf)
        except Exception:
            logger.warning("Failed to read %s -- skipping.", pf)
            continue

        # Expect columns like tokens and ner_tags / tags
        tokens_col = None
        tags_col = None
        for col in df.columns:
            cl = col.lower()
            if "token" in cl:
                tokens_col = col
            elif "tag" in cl or "ner" in cl or "label" in cl:
                tags_col = col

        if tokens_col is None or tags_col is None:
            # Try sentence-level format
            for _, row in df.iterrows():
                text = ""
                for col in df.columns:
                    val = row[col]
                    if isinstance(val, str) and len(val) > 20:
                        text = val
                        break
                if not text:
                    continue
                tokens, _ = _tokenize_with_offsets(text)
                # Default all to SOFT_SKILL since this is a soft-skills dataset
                tags = ["B-SOFT_SKILL"] + ["I-SOFT_SKILL"] * (len(tokens) - 1) if tokens else []
                records.append({
                    "tokens": tokens,
                    "ner_tags": _label_to_id(tags),
                    "source": "sayfullina",
                })
            continue

        for _, row in df.iterrows():
            tokens = row[tokens_col]
            raw_tags = row[tags_col]
            if not isinstance(tokens, list) or not isinstance(raw_tags, list):
                continue
            mapped_tags = [tag_map.get(str(t).strip(), "O") for t in raw_tags]
            records.append({
                "tokens": tokens,
                "ner_tags": _label_to_id(mapped_tags),
                "source": "sayfullina",
            })

    logger.info("Sayfullina: loaded %d sequences.", len(records))
    return records


# ---------------------------------------------------------------------------
# Unification and Splitting
# ---------------------------------------------------------------------------


def unify_all() -> list[dict[str, Any]]:
    """Run all loaders and merge into a single list of records."""
    loaders = [
        ("SkillSpan", load_skillspan),
        ("Green", load_green),
        ("JD2Skills", load_jd2skills),
        ("Google Job Skills", load_google_job_skills),
        ("Djinni", load_djinni),
        ("Sayfullina", load_sayfullina),
    ]

    all_records: list[dict[str, Any]] = []
    for name, loader in loaders:
        try:
            records = loader()
            all_records.extend(records)
            logger.info("[%s] contributed %d records (total so far: %d).", name, len(records), len(all_records))
        except Exception:
            logger.exception("Loader '%s' raised an unexpected error -- skipping.", name)

    logger.info("Unified dataset: %d total sequences.", len(all_records))
    return all_records


def split_dataset(
    records: list[dict[str, Any]],
    train_ratio: float = 0.85,
    val_ratio: float = 0.10,
    seed: int = 42,
) -> DatasetDict:
    """Split records into train / validation / test HuggingFace DatasetDict.

    Parameters
    ----------
    records : list
        Each record has keys ``tokens``, ``ner_tags``, ``source``.
    train_ratio, val_ratio : float
        Fractions for train and validation. Test gets the remainder.
    seed : int
        Random seed for reproducibility.
    """
    import random

    rng = random.Random(seed)
    rng.shuffle(records)

    n = len(records)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    splits = {
        "train": records[:n_train],
        "validation": records[n_train : n_train + n_val],
        "test": records[n_train + n_val :],
    }

    features = Features({
        "tokens": Sequence(Value("string")),
        "ner_tags": Sequence(Value("int32")),
        "source": Value("string"),
    })

    dd = DatasetDict()
    for split_name, split_records in splits.items():
        dd[split_name] = Dataset.from_dict(
            {
                "tokens": [r["tokens"] for r in split_records],
                "ner_tags": [r["ner_tags"] for r in split_records],
                "source": [r["source"] for r in split_records],
            },
            features=features,
        )
        logger.info("Split '%s': %d examples.", split_name, len(split_records))

    return dd


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run full M1 data pipeline: load, unify, split, and save."""
    logger.info("=" * 60)
    logger.info("M1 Data Preparation: JD NER")
    logger.info("DATA_DIR: %s", DATA_DIR)
    logger.info("OUTPUT_DIR: %s", OUTPUT_DIR)
    logger.info("Entity types (%d): %s", len(ENTITY_TYPES), ENTITY_TYPES)
    logger.info("Label count: %d (O + %d BIO tags)", len(LABELS), len(LABELS) - 1)
    logger.info("=" * 60)

    records = unify_all()
    if not records:
        logger.error("No records produced -- aborting.")
        return

    dataset = split_dataset(records)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(OUTPUT_DIR))
    logger.info("Dataset saved to %s", OUTPUT_DIR)

    # Save label mapping as JSON for downstream training scripts
    label_map_path = OUTPUT_DIR / "label_map.json"
    with open(label_map_path, "w") as f:
        json.dump({"labels": LABELS, "label2id": LABEL2ID, "id2label": {str(k): v for k, v in ID2LABEL.items()}}, f, indent=2)
    logger.info("Label map saved to %s", label_map_path)

    # Summary statistics
    for split_name in dataset:
        ds = dataset[split_name]
        non_o = sum(1 for row in ds for tag in row["ner_tags"] if tag != 0)
        total_tokens = sum(len(row["tokens"]) for row in ds)
        logger.info(
            "  %s: %d seqs, %d tokens, %d entity tokens (%.1f%%)",
            split_name,
            len(ds),
            total_tokens,
            non_o,
            100 * non_o / max(total_tokens, 1),
        )


if __name__ == "__main__":
    main()
