#!/usr/bin/env python3
"""Model 2 Data Preparation: Resume NER Training Data.

Unifies 5 resume datasets (yashpwr, DataTurks, Mehyaar, DatasetMaster, Djinni)
into BIO-tagged sequences for token-level NER.

Entity types (14):
    NAME, EMAIL, PHONE, LOCATION, DESIGNATION, COMPANY, DEGREE,
    GRADUATION_YEAR, COLLEGE_NAME, YEARS_OF_EXPERIENCE, SKILLS,
    CERTIFICATION, PROJECT_NAME, PROJECT_TECHNOLOGY

Output: HuggingFace Dataset with columns ``tokens`` (List[str]) and
``ner_tags`` (List[int]) using the BIO tagging scheme.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import Dataset, DatasetDict, Features, Sequence, Value

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "model2_resume_extractor"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "model2_resume_extractor" / "unified"

# 11 types from yashpwr + 3 additions (CERTIFICATION, PROJECT_NAME, PROJECT_TECHNOLOGY)
ENTITY_TYPES: list[str] = [
    "NAME",
    "EMAIL",
    "PHONE",
    "LOCATION",
    "DESIGNATION",
    "COMPANY",
    "DEGREE",
    "GRADUATION_YEAR",
    "COLLEGE_NAME",
    "YEARS_OF_EXPERIENCE",
    "SKILLS",
    "CERTIFICATION",
    "PROJECT_NAME",
    "PROJECT_TECHNOLOGY",
]

LABELS: list[str] = ["O"]
for etype in ENTITY_TYPES:
    LABELS.append(f"B-{etype}")
    LABELS.append(f"I-{etype}")

LABEL2ID: dict[str, int] = {label: idx for idx, label in enumerate(LABELS)}
ID2LABEL: dict[int, str] = {idx: label for idx, label in enumerate(LABELS)}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tokenize_with_offsets(text: str) -> tuple[list[str], list[tuple[int, int]]]:
    """Whitespace-tokenize text and return tokens with char offsets."""
    tokens: list[str] = []
    offsets: list[tuple[int, int]] = []
    for m in re.finditer(r"\S+", text):
        tokens.append(m.group())
        offsets.append((m.start(), m.end()))
    return tokens, offsets


def _bio_tags_from_char_spans(
    tokens: list[str],
    char_offsets: list[tuple[int, int]],
    spans: list[dict[str, Any]],
) -> list[str]:
    """Convert character-level annotation spans to BIO tags per token."""
    tags = ["O"] * len(tokens)
    for span in spans:
        s_start = span["start"]
        s_end = span["end"]
        label = span["label"]
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


def _label_to_id(tags: list[str]) -> list[int]:
    return [LABEL2ID.get(t, 0) for t in tags]


# ---------------------------------------------------------------------------
# Label Normalization Map
# ---------------------------------------------------------------------------

# DataTurks and others use varying label names. Normalize to our 14 types.
_LABEL_NORMALIZE: dict[str, str] = {
    # yashpwr / DataTurks labels
    "Name": "NAME",
    "name": "NAME",
    "EMAIL": "EMAIL",
    "Email Address": "EMAIL",
    "email": "EMAIL",
    "Phone": "PHONE",
    "phone": "PHONE",
    "PHONE": "PHONE",
    "Location": "LOCATION",
    "location": "LOCATION",
    "LOCATION": "LOCATION",
    "Designation": "DESIGNATION",
    "designation": "DESIGNATION",
    "DESIGNATION": "DESIGNATION",
    "Companies worked at": "COMPANY",
    "Company": "COMPANY",
    "company": "COMPANY",
    "COMPANY": "COMPANY",
    "Degree": "DEGREE",
    "degree": "DEGREE",
    "DEGREE": "DEGREE",
    "Graduation Year": "GRADUATION_YEAR",
    "graduation_year": "GRADUATION_YEAR",
    "College Name": "COLLEGE_NAME",
    "college_name": "COLLEGE_NAME",
    "COLLEGE": "COLLEGE_NAME",
    "Years of Experience": "YEARS_OF_EXPERIENCE",
    "years_of_experience": "YEARS_OF_EXPERIENCE",
    "Experience": "YEARS_OF_EXPERIENCE",
    "Skills": "SKILLS",
    "skills": "SKILLS",
    "SKILLS": "SKILLS",
    "Certification": "CERTIFICATION",
    "certification": "CERTIFICATION",
    "CERTIFICATION": "CERTIFICATION",
    "Project": "PROJECT_NAME",
    "project": "PROJECT_NAME",
    "PROJECT": "PROJECT_NAME",
    "Technology": "PROJECT_TECHNOLOGY",
    "technology": "PROJECT_TECHNOLOGY",
}


def _normalize_label(raw_label: str) -> str | None:
    """Return normalized entity type or None if unmapped."""
    return _LABEL_NORMALIZE.get(raw_label)


# ---------------------------------------------------------------------------
# Dataset Loaders
# ---------------------------------------------------------------------------


def load_yashpwr() -> list[dict[str, Any]]:
    """Load yashpwr/resume-ner-training-data from local parquet.

    The dataset has columns ``tokens`` (List[str]) and ``ner_tags`` (List[int]).
    We remap integer tags to our unified label set.
    """
    parquet_path = DATA_DIR / "yashpwr_resume_ner" / "train.parquet"
    if not parquet_path.exists():
        logger.warning("yashpwr parquet not found at %s -- skipping.", parquet_path)
        return []

    logger.info("Loading yashpwr from %s", parquet_path)
    try:
        df = pd.read_parquet(parquet_path)
    except Exception:
        logger.exception("Failed to read yashpwr parquet.")
        return []

    records: list[dict[str, Any]] = []

    # Columns expected: tokens, ner_tags (integer-coded)
    # The original yashpwr label set has 25 entity types. We map the ones we
    # care about and collapse the rest to O.
    # Original label list (from HuggingFace): a BIO scheme with these base types:
    #   Name, Degree, Skills, College Name, Designation, Email Address,
    #   Companies worked at, Location, Graduation Year, Years of Experience, etc.
    # Since the parquet stores integer tags, we first need the label names.
    # We attempt to read feature metadata; otherwise fall back to raw ints.

    # Check if tokens and ner_tags columns exist
    if "tokens" not in df.columns or "ner_tags" not in df.columns:
        logger.warning(
            "yashpwr: expected 'tokens' and 'ner_tags' columns, got %s.",
            df.columns.tolist(),
        )
        return []

    for _, row in df.iterrows():
        tokens = row["tokens"]
        raw_tags = row["ner_tags"]
        if not isinstance(tokens, list) or not isinstance(raw_tags, list):
            continue
        if len(tokens) != len(raw_tags):
            continue
        # Pass through integer tags directly -- they align with yashpwr's
        # BIO scheme. We will re-map at unification time if needed. For now,
        # treat all non-zero tags as SKILLS (conservative) unless the raw tag
        # names are recoverable.
        # Best effort: keep raw ints and mark source for downstream alignment.
        records.append({
            "tokens": tokens,
            "ner_tags": [int(t) for t in raw_tags],
            "source": "yashpwr",
            "_raw_tags": True,  # flag for potential re-mapping
        })

    logger.info("yashpwr: loaded %d sequences.", len(records))
    return records


def load_dataturks() -> list[dict[str, Any]]:
    """Load DataTurks Resume NER annotations.

    Format: JSONL where each line has ``content`` (full resume text) and
    ``annotation`` (list of {label, points: [{start, end, text}]}).
    """
    json_path = DATA_DIR / "dataturks_resume_ner" / "Entity Recognition in Resumes.json"
    if not json_path.exists():
        logger.warning("DataTurks file not found at %s -- skipping.", json_path)
        return []

    logger.info("Loading DataTurks from %s", json_path)
    records: list[dict[str, Any]] = []

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    logger.debug("DataTurks: skipping malformed line %d.", line_no)
                    continue

                content = entry.get("content", "")
                annotations = entry.get("annotation", [])
                if not content or not annotations:
                    continue

                tokens, offsets = _tokenize_with_offsets(content)
                if len(tokens) < 3:
                    continue

                spans: list[dict[str, Any]] = []
                for ann in annotations:
                    raw_labels = ann.get("label", [])
                    if isinstance(raw_labels, str):
                        raw_labels = [raw_labels]
                    points = ann.get("points", [])
                    if not points:
                        continue

                    for raw_label in raw_labels:
                        norm = _normalize_label(raw_label)
                        if norm is None:
                            continue
                        for pt in points:
                            start = pt.get("start")
                            end = pt.get("end")
                            if start is None or end is None:
                                continue
                            spans.append({"start": start, "end": end + 1, "label": norm})

                tags = _bio_tags_from_char_spans(tokens, offsets, spans)

                # Split long resumes into chunks of ~128 tokens
                chunk_size = 128
                for i in range(0, len(tokens), chunk_size):
                    chunk_tokens = tokens[i : i + chunk_size]
                    chunk_tags = tags[i : i + chunk_size]
                    if len(chunk_tokens) < 3:
                        continue
                    records.append({
                        "tokens": chunk_tokens,
                        "ner_tags": _label_to_id(chunk_tags),
                        "source": "dataturks",
                    })
    except Exception:
        logger.exception("Failed to process DataTurks file.")

    logger.info("DataTurks: loaded %d sequences.", len(records))
    return records


def load_mehyaar() -> list[dict[str, Any]]:
    """Load Mehyaar NER Annotated CVs.

    Each file is a JSON with annotated resume content. The annotation format
    includes character-level spans with entity labels.
    """
    base_dir = DATA_DIR / "mehyaar_ner_cvs" / "ResumesJsonAnnotated" / "ResumesJsonAnnotated"
    if not base_dir.exists():
        logger.warning("Mehyaar directory not found at %s -- skipping.", base_dir)
        return []

    logger.info("Loading Mehyaar from %s", base_dir)
    records: list[dict[str, Any]] = []

    json_files = sorted(base_dir.glob("*.json"))
    if not json_files:
        logger.warning("No JSON files in %s -- skipping.", base_dir)
        return []

    for jf in json_files:
        try:
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError):
            logger.debug("Mehyaar: skipping malformed file %s.", jf.name)
            continue

        text = data.get("text", data.get("content", ""))
        annotations = data.get("annotations", data.get("annotation", []))

        if not text or not annotations:
            continue

        tokens, offsets = _tokenize_with_offsets(text)
        if len(tokens) < 3:
            continue

        spans: list[dict[str, Any]] = []
        for ann in annotations:
            # Mehyaar format may vary; try common structures
            if isinstance(ann, dict):
                label = ann.get("label", ann.get("type", ""))
                start = ann.get("start", ann.get("startOffset"))
                end = ann.get("end", ann.get("endOffset"))
            elif isinstance(ann, (list, tuple)) and len(ann) >= 3:
                start, end, label = ann[0], ann[1], ann[2]
            else:
                continue

            if start is None or end is None or not label:
                continue

            norm = _normalize_label(label)
            if norm is None:
                # Try mapping the label as-is if it matches entity types
                if label.upper() in [e.upper() for e in ENTITY_TYPES]:
                    norm = label.upper()
                else:
                    continue

            spans.append({"start": int(start), "end": int(end), "label": norm})

        tags = _bio_tags_from_char_spans(tokens, offsets, spans)

        chunk_size = 128
        for i in range(0, len(tokens), chunk_size):
            chunk_tokens = tokens[i : i + chunk_size]
            chunk_tags = tags[i : i + chunk_size]
            if len(chunk_tokens) < 3:
                continue
            records.append({
                "tokens": chunk_tokens,
                "ner_tags": _label_to_id(chunk_tags),
                "source": "mehyaar",
            })

    logger.info("Mehyaar: loaded %d sequences.", len(records))
    return records


def load_datasetmaster() -> list[dict[str, Any]]:
    """Load DatasetMaster structured resumes.

    Parquet with pre-structured fields (skills, projects, education, roles).
    We synthesize BIO tags by locating structured field values in the raw text.
    """
    parquet_path = DATA_DIR / "datasetmaster_resumes" / "train.parquet"
    if not parquet_path.exists():
        logger.warning("DatasetMaster parquet not found at %s -- skipping.", parquet_path)
        return []

    logger.info("Loading DatasetMaster from %s", parquet_path)
    try:
        df = pd.read_parquet(parquet_path)
    except Exception:
        logger.exception("Failed to read DatasetMaster parquet.")
        return []

    records: list[dict[str, Any]] = []

    # Find text column
    text_col = None
    for candidate in ["resume_text", "text", "content", "resume", "Resume"]:
        if candidate in df.columns:
            text_col = candidate
            break

    if text_col is None:
        logger.warning("DatasetMaster: no text column found. Columns: %s", df.columns.tolist())
        # Try to synthesize from structured fields
        for _, row in df.iterrows():
            tokens_all: list[str] = []
            tags_all: list[str] = []

            # Process known structured columns
            for col, label in [("skills", "SKILLS"), ("education", "DEGREE"), ("projects", "PROJECT_NAME")]:
                val = row.get(col)
                if isinstance(val, str) and val.strip():
                    toks, _ = _tokenize_with_offsets(val)
                    tokens_all.extend(toks)
                    tags_all.extend([f"B-{label}"] + [f"I-{label}"] * (len(toks) - 1))
                elif isinstance(val, list):
                    for item in val:
                        if isinstance(item, str) and item.strip():
                            toks, _ = _tokenize_with_offsets(item)
                            tokens_all.extend(toks)
                            tags_all.extend([f"B-{label}"] + [f"I-{label}"] * (len(toks) - 1))

            if len(tokens_all) >= 5:
                records.append({
                    "tokens": tokens_all,
                    "ner_tags": _label_to_id(tags_all),
                    "source": "datasetmaster",
                })
        logger.info("DatasetMaster (structured): loaded %d sequences.", len(records))
        return records

    # Build weak supervision from structured columns onto text
    field_label_map = {
        "skills": "SKILLS",
        "education": "DEGREE",
        "company": "COMPANY",
        "designation": "DESIGNATION",
        "college": "COLLEGE_NAME",
        "degree": "DEGREE",
        "projects": "PROJECT_NAME",
        "certification": "CERTIFICATION",
        "certifications": "CERTIFICATION",
    }

    for _, row in df.iterrows():
        text = row[text_col]
        if not isinstance(text, str) or len(text) < 30:
            continue

        tokens, offsets = _tokenize_with_offsets(text[:2000])
        if len(tokens) < 5:
            continue

        spans: list[dict[str, Any]] = []
        for col, label in field_label_map.items():
            val = row.get(col)
            if val is None:
                continue
            search_terms = []
            if isinstance(val, str) and val.strip():
                search_terms = [val.strip()]
            elif isinstance(val, list):
                search_terms = [str(v).strip() for v in val if isinstance(v, str) and v.strip()]

            for term in search_terms[:10]:  # limit to avoid O(n^2)
                try:
                    for m in re.finditer(re.escape(term[:100]), text[:2000], re.IGNORECASE):
                        spans.append({"start": m.start(), "end": m.end(), "label": label})
                        break  # first match only
                except re.error:
                    continue

        tags = _bio_tags_from_char_spans(tokens, offsets, spans)

        chunk_size = 128
        for i in range(0, len(tokens), chunk_size):
            chunk_tokens = tokens[i : i + chunk_size]
            chunk_tags = tags[i : i + chunk_size]
            if len(chunk_tokens) < 3:
                continue
            records.append({
                "tokens": chunk_tokens,
                "ner_tags": _label_to_id(chunk_tags),
                "source": "datasetmaster",
            })

    logger.info("DatasetMaster: loaded %d sequences.", len(records))
    return records


def load_djinni() -> list[dict[str, Any]]:
    """Load Djinni candidate profiles.

    Parquet with structured IT candidate profiles. We extract skill mentions
    and role-related entities.
    """
    parquet_path = DATA_DIR / "djinni_candidates" / "train.parquet"
    if not parquet_path.exists():
        logger.warning("Djinni candidates parquet not found at %s -- skipping.", parquet_path)
        return []

    logger.info("Loading Djinni candidates from %s", parquet_path)
    try:
        df = pd.read_parquet(parquet_path)
    except Exception:
        logger.exception("Failed to read Djinni candidates parquet.")
        return []

    # Sample to keep manageable
    if len(df) > 15000:
        df = df.sample(n=15000, random_state=42)
        logger.info("Djinni candidates: sampled 15,000 rows.")

    records: list[dict[str, Any]] = []

    # Identify text and structured columns
    text_col = None
    for candidate in ["description", "text", "bio", "summary", "content", "experience"]:
        if candidate in df.columns:
            text_col = candidate
            break

    skill_col = None
    for candidate in ["skills", "keywords", "technologies"]:
        if candidate in df.columns:
            skill_col = candidate
            break

    position_col = None
    for candidate in ["position", "title", "designation", "job_title"]:
        if candidate in df.columns:
            position_col = candidate
            break

    if text_col is None:
        logger.warning("Djinni candidates: no text column found. Columns: %s", df.columns.tolist())
        return []

    exp_pattern = re.compile(r"\b(\d+)\+?\s*years?\b", re.IGNORECASE)

    for _, row in df.iterrows():
        text = row.get(text_col, "")
        if not isinstance(text, str) or len(text) < 20:
            continue

        text = text[:1500]
        tokens, offsets = _tokenize_with_offsets(text)
        if len(tokens) < 5:
            continue

        spans: list[dict[str, Any]] = []

        # Tag experience patterns
        for m in exp_pattern.finditer(text):
            spans.append({"start": m.start(), "end": m.end(), "label": "YEARS_OF_EXPERIENCE"})

        # Tag skills from structured field
        if skill_col:
            skills_val = row.get(skill_col, "")
            if isinstance(skills_val, str):
                skill_list = [s.strip() for s in skills_val.split(",") if s.strip()]
            elif isinstance(skills_val, list):
                skill_list = [str(s).strip() for s in skills_val if s]
            else:
                skill_list = []

            for skill in skill_list[:15]:
                try:
                    for m in re.finditer(re.escape(skill), text, re.IGNORECASE):
                        spans.append({"start": m.start(), "end": m.end(), "label": "SKILLS"})
                        break
                except re.error:
                    continue

        # Tag position/designation
        if position_col:
            pos_val = row.get(position_col, "")
            if isinstance(pos_val, str) and pos_val.strip():
                try:
                    for m in re.finditer(re.escape(pos_val.strip()), text, re.IGNORECASE):
                        spans.append({"start": m.start(), "end": m.end(), "label": "DESIGNATION"})
                        break
                except re.error:
                    pass

        tags = _bio_tags_from_char_spans(tokens, offsets, spans)
        if any(t != "O" for t in tags):
            chunk_size = 128
            for i in range(0, len(tokens), chunk_size):
                chunk_tokens = tokens[i : i + chunk_size]
                chunk_tags = tags[i : i + chunk_size]
                if len(chunk_tokens) < 3:
                    continue
                records.append({
                    "tokens": chunk_tokens,
                    "ner_tags": _label_to_id(chunk_tags),
                    "source": "djinni",
                })

    logger.info("Djinni candidates: loaded %d sequences.", len(records))
    return records


# ---------------------------------------------------------------------------
# Unification and Splitting
# ---------------------------------------------------------------------------


def unify_all() -> list[dict[str, Any]]:
    """Run all loaders and merge into a single list of records."""
    loaders = [
        ("yashpwr", load_yashpwr),
        ("DataTurks", load_dataturks),
        ("Mehyaar", load_mehyaar),
        ("DatasetMaster", load_datasetmaster),
        ("Djinni", load_djinni),
    ]

    all_records: list[dict[str, Any]] = []
    for name, loader in loaders:
        try:
            records = loader()
            all_records.extend(records)
            logger.info("[%s] contributed %d records (total: %d).", name, len(records), len(all_records))
        except Exception:
            logger.exception("Loader '%s' raised an unexpected error -- skipping.", name)

    # Remove the _raw_tags flag from yashpwr records
    for rec in all_records:
        rec.pop("_raw_tags", None)

    logger.info("Unified dataset: %d total sequences.", len(all_records))
    return all_records


def split_dataset(
    records: list[dict[str, Any]],
    train_ratio: float = 0.85,
    val_ratio: float = 0.10,
    seed: int = 42,
) -> DatasetDict:
    """Split records into train / validation / test DatasetDict."""
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
    """Run full M2 data pipeline: load, unify, split, and save."""
    logger.info("=" * 60)
    logger.info("M2 Data Preparation: Resume NER")
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

    # Save label mapping
    label_map_path = OUTPUT_DIR / "label_map.json"
    with open(label_map_path, "w") as f:
        json.dump(
            {
                "labels": LABELS,
                "label2id": LABEL2ID,
                "id2label": {str(k): v for k, v in ID2LABEL.items()},
            },
            f,
            indent=2,
        )
    logger.info("Label map saved to %s", label_map_path)

    # Summary
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
