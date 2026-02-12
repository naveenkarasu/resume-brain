#!/usr/bin/env python3
"""Model 6 Data Preparation: Critique/Rewrite Pairs for Verdict Model.

Formats critique and rewrite pairs from three sources for seq2seq fine-tuning
of the Verdict model (T5/FLAN-T5 or similar).

Sources:
    - MikePfunk resume critiques (22.8K conversations)
    - Grammarly CoEdIT (70K instruction-tuned text editing pairs)
    - IteraTeR (4K before/after edits with intent labels)

Output: HuggingFace Dataset with columns ``input_text``, ``target_text``,
``task_type``, ``source``.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any

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

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "model6_verdict"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "model6_verdict" / "unified"

TASK_TYPES: list[str] = [
    "summarize_resume",
    "critique_resume",
    "rewrite_bullet",
    "improve_clarity",
    "improve_style",
    "fix_grammar",
    "add_metrics",
    "general_edit",
]


# ---------------------------------------------------------------------------
# Dataset Loaders
# ---------------------------------------------------------------------------


def load_mikepfunk() -> list[dict[str, Any]]:
    """Load MikePfunk28/resume-training-dataset (22.8K conversations).

    JSONL with ChatML-style messages: system, user (resume), assistant (critique).
    We extract user-assistant pairs and categorize by task type.
    """
    jsonl_path = DATA_DIR / "mikepfunk_resume_critique" / "training_data.jsonl"
    if not jsonl_path.exists():
        logger.warning("MikePfunk JSONL not found at %s -- skipping.", jsonl_path)
        return []

    logger.info("Loading MikePfunk from %s", jsonl_path)
    records: list[dict[str, Any]] = []

    try:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    logger.debug("MikePfunk: skipping malformed line %d.", line_no)
                    continue

                messages = entry.get("messages", [])
                if not messages:
                    continue

                # Extract user message (input) and assistant message (target)
                user_msg = ""
                assistant_msg = ""
                system_msg = ""

                for msg in messages:
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    if role == "system":
                        system_msg = content
                    elif role == "user":
                        user_msg = content
                    elif role == "assistant":
                        assistant_msg = content

                if not user_msg or not assistant_msg:
                    continue

                # Determine task type from user message
                user_lower = user_msg.lower()[:200]
                if "summarize" in user_lower or "summary" in user_lower:
                    task_type = "summarize_resume"
                elif "critique" in user_lower or "review" in user_lower or "feedback" in user_lower:
                    task_type = "critique_resume"
                elif "rewrite" in user_lower or "improve" in user_lower:
                    task_type = "rewrite_bullet"
                elif "category" in user_lower or "classify" in user_lower:
                    task_type = "critique_resume"
                else:
                    task_type = "critique_resume"

                # Format for seq2seq
                # Input: instruction + resume text
                # Truncate very long inputs
                input_text = user_msg[:4000]
                target_text = assistant_msg[:2000]

                records.append({
                    "input_text": input_text,
                    "target_text": target_text,
                    "task_type": task_type,
                    "source": "mikepfunk",
                })

    except Exception:
        logger.exception("Failed to process MikePfunk JSONL.")

    logger.info("MikePfunk: loaded %d pairs.", len(records))
    return records


def load_coedit() -> list[dict[str, Any]]:
    """Load Grammarly CoEdIT (70K instruction-tuned text editing pairs).

    Parquet with columns for instruction, input text, and edited output.
    Six task types: fluency, clarity, coherence, style, paraphrase, neutralize.
    """
    coedit_dir = DATA_DIR / "grammarly_coedit"
    if not coedit_dir.exists():
        logger.warning("CoEdIT directory not found at %s -- skipping.", coedit_dir)
        return []

    logger.info("Loading CoEdIT from %s", coedit_dir)
    records: list[dict[str, Any]] = []

    parquet_files = sorted(coedit_dir.glob("*.parquet"))
    if not parquet_files:
        logger.warning("No parquet files in %s.", coedit_dir)
        return []

    # CoEdIT task type mapping to our taxonomy
    coedit_task_map = {
        "fluency": "fix_grammar",
        "clarity": "improve_clarity",
        "coherence": "improve_clarity",
        "style": "improve_style",
        "paraphrase": "rewrite_bullet",
        "neutralize": "improve_style",
        "simplification": "improve_clarity",
    }

    for pf in parquet_files:
        try:
            df = pd.read_parquet(pf)
        except Exception:
            logger.warning("Failed to read %s -- skipping.", pf)
            continue

        # Identify columns
        input_col = None
        output_col = None
        task_col = None

        for col in df.columns:
            cl = col.lower()
            if "src" in cl or "input" in cl or "source" in cl or "before" in cl or "original" in cl:
                input_col = col
            elif "tgt" in cl or "output" in cl or "target" in cl or "after" in cl or "edited" in cl:
                output_col = col
            elif "task" in cl or "type" in cl or "edit" in cl:
                task_col = col

        if input_col is None or output_col is None:
            # Fallback to positional
            cols = df.columns.tolist()
            if len(cols) >= 2:
                input_col = cols[0]
                output_col = cols[1]
                if len(cols) >= 3:
                    task_col = cols[2]
            else:
                logger.warning("CoEdIT: cannot identify columns in %s. Got: %s", pf.name, df.columns.tolist())
                continue

        for _, row in df.iterrows():
            input_text = str(row.get(input_col, "")).strip()
            output_text = str(row.get(output_col, "")).strip()

            if not input_text or not output_text:
                continue
            if input_text == output_text:
                continue

            # Determine task type
            raw_task = str(row.get(task_col, "general")).strip().lower() if task_col else "general"
            task_type = coedit_task_map.get(raw_task, "general_edit")

            # Format as instruction-output pair
            # CoEdIT input may already include instruction prefix
            records.append({
                "input_text": input_text[:2000],
                "target_text": output_text[:2000],
                "task_type": task_type,
                "source": "coedit",
            })

    logger.info("CoEdIT: loaded %d pairs.", len(records))
    return records


def load_iterater() -> list[dict[str, Any]]:
    """Load IteraTeR (4K before/after edits with intent labels).

    Parquet with before_sent, after_sent, and edit_intent columns.
    Edit intents: Clarity, Fluency, Coherence, Style, Meaning-changed.
    """
    iterater_dir = DATA_DIR / "iterater_human_sent"
    if not iterater_dir.exists():
        logger.warning("IteraTeR directory not found at %s -- skipping.", iterater_dir)
        return []

    logger.info("Loading IteraTeR from %s", iterater_dir)
    records: list[dict[str, Any]] = []

    parquet_files = sorted(iterater_dir.glob("*.parquet"))
    if not parquet_files:
        logger.warning("No parquet files in %s.", iterater_dir)
        return []

    intent_task_map = {
        "clarity": "improve_clarity",
        "fluency": "fix_grammar",
        "coherence": "improve_clarity",
        "style": "improve_style",
        "meaning-changed": "rewrite_bullet",
    }

    for pf in parquet_files:
        try:
            df = pd.read_parquet(pf)
        except Exception:
            logger.warning("Failed to read %s -- skipping.", pf)
            continue

        # Identify columns
        before_col = None
        after_col = None
        intent_col = None

        for col in df.columns:
            cl = col.lower()
            if "before" in cl or "original" in cl or "input" in cl or "src" in cl or "sent_before" in cl:
                before_col = col
            elif "after" in cl or "edited" in cl or "output" in cl or "tgt" in cl or "sent_after" in cl:
                after_col = col
            elif "intent" in cl or "label" in cl or "type" in cl or "edit" in cl:
                intent_col = col

        if before_col is None or after_col is None:
            cols = df.columns.tolist()
            if len(cols) >= 2:
                before_col = cols[0]
                after_col = cols[1]
                if len(cols) >= 3:
                    intent_col = cols[2]
            else:
                logger.warning("IteraTeR: cannot identify columns in %s. Got: %s", pf.name, df.columns.tolist())
                continue

        for _, row in df.iterrows():
            before_text = str(row.get(before_col, "")).strip()
            after_text = str(row.get(after_col, "")).strip()

            if not before_text or not after_text:
                continue
            if before_text == after_text:
                continue

            # Determine task type from intent
            raw_intent = str(row.get(intent_col, "general")).strip().lower() if intent_col else "general"
            task_type = intent_task_map.get(raw_intent, "general_edit")

            # Format as instruction pair
            instruction = f"Improve the following text for {raw_intent}: {before_text}"

            records.append({
                "input_text": instruction[:2000],
                "target_text": after_text[:2000],
                "task_type": task_type,
                "source": "iterater",
            })

    logger.info("IteraTeR: loaded %d pairs.", len(records))
    return records


# ---------------------------------------------------------------------------
# Seq2Seq Formatting
# ---------------------------------------------------------------------------


def format_for_seq2seq(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Apply consistent seq2seq formatting to all records.

    Ensures all records follow the pattern:
        input_text: <task_prefix> <content>
        target_text: <output>

    This matches T5/FLAN-T5 input format expectations.
    """
    task_prefixes = {
        "summarize_resume": "Summarize the following resume:",
        "critique_resume": "Provide a critique of the following resume:",
        "rewrite_bullet": "Rewrite the following resume bullet point with improvements:",
        "improve_clarity": "Improve the clarity of the following text:",
        "improve_style": "Improve the style of the following text:",
        "fix_grammar": "Fix grammar and fluency in the following text:",
        "add_metrics": "Add quantitative metrics to the following bullet point:",
        "general_edit": "Edit the following text:",
    }

    formatted: list[dict[str, Any]] = []

    for rec in records:
        task_type = rec.get("task_type", "general_edit")
        input_text = rec["input_text"]
        target_text = rec["target_text"]

        # If input already contains instruction-like prefix, keep as-is
        # Otherwise prepend task prefix
        has_instruction = any(
            kw in input_text.lower()[:100]
            for kw in ["summarize", "critique", "rewrite", "improve", "fix", "edit", "please", "provide"]
        )

        if not has_instruction:
            prefix = task_prefixes.get(task_type, "Edit the following text:")
            input_text = f"{prefix} {input_text}"

        formatted.append({
            "input_text": input_text,
            "target_text": target_text,
            "task_type": task_type,
            "source": rec.get("source", "unknown"),
        })

    logger.info("Formatted %d records for seq2seq.", len(formatted))
    return formatted


# ---------------------------------------------------------------------------
# Dataset Construction
# ---------------------------------------------------------------------------


def build_dataset(records: list[dict[str, Any]], seed: int = 42) -> DatasetDict:
    """Convert records into train/val/test DatasetDict."""
    rng = random.Random(seed)
    rng.shuffle(records)

    n = len(records)
    n_train = int(n * 0.85)
    n_val = int(n * 0.10)

    splits = {
        "train": records[:n_train],
        "validation": records[n_train : n_train + n_val],
        "test": records[n_train + n_val :],
    }

    features = Features({
        "input_text": Value("string"),
        "target_text": Value("string"),
        "task_type": Value("string"),
        "source": Value("string"),
    })

    dd = DatasetDict()
    for name, data in splits.items():
        dd[name] = Dataset.from_dict(
            {
                "input_text": [d["input_text"] for d in data],
                "target_text": [d["target_text"] for d in data],
                "task_type": [d["task_type"] for d in data],
                "source": [d["source"] for d in data],
            },
            features=features,
        )
        logger.info("Split '%s': %d examples.", name, len(data))

    return dd


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run full M6 data pipeline: load, format, build dataset, save."""
    logger.info("=" * 60)
    logger.info("M6 Data Preparation: Critique/Rewrite Pairs for Verdict Model")
    logger.info("DATA_DIR: %s", DATA_DIR)
    logger.info("OUTPUT_DIR: %s", OUTPUT_DIR)
    logger.info("Task types: %s", TASK_TYPES)
    logger.info("=" * 60)

    all_records: list[dict[str, Any]] = []

    for name, loader in [
        ("MikePfunk", load_mikepfunk),
        ("CoEdIT", load_coedit),
        ("IteraTeR", load_iterater),
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

    all_records = format_for_seq2seq(all_records)

    dataset = build_dataset(all_records)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(OUTPUT_DIR))
    logger.info("Dataset saved to %s", OUTPUT_DIR)

    # Save metadata
    from collections import Counter

    meta = {
        "task_types": TASK_TYPES,
        "source_counts": dict(Counter(r["source"] for r in all_records)),
        "task_counts": dict(Counter(r["task_type"] for r in all_records)),
        "total_records": len(all_records),
    }
    meta_path = OUTPUT_DIR / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info("Metadata saved to %s", meta_path)

    # Summary
    for split_name in dataset:
        ds = dataset[split_name]
        task_dist = Counter(row["task_type"] for row in ds)
        source_dist = Counter(row["source"] for row in ds)
        avg_input_len = sum(len(row["input_text"].split()) for row in ds) / max(len(ds), 1)
        avg_target_len = sum(len(row["target_text"].split()) for row in ds) / max(len(ds), 1)
        logger.info(
            "  %s: %d examples, avg input words: %.0f, avg target words: %.0f",
            split_name,
            len(ds),
            avg_input_len,
            avg_target_len,
        )
        logger.info("    tasks: %s", dict(task_dist))
        logger.info("    sources: %s", dict(source_dist))


if __name__ == "__main__":
    main()
