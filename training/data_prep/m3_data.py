#!/usr/bin/env python3
"""Model 3 Data Preparation: Skill Triplets for Contrastive Learning.

Builds (anchor_skill, positive_skill, negative_skill) triplets from ESCO,
Tabiya, Nesta, and StackLite data for training a skill embedding model.

Sources:
    - TechWolf ESCO Skill Sentences (138K pairs)
    - Tabiya ESCO Open Dataset (skill synonyms / alt labels)
    - Nesta UK Skills Taxonomy (10.5K skills, 143 clusters)
    - StackLite (SO tag co-occurrence for hard negatives)

Output: HuggingFace Dataset with columns ``anchor``, ``positive``,
``negative`` (all strings).
"""

from __future__ import annotations

import csv
import gzip
import json
import logging
import random
from collections import defaultdict
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

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "model3_skills_comparator"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "model3_skills_comparator" / "unified"


# ---------------------------------------------------------------------------
# Dataset Loaders
# ---------------------------------------------------------------------------


def load_esco_sentences() -> list[dict[str, str]]:
    """Load TechWolf Synthetic-ESCO-Skill-Sentences.

    Parquet with skill-sentence contrastive pairs. We extract unique skill
    labels and their associated sentences for anchor-positive construction.

    Returns list of dicts: {skill, sentence}.
    """
    parquet_path = DATA_DIR / "techwolf_esco_sentences" / "train.parquet"
    if not parquet_path.exists():
        logger.warning("TechWolf ESCO sentences not found at %s -- skipping.", parquet_path)
        return []

    logger.info("Loading TechWolf ESCO sentences from %s", parquet_path)
    try:
        df = pd.read_parquet(parquet_path)
    except Exception:
        logger.exception("Failed to read TechWolf parquet.")
        return []

    # Expected columns: skill_label (or similar), sentence
    skill_col = None
    sentence_col = None
    for col in df.columns:
        cl = col.lower()
        if "skill" in cl and "sent" not in cl:
            skill_col = col
        elif "sent" in cl or "text" in cl or "description" in cl:
            sentence_col = col

    if skill_col is None or sentence_col is None:
        # Fallback: assume first two columns
        cols = df.columns.tolist()
        if len(cols) >= 2:
            skill_col, sentence_col = cols[0], cols[1]
        else:
            logger.warning("TechWolf: cannot identify columns. Got: %s", df.columns.tolist())
            return []

    records = []
    for _, row in df.iterrows():
        skill = str(row[skill_col]).strip()
        sentence = str(row[sentence_col]).strip()
        if skill and sentence and len(skill) > 1:
            records.append({"skill": skill, "sentence": sentence})

    logger.info("TechWolf ESCO: loaded %d skill-sentence pairs.", len(records))
    return records


def load_tabiya_synonyms() -> dict[str, list[str]]:
    """Load Tabiya ESCO Open Dataset for skill synonyms / alternative labels.

    Returns a dict mapping canonical skill URI or label to list of alt labels.
    """
    csv_dir = DATA_DIR / "tabiya_esco" / "tabiya-esco-v1.1.1" / "csv"
    skills_csv = csv_dir / "skills.csv"

    if not skills_csv.exists():
        logger.warning("Tabiya skills.csv not found at %s -- skipping.", skills_csv)
        return {}

    logger.info("Loading Tabiya ESCO synonyms from %s", skills_csv)
    synonyms: dict[str, list[str]] = defaultdict(list)

    try:
        df = pd.read_csv(skills_csv)
    except Exception:
        logger.exception("Failed to read Tabiya skills CSV.")
        return {}

    # Expected columns: preferredLabel, altLabels (pipe-separated), conceptUri
    preferred_col = None
    alt_col = None
    for col in df.columns:
        cl = col.lower()
        if "preferred" in cl and "label" in cl:
            preferred_col = col
        elif "alt" in cl and "label" in cl:
            alt_col = col

    if preferred_col is None:
        # Fallback
        for col in df.columns:
            if "label" in col.lower() or "name" in col.lower():
                preferred_col = col
                break

    if preferred_col is None:
        logger.warning("Tabiya: cannot identify preferred label column. Got: %s", df.columns.tolist())
        return {}

    for _, row in df.iterrows():
        pref = str(row[preferred_col]).strip()
        if not pref or pref == "nan":
            continue
        alts = []
        if alt_col and pd.notna(row.get(alt_col)):
            alt_str = str(row[alt_col])
            # Alt labels are often pipe-separated or newline-separated
            for sep in ["|", "\n", ";"]:
                if sep in alt_str:
                    alts = [a.strip() for a in alt_str.split(sep) if a.strip()]
                    break
            if not alts and alt_str.strip():
                alts = [alt_str.strip()]
        synonyms[pref] = alts

    logger.info("Tabiya ESCO: loaded %d skills with synonyms.", len(synonyms))
    return synonyms


def load_nesta_clusters() -> dict[int, list[str]]:
    """Load Nesta UK Skills Taxonomy cluster assignments.

    Returns dict mapping cluster_id to list of skill names in that cluster.
    """
    # Nesta data may be in various locations within the repo structure
    nesta_base = DATA_DIR / "nesta_skills_taxonomy"

    # Look for taxonomy data files
    candidates = [
        nesta_base / "outputs" / "skills_taxonomy" / "skills_taxonomy.json",
        nesta_base / "skills_taxonomy_v2" / "taxonomy_data",
        nesta_base / "outputs",
    ]

    clusters: dict[int, list[str]] = defaultdict(list)

    # Try to find cluster data in the repo
    json_files = list(nesta_base.rglob("*.json"))
    csv_files = list(nesta_base.rglob("*.csv"))

    if not json_files and not csv_files:
        logger.warning("Nesta: no data files found under %s -- skipping.", nesta_base)
        return {}

    logger.info("Loading Nesta clusters from %s", nesta_base)

    # Try JSON files first
    for jf in json_files:
        if "cluster" in jf.name.lower() or "taxonomy" in jf.name.lower():
            try:
                with open(jf, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    for key, val in data.items():
                        if isinstance(val, list):
                            cluster_id = hash(key) % 1000
                            clusters[cluster_id] = [str(v) for v in val if v]
                        elif isinstance(val, dict) and "skills" in val:
                            cluster_id = hash(key) % 1000
                            clusters[cluster_id] = [str(s) for s in val["skills"] if s]
                logger.info("Nesta: loaded clusters from %s", jf.name)
                break
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue

    # Try CSV files
    if not clusters:
        for cf in csv_files:
            if "skill" in cf.name.lower() or "cluster" in cf.name.lower():
                try:
                    df = pd.read_csv(cf)
                    # Look for skill name and cluster columns
                    skill_col = None
                    cluster_col = None
                    for col in df.columns:
                        cl = col.lower()
                        if "skill" in cl or "name" in cl or "label" in cl:
                            skill_col = col
                        elif "cluster" in cl or "group" in cl or "category" in cl:
                            cluster_col = col
                    if skill_col and cluster_col:
                        for _, row in df.iterrows():
                            skill = str(row[skill_col]).strip()
                            cid = row[cluster_col]
                            if skill and skill != "nan":
                                clusters[int(hash(str(cid)) % 1000)].append(skill)
                        logger.info("Nesta: loaded clusters from %s", cf.name)
                        break
                except Exception:
                    continue

    logger.info("Nesta: loaded %d clusters with %d total skills.", len(clusters), sum(len(v) for v in clusters.values()))
    return clusters


def load_stacklite_cooccurrence() -> dict[str, set[str]]:
    """Load StackLite tag co-occurrence data.

    Returns dict mapping tag -> set of co-occurring tags (from same question).
    """
    tags_path = DATA_DIR / "stacklite" / "question_tags.csv.gz"
    if not tags_path.exists():
        logger.warning("StackLite tags not found at %s -- skipping.", tags_path)
        return {}

    logger.info("Loading StackLite co-occurrence from %s", tags_path)

    question_tags: dict[int, list[str]] = defaultdict(list)
    try:
        with gzip.open(str(tags_path), "rt", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                qid = int(row.get("Id", row.get("id", 0)))
                tag = row.get("Tag", row.get("tag", "")).strip()
                if qid and tag:
                    question_tags[qid].append(tag)
                if i >= 2_000_000:  # limit for memory
                    break
    except Exception:
        logger.exception("Failed to read StackLite tags.")
        return {}

    # Build co-occurrence
    cooccur: dict[str, set[str]] = defaultdict(set)
    for qid, tags in question_tags.items():
        for tag in tags:
            for other in tags:
                if other != tag:
                    cooccur[tag].add(other)

    logger.info("StackLite: built co-occurrence for %d tags.", len(cooccur))
    return cooccur


# ---------------------------------------------------------------------------
# Triplet Construction
# ---------------------------------------------------------------------------


def build_triplets(
    esco_pairs: list[dict[str, str]],
    synonyms: dict[str, list[str]],
    clusters: dict[int, list[str]],
    cooccur: dict[str, set[str]],
    max_triplets: int = 200_000,
    seed: int = 42,
) -> list[dict[str, str]]:
    """Build (anchor, positive, negative) triplets from all data sources.

    Strategy:
        1. Synonym-based positives: anchor = preferred label, positive = alt label.
        2. Cluster-based positives: skills in the same Nesta cluster are positives.
        3. Co-occurrence positives: frequently co-occurring SO tags are positives.
        4. Negatives: random skills from different clusters or non-co-occurring.
    """
    rng = random.Random(seed)
    triplets: list[dict[str, str]] = []

    # Build a global skill vocabulary for negative sampling
    all_skills: list[str] = list(set(
        list(synonyms.keys())
        + [alt for alts in synonyms.values() for alt in alts]
        + [skill for skills in clusters.values() for skill in skills]
        + list(cooccur.keys())
    ))

    if len(all_skills) < 10:
        logger.warning("Insufficient skill vocabulary (%d) for triplet construction.", len(all_skills))
        return []

    all_skills_set = set(all_skills)

    # 1. Synonym-based triplets
    for pref, alts in synonyms.items():
        for alt in alts:
            if alt == pref:
                continue
            neg = rng.choice(all_skills)
            while neg == pref or neg == alt:
                neg = rng.choice(all_skills)
            triplets.append({"anchor": pref, "positive": alt, "negative": neg})
            if len(triplets) >= max_triplets:
                break
        if len(triplets) >= max_triplets:
            break

    # 2. Cluster-based triplets
    cluster_list = list(clusters.items())
    all_cluster_skills = [s for skills in clusters.values() for s in skills]
    if cluster_list and all_cluster_skills:
        for cid, skills in cluster_list:
            if len(skills) < 2:
                continue
            other_cluster_skills = [s for oid, oss in cluster_list if oid != cid for s in oss]
            if not other_cluster_skills:
                other_cluster_skills = all_skills

            for i in range(len(skills)):
                for j in range(i + 1, min(i + 3, len(skills))):
                    anchor = skills[i]
                    positive = skills[j]
                    negative = rng.choice(other_cluster_skills) if other_cluster_skills else rng.choice(all_skills)
                    triplets.append({"anchor": anchor, "positive": positive, "negative": negative})
                    if len(triplets) >= max_triplets:
                        break
                if len(triplets) >= max_triplets:
                    break
            if len(triplets) >= max_triplets:
                break

    # 3. ESCO sentence-based triplets (skill <-> skill via shared sentences)
    skill_to_sentences: dict[str, list[str]] = defaultdict(list)
    for pair in esco_pairs:
        skill_to_sentences[pair["skill"]].append(pair["sentence"])

    skill_keys = list(skill_to_sentences.keys())
    if len(skill_keys) >= 2 and len(triplets) < max_triplets:
        for skill in skill_keys:
            sentences = skill_to_sentences[skill]
            if len(sentences) < 2:
                continue
            # Two sentences about the same skill are positive pairs
            for i in range(min(len(sentences), 3)):
                for j in range(i + 1, min(len(sentences), 4)):
                    neg_skill = rng.choice(skill_keys)
                    while neg_skill == skill:
                        neg_skill = rng.choice(skill_keys)
                    neg_sentences = skill_to_sentences[neg_skill]
                    neg_sentence = rng.choice(neg_sentences) if neg_sentences else neg_skill
                    triplets.append({
                        "anchor": sentences[i],
                        "positive": sentences[j],
                        "negative": neg_sentence,
                    })
                    if len(triplets) >= max_triplets:
                        break
                if len(triplets) >= max_triplets:
                    break
            if len(triplets) >= max_triplets:
                break

    rng.shuffle(triplets)
    triplets = triplets[:max_triplets]

    logger.info("Built %d triplets.", len(triplets))
    return triplets


def create_hard_negatives(
    triplets: list[dict[str, str]],
    cooccur: dict[str, set[str]],
    all_skills: list[str] | None = None,
    fraction: float = 0.3,
    seed: int = 42,
) -> list[dict[str, str]]:
    """Replace a fraction of easy negatives with hard negatives.

    Hard negatives are skills that co-occur with the anchor (related but
    different) rather than completely random skills.
    """
    rng = random.Random(seed)

    if not cooccur or not triplets:
        return triplets

    n_replace = int(len(triplets) * fraction)
    indices = rng.sample(range(len(triplets)), min(n_replace, len(triplets)))

    replaced = 0
    for idx in indices:
        anchor = triplets[idx]["anchor"]
        positive = triplets[idx]["positive"]

        # Find co-occurring skills that are NOT the positive
        cooccurring = cooccur.get(anchor, set()) - {positive, anchor}
        if not cooccurring:
            # Try lowercase match
            cooccurring = cooccur.get(anchor.lower(), set()) - {positive.lower(), anchor.lower()}

        if cooccurring:
            hard_neg = rng.choice(list(cooccurring))
            triplets[idx]["negative"] = hard_neg
            replaced += 1

    logger.info("Replaced %d / %d negatives with hard negatives.", replaced, len(triplets))
    return triplets


# ---------------------------------------------------------------------------
# Dataset Construction
# ---------------------------------------------------------------------------


def build_dataset(triplets: list[dict[str, str]], seed: int = 42) -> DatasetDict:
    """Convert triplets list into a train/val/test DatasetDict."""
    rng = random.Random(seed)
    rng.shuffle(triplets)

    n = len(triplets)
    n_train = int(n * 0.85)
    n_val = int(n * 0.10)

    splits = {
        "train": triplets[:n_train],
        "validation": triplets[n_train : n_train + n_val],
        "test": triplets[n_train + n_val :],
    }

    features = Features({
        "anchor": Value("string"),
        "positive": Value("string"),
        "negative": Value("string"),
    })

    dd = DatasetDict()
    for name, data in splits.items():
        dd[name] = Dataset.from_dict(
            {
                "anchor": [d["anchor"] for d in data],
                "positive": [d["positive"] for d in data],
                "negative": [d["negative"] for d in data],
            },
            features=features,
        )
        logger.info("Split '%s': %d triplets.", name, len(data))

    return dd


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run full M3 data pipeline: load sources, build triplets, save."""
    logger.info("=" * 60)
    logger.info("M3 Data Preparation: Skill Triplets for Contrastive Learning")
    logger.info("DATA_DIR: %s", DATA_DIR)
    logger.info("OUTPUT_DIR: %s", OUTPUT_DIR)
    logger.info("=" * 60)

    esco_pairs = load_esco_sentences()
    synonyms = load_tabiya_synonyms()
    clusters = load_nesta_clusters()
    cooccur = load_stacklite_cooccurrence()

    triplets = build_triplets(esco_pairs, synonyms, clusters, cooccur)
    if not triplets:
        logger.error("No triplets produced -- aborting.")
        return

    triplets = create_hard_negatives(triplets, cooccur)

    dataset = build_dataset(triplets)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(OUTPUT_DIR))
    logger.info("Dataset saved to %s", OUTPUT_DIR)

    # Summary
    logger.info("Triplet statistics:")
    for split_name in dataset:
        ds = dataset[split_name]
        avg_anchor_len = sum(len(row["anchor"].split()) for row in ds) / max(len(ds), 1)
        avg_pos_len = sum(len(row["positive"].split()) for row in ds) / max(len(ds), 1)
        logger.info(
            "  %s: %d triplets, avg anchor words: %.1f, avg positive words: %.1f",
            split_name,
            len(ds),
            avg_anchor_len,
            avg_pos_len,
        )


if __name__ == "__main__":
    main()
