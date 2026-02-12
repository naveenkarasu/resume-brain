#!/usr/bin/env python3
"""Model 4 Data Preparation: Experience/Education Feature Vectors.

Builds structured feature vectors from JobHop, Karrierewege, and job
classification datasets for training a LightGBM experience/education comparator.

Features (14) – aligned with inference (m4_exp_edu_comparator.py):
    years_gap, resume_years, required_years, title_cosine_sim, edu_gap,
    field_match, num_roles, avg_tenure_months, has_leadership, career_velocity,
    domain_sim, num_skills, edu_level_ordinal, jd_edu_ordinal

Sources:
    - JobHop (1.68M experiences from 391K resumes)
    - Karrierewege (100K sample, career paths with skills)
    - jobclassinfo2.csv (66 job classes with education/experience/domain)

Output: HuggingFace Dataset with 14 feature columns + label column.
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

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "model4_exp_edu_comparator"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "model4_exp_edu_comparator" / "unified"

# Must match inference FEATURE_NAMES in m4_exp_edu_comparator.py exactly
FEATURE_NAMES: list[str] = [
    "years_gap",
    "resume_years",
    "required_years",
    "title_cosine_sim",
    "edu_gap",
    "field_match",
    "num_roles",
    "avg_tenure_months",
    "has_leadership",
    "career_velocity",
    "domain_sim",
    "num_skills",
    "edu_level_ordinal",
    "jd_edu_ordinal",
]

# Education level mapping (numeric ordinal) – matches inference _EDU_ORDINAL
EDU_LEVELS: dict[str, int] = {
    "high_school": 0,
    "associate": 1,
    "bachelor": 2,
    "bachelors": 2,
    "master": 3,
    "masters": 3,
    "mba": 3,
    "phd": 4,
    "doctorate": 4,
}

# Map jobclassinfo2.csv EducationLevel (1-6) to ordinal (0-4)
_JOB_CLASS_EDU_MAP: dict[int, int] = {
    1: 0,  # high school
    2: 1,  # associate
    3: 2,  # bachelor
    4: 3,  # master
    5: 4,  # phd
    6: 4,  # phd (top tier)
}

# Map jobclassinfo2.csv EducationLevel (1-6) to education level names
_JOB_CLASS_EDU_NAME: dict[int, str] = {
    1: "high_school",
    2: "associate",
    3: "bachelor",
    4: "master",
    5: "phd",
    6: "phd",
}

# Leadership title keywords – matches inference _LEADERSHIP_KEYWORDS
_LEADERSHIP_KEYWORDS = {"lead", "senior", "principal", "staff", "director", "vp",
                         "head", "chief", "manager", "architect"}

# Map JobFamilyDescription to a synthetic education field
_FAMILY_TO_FIELD: dict[str, str] = {
    "accounting and finance": "accounting",
    "administrative support": "business administration",
    "baker": "culinary arts",
    "buildings and facilities": "facilities management",
    "buyer": "supply chain management",
    "cashier": "retail management",
    "communications and media": "communications",
    "corporate research": "data science",
    "finance  and accounting": "finance",
    "finance and accounting": "finance",
    "human resources": "human resources",
    "meat cutter": "food processing",
    "produce": "agriculture",
    "secretary": "office administration",
    "stockkeeping": "logistics",
    "systems analyst": "information technology",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _simple_cosine(a: str, b: str) -> float:
    """Compute word-level cosine similarity between two strings."""
    if not a or not b:
        return 0.0
    tokens_a = Counter(a.lower().split())
    tokens_b = Counter(b.lower().split())
    intersection = set(tokens_a.keys()) & set(tokens_b.keys())
    if not intersection:
        return 0.0
    dot = sum(tokens_a[t] * tokens_b[t] for t in intersection)
    norm_a = math.sqrt(sum(v ** 2 for v in tokens_a.values()))
    norm_b = math.sqrt(sum(v ** 2 for v in tokens_b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _has_leadership(title: str) -> float:
    """Check if title contains any leadership keyword. Returns 1.0 or 0.0."""
    if not title:
        return 0.0
    words = set(title.lower().split())
    return 1.0 if words & _LEADERSHIP_KEYWORDS else 0.0


def _parse_education_level(edu_str: str) -> int:
    """Parse education string into ordinal level."""
    if not edu_str:
        return 0
    edu_lower = edu_str.lower()
    for keyword, level in sorted(EDU_LEVELS.items(), key=lambda x: -x[1]):
        if keyword in edu_lower:
            return level
    return 0


def _word_overlap_score(a: str, b: str) -> float:
    """Compute word overlap ratio between two strings."""
    if not a or not b:
        return 0.0
    words_a = set(a.lower().split())
    words_b = set(b.lower().split())
    if not words_a or not words_b:
        return 0.0
    overlap = len(words_a & words_b)
    return overlap / min(len(words_a), len(words_b))


# ---------------------------------------------------------------------------
# Dataset Loaders
# ---------------------------------------------------------------------------


def load_jobhop() -> pd.DataFrame:
    """Load JobHop career trajectory data."""
    records = []

    for split in ["train", "validation", "test"]:
        path = DATA_DIR / "jobhop" / f"{split}.parquet"
        if not path.exists():
            continue
        logger.info("Loading JobHop %s from %s", split, path)
        try:
            df = pd.read_parquet(path)
            records.append(df)
        except Exception:
            logger.exception("Failed to read JobHop %s.", split)

    if not records:
        logger.warning("No JobHop data found at %s.", DATA_DIR / "jobhop")
        return pd.DataFrame()

    df = pd.concat(records, ignore_index=True)

    if len(df) > 200_000:
        df = df.sample(n=200_000, random_state=42)
        logger.info("JobHop: sampled 200K of %d total rows.", len(df))

    logger.info("JobHop: loaded %d rows with columns %s.", len(df), df.columns.tolist())
    return df


def load_karrierewege() -> pd.DataFrame:
    """Load Karrierewege career paths (100K sample)."""
    path = DATA_DIR / "karrierewege" / "train_sample_100k.parquet"
    if not path.exists():
        logger.warning("Karrierewege not found at %s -- skipping.", path)
        return pd.DataFrame()

    logger.info("Loading Karrierewege from %s", path)
    try:
        df = pd.read_parquet(path)
    except Exception:
        logger.exception("Failed to read Karrierewege parquet.")
        return pd.DataFrame()

    logger.info("Karrierewege: loaded %d rows with columns %s.", len(df), df.columns.tolist())
    return df


def _load_job_classification() -> pd.DataFrame:
    """Load job classification data from jobclassinfo2.csv.

    Returns a DataFrame with columns: JobClassDescription, JobFamilyDescription,
    EducationLevel, Experience, etc.
    """
    path = DATA_DIR / "job_classification" / "jobclassinfo2.csv"
    if not path.exists():
        logger.warning("Job classification not found at %s", path)
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
    except Exception:
        logger.exception("Failed to read job classification CSV.")
        return pd.DataFrame()

    logger.info("Job classification: loaded %d entries.", len(df))
    return df


# Module-level job classification (loaded once)
_JOB_CLASS_DF: pd.DataFrame | None = None


def _get_job_classification() -> pd.DataFrame:
    """Get (and cache) the job classification DataFrame."""
    global _JOB_CLASS_DF
    if _JOB_CLASS_DF is None:
        _JOB_CLASS_DF = _load_job_classification()
    return _JOB_CLASS_DF


def _fuzzy_match_job_class(title: str, job_class_df: pd.DataFrame) -> pd.Series | None:
    """Fuzzy-match a job title to the nearest job class row.

    Uses word overlap between the title and JobClassDescription.
    Returns the best-matching row, or None if no match found.
    """
    if job_class_df.empty or not title:
        return None

    best_score = 0.0
    best_row = None
    title_lower = title.lower()

    for _, row in job_class_df.iterrows():
        desc = str(row["JobClassDescription"]).lower()
        score = _word_overlap_score(title_lower, desc)
        if score > best_score:
            best_score = score
            best_row = row

    if best_score >= 0.3:
        return best_row
    return None


def _enrich_with_job_class(
    resume_exp: dict[str, Any],
    jd_req: dict[str, Any],
    job_class_df: pd.DataFrame,
    rng: random.Random,
) -> None:
    """Enrich resume_exp and jd_req dicts with job classification data in-place.

    Fuzzy-matches titles to job classes and populates education_level,
    education_field, and domain from the matched class.
    """
    if job_class_df.empty:
        return

    # Enrich resume side
    if not resume_exp.get("education_level"):
        match = _fuzzy_match_job_class(str(resume_exp.get("title", "")), job_class_df)
        if match is None:
            # Random fallback
            match = job_class_df.iloc[rng.randint(0, len(job_class_df) - 1)]
        edu_level_raw = int(match["EducationLevel"])
        resume_exp["education_level"] = _JOB_CLASS_EDU_NAME.get(edu_level_raw, "bachelor")
        family = str(match["JobFamilyDescription"]).strip().lower()
        resume_exp["education_field"] = _FAMILY_TO_FIELD.get(family, family)
        if not resume_exp.get("domain"):
            resume_exp["domain"] = str(match["JobFamilyDescription"]).strip()

    # Enrich JD side
    if not jd_req.get("education_level"):
        match = _fuzzy_match_job_class(str(jd_req.get("title", "")), job_class_df)
        if match is None:
            match = job_class_df.iloc[rng.randint(0, len(job_class_df) - 1)]
        edu_level_raw = int(match["EducationLevel"])
        jd_req["education_level"] = _JOB_CLASS_EDU_NAME.get(edu_level_raw, "bachelor")
        family = str(match["JobFamilyDescription"]).strip().lower()
        jd_req["education_field"] = _FAMILY_TO_FIELD.get(family, family)
        if not jd_req.get("domain"):
            jd_req["domain"] = str(match["JobFamilyDescription"]).strip()


# ---------------------------------------------------------------------------
# Feature Extraction (aligned with inference)
# ---------------------------------------------------------------------------


def extract_features(
    resume_exp: dict[str, Any],
    jd_req: dict[str, Any],
) -> dict[str, float]:
    """Extract 14 features comparing a resume experience profile to JD requirements.

    Feature names match inference (m4_exp_edu_comparator.py) exactly.

    Parameters
    ----------
    resume_exp : dict
        Keys may include: years_experience, title, education_level,
        education_field, domain, skills (list), companies (list of dicts
        with title, duration), num_roles (int).
    jd_req : dict
        Keys may include: required_years, title, education_level,
        education_field, domain, required_skills (list).

    Returns
    -------
    dict mapping feature name to float value.
    """
    features: dict[str, float] = {}

    # 1. years_gap: resume years - required years
    resume_years = float(resume_exp.get("years_experience", 0))
    required_years = float(jd_req.get("required_years", 0))
    features["years_gap"] = resume_years - required_years

    # 2. resume_years
    features["resume_years"] = resume_years

    # 3. required_years
    features["required_years"] = required_years

    # 4. title_cosine_sim
    resume_title = str(resume_exp.get("title", ""))
    jd_title = str(jd_req.get("title", ""))
    features["title_cosine_sim"] = _simple_cosine(resume_title, jd_title)

    # 5. edu_gap: ordinal difference in education levels
    resume_edu = _parse_education_level(str(resume_exp.get("education_level", "")))
    jd_edu = _parse_education_level(str(jd_req.get("education_level", "")))
    features["edu_gap"] = float(resume_edu - jd_edu)

    # 6. field_match: binary -- does education field match?
    resume_field = str(resume_exp.get("education_field", "")).lower()
    jd_field = str(jd_req.get("education_field", "")).lower()
    features["field_match"] = 1.0 if resume_field and jd_field and (
        resume_field in jd_field or jd_field in resume_field
    ) else 0.0

    # 7. num_roles: number of companies/roles
    companies = resume_exp.get("companies", [])
    n_roles = resume_exp.get("num_roles", len(companies))
    n_roles = max(int(n_roles), 1)
    features["num_roles"] = float(n_roles)

    # 8. avg_tenure_months: average tenure in months
    tenures = []
    for comp in companies:
        dur = comp.get("duration", comp.get("years", 0))
        if isinstance(dur, (int, float)) and dur > 0:
            tenures.append(float(dur))
    avg_tenure_years = float(np.mean(tenures)) if tenures else resume_years / max(n_roles, 1)
    features["avg_tenure_months"] = avg_tenure_years * 12.0

    # 9. has_leadership: binary leadership detection
    features["has_leadership"] = _has_leadership(resume_title)

    # 10. career_velocity: num_roles / resume_years
    features["career_velocity"] = float(n_roles) / max(resume_years, 1.0) if resume_years > 0 else 0.0

    # 11. domain_sim: cosine similarity between domain descriptions
    resume_domain = str(resume_exp.get("domain", ""))
    jd_domain = str(jd_req.get("domain", ""))
    features["domain_sim"] = _simple_cosine(resume_domain, jd_domain)

    # 12. num_skills: count of resume skills
    resume_skills = resume_exp.get("skills", [])
    features["num_skills"] = float(len(resume_skills))

    # 13. edu_level_ordinal: absolute resume education ordinal
    features["edu_level_ordinal"] = float(resume_edu)

    # 14. jd_edu_ordinal: absolute JD education ordinal
    features["jd_edu_ordinal"] = float(jd_edu)

    return features


def _build_pairs_from_jobhop(df: pd.DataFrame, max_pairs: int = 50000, seed: int = 42) -> list[dict[str, Any]]:
    """Synthesize resume-JD comparison pairs from JobHop trajectories.

    Enriches pairs with job classification data for education/domain variance.
    """
    rng = random.Random(seed)
    pairs: list[dict[str, Any]] = []

    if df.empty:
        return pairs

    job_class_df = _get_job_classification()

    # Group by person/resume identifier
    id_col = None
    for candidate in ["resume_id", "person_id", "id", "user_id"]:
        if candidate in df.columns:
            id_col = candidate
            break

    if id_col is None:
        logger.warning("JobHop: no person ID column found. Columns: %s", df.columns.tolist())
        title_col = None
        for c in df.columns:
            if "title" in c.lower() or "job" in c.lower():
                title_col = c
                break
        if title_col is None:
            return pairs

        rows = df.to_dict("records")
        rng.shuffle(rows)
        for i in range(0, min(len(rows) - 1, max_pairs * 2), 2):
            resume_row = rows[i]
            jd_row = rows[i + 1]

            resume_years = rng.uniform(1, 15)
            n_roles = rng.randint(1, 5)
            resume_exp = {
                "years_experience": resume_years,
                "title": str(resume_row.get(title_col, "")),
                "domain": str(resume_row.get("domain", resume_row.get("industry", ""))),
                "skills": [],
                "companies": [{"title": "", "duration": resume_years / n_roles} for _ in range(n_roles)],
                "num_roles": n_roles,
                "education_level": "",
                "education_field": "",
            }
            jd_req = {
                "required_years": rng.uniform(1, 10),
                "title": str(jd_row.get(title_col, "")),
                "domain": str(jd_row.get("domain", jd_row.get("industry", ""))),
                "required_skills": [],
                "education_level": "",
                "education_field": "",
            }

            _enrich_with_job_class(resume_exp, jd_req, job_class_df, rng)

            feats = extract_features(resume_exp, jd_req)
            label = feats["title_cosine_sim"]
            pairs.append({**feats, "label": label})

            if len(pairs) >= max_pairs:
                break

        return pairs

    # Group trajectories
    groups = df.groupby(id_col)
    group_keys = list(groups.groups.keys())
    rng.shuffle(group_keys)

    title_col = None
    for c in df.columns:
        if "title" in c.lower() or "job" in c.lower() or "position" in c.lower():
            title_col = c
            break

    for gid in group_keys:
        group = groups.get_group(gid).sort_index()
        if len(group) < 2:
            continue

        rows = group.to_dict("records")
        latest = rows[-1]
        prev = rows[:-1]

        resume_years = len(prev) * rng.uniform(1.5, 3.5)
        n_roles = len(prev)
        resume_exp = {
            "years_experience": resume_years,
            "title": str(prev[-1].get(title_col, "")) if title_col and prev else "",
            "domain": "",
            "skills": [],
            "companies": [{"title": str(r.get(title_col, "")), "duration": rng.uniform(1, 4)} for r in prev],
            "num_roles": n_roles,
            "education_level": "",
            "education_field": "",
        }

        jd_req = {
            "required_years": rng.uniform(2, resume_exp["years_experience"]),
            "title": str(latest.get(title_col, "")) if title_col else "",
            "domain": "",
            "required_skills": [],
            "education_level": "",
            "education_field": "",
        }

        _enrich_with_job_class(resume_exp, jd_req, job_class_df, rng)

        feats = extract_features(resume_exp, jd_req)
        label = 0.5 * feats["title_cosine_sim"] + 0.3 * min(feats["years_gap"] / 5.0, 1.0) + 0.2 * rng.random()
        label = max(0.0, min(1.0, label))
        pairs.append({**feats, "label": label})

        if len(pairs) >= max_pairs:
            break

    return pairs


def _build_pairs_from_karrierewege(df: pd.DataFrame, max_pairs: int = 30000, seed: int = 42) -> list[dict[str, Any]]:
    """Synthesize feature vectors from Karrierewege career paths."""
    rng = random.Random(seed)
    pairs: list[dict[str, Any]] = []

    if df.empty:
        return pairs

    job_class_df = _get_job_classification()

    title_col = None
    skill_col = None
    for c in df.columns:
        cl = c.lower()
        if "title" in cl or "job" in cl or "position" in cl or "occupation" in cl:
            title_col = c
        elif "skill" in cl or "competenc" in cl:
            skill_col = c

    rows = df.to_dict("records")
    rng.shuffle(rows)

    for i in range(0, min(len(rows) - 1, max_pairs * 2), 2):
        r1 = rows[i]
        r2 = rows[i + 1]

        skills_1 = []
        if skill_col and r1.get(skill_col):
            val = r1[skill_col]
            if isinstance(val, str):
                skills_1 = [s.strip() for s in val.split(",") if s.strip()]
            elif isinstance(val, list):
                skills_1 = [str(s) for s in val]

        skills_2 = []
        if skill_col and r2.get(skill_col):
            val = r2[skill_col]
            if isinstance(val, str):
                skills_2 = [s.strip() for s in val.split(",") if s.strip()]
            elif isinstance(val, list):
                skills_2 = [str(s) for s in val]

        resume_years = rng.uniform(2, 12)
        n_roles = rng.randint(1, 5)
        resume_exp = {
            "years_experience": resume_years,
            "title": str(r1.get(title_col, "")) if title_col else "",
            "skills": skills_1,
            "companies": [{"title": "", "duration": resume_years / n_roles} for _ in range(n_roles)],
            "num_roles": n_roles,
            "education_level": "",
            "education_field": "",
            "domain": "",
        }

        jd_req = {
            "required_years": rng.uniform(1, 8),
            "title": str(r2.get(title_col, "")) if title_col else "",
            "required_skills": skills_2,
            "education_level": "",
            "education_field": "",
            "domain": "",
        }

        _enrich_with_job_class(resume_exp, jd_req, job_class_df, rng)

        feats = extract_features(resume_exp, jd_req)
        label = (
            0.4 * feats["title_cosine_sim"]
            + 0.3 * min(feats["num_skills"] / 10.0, 1.0)
            + 0.2 * min(feats["years_gap"] / 5.0, 1.0)
            + 0.1 * rng.random()
        )
        label = max(0.0, min(1.0, label))
        pairs.append({**feats, "label": label})

        if len(pairs) >= max_pairs:
            break

    return pairs


def _build_pairs_from_classification(max_pairs: int = 20000, seed: int = 42) -> list[dict[str, Any]]:
    """Generate combinatorial pairs from job classification data.

    Creates pairs where one job class acts as "resume" and another as "JD",
    giving real education/domain variance. Expands with Gaussian noise.
    """
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)
    pairs: list[dict[str, Any]] = []

    job_class_df = _get_job_classification()
    if job_class_df.empty:
        logger.warning("No job classification data – skipping classification pairs.")
        return pairs

    rows = job_class_df.to_dict("records")
    n_classes = len(rows)

    # Generate all combinatorial pairs (66 * 65 = 4290 base pairs)
    base_pairs: list[tuple[dict, dict]] = []
    for i in range(n_classes):
        for j in range(n_classes):
            if i != j:
                base_pairs.append((rows[i], rows[j]))

    rng.shuffle(base_pairs)

    # Determine how many noise variations per base pair
    variations_per_pair = max(1, max_pairs // len(base_pairs))

    for resume_class, jd_class in base_pairs:
        for _ in range(variations_per_pair):
            # Resume side
            resume_edu_raw = int(resume_class["EducationLevel"])
            resume_edu_name = _JOB_CLASS_EDU_NAME.get(resume_edu_raw, "bachelor")
            resume_family = str(resume_class["JobFamilyDescription"]).strip()
            resume_field = _FAMILY_TO_FIELD.get(resume_family.lower(), resume_family.lower())
            resume_experience = float(resume_class["Experience"]) + np_rng.normal(0, 1.5)
            resume_experience = max(0.5, resume_experience)
            n_roles = max(1, int(resume_experience / rng.uniform(1.5, 3.5)))
            n_skills = rng.randint(3, 15)

            resume_exp = {
                "years_experience": resume_experience,
                "title": str(resume_class["JobClassDescription"]).strip(),
                "education_level": resume_edu_name,
                "education_field": resume_field,
                "domain": resume_family,
                "skills": [f"skill_{k}" for k in range(n_skills)],
                "companies": [
                    {"title": str(resume_class["JobClassDescription"]), "duration": resume_experience / n_roles}
                    for _ in range(n_roles)
                ],
                "num_roles": n_roles,
            }

            # JD side
            jd_edu_raw = int(jd_class["EducationLevel"])
            jd_edu_name = _JOB_CLASS_EDU_NAME.get(jd_edu_raw, "bachelor")
            jd_family = str(jd_class["JobFamilyDescription"]).strip()
            jd_field = _FAMILY_TO_FIELD.get(jd_family.lower(), jd_family.lower())
            jd_experience = float(jd_class["Experience"]) + np_rng.normal(0, 1.0)
            jd_experience = max(0.0, jd_experience)

            jd_req = {
                "required_years": jd_experience,
                "title": str(jd_class["JobClassDescription"]).strip(),
                "education_level": jd_edu_name,
                "education_field": jd_field,
                "domain": jd_family,
                "required_skills": [],
            }

            feats = extract_features(resume_exp, jd_req)

            # Label: composite match quality
            label = (
                0.3 * feats["title_cosine_sim"]
                + 0.2 * max(0.0, min(1.0, (feats["years_gap"] + 3) / 6.0))
                + 0.2 * feats["field_match"]
                + 0.15 * feats["domain_sim"]
                + 0.15 * max(0.0, min(1.0, (feats["edu_gap"] + 2) / 4.0))
            )
            label = max(0.0, min(1.0, label + np_rng.normal(0, 0.03)))
            pairs.append({**feats, "label": label})

            if len(pairs) >= max_pairs:
                break
        if len(pairs) >= max_pairs:
            break

    logger.info("Classification pairs generated: %d", len(pairs))
    return pairs


# ---------------------------------------------------------------------------
# Dataset Construction
# ---------------------------------------------------------------------------


def build_dataset(pairs: list[dict[str, Any]], seed: int = 42) -> DatasetDict:
    """Convert feature-vector pairs into train/val/test DatasetDict."""
    rng = random.Random(seed)
    rng.shuffle(pairs)

    n = len(pairs)
    n_train = int(n * 0.80)
    n_val = int(n * 0.10)

    splits = {
        "train": pairs[:n_train],
        "validation": pairs[n_train : n_train + n_val],
        "test": pairs[n_train + n_val :],
    }

    feature_dict: dict[str, Any] = {name: Value("float64") for name in FEATURE_NAMES}
    feature_dict["label"] = Value("float64")
    features = Features(feature_dict)

    dd = DatasetDict()
    for name, data in splits.items():
        col_data: dict[str, list] = {fname: [] for fname in FEATURE_NAMES}
        col_data["label"] = []
        for row in data:
            for fname in FEATURE_NAMES:
                col_data[fname].append(float(row.get(fname, 0.0)))
            col_data["label"].append(float(row.get("label", 0.0)))

        dd[name] = Dataset.from_dict(col_data, features=features)
        logger.info("Split '%s': %d examples.", name, len(data))

    return dd


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run full M4 data pipeline: load, extract features, build dataset, save."""
    logger.info("=" * 60)
    logger.info("M4 Data Preparation: Exp/Edu Feature Vectors for LightGBM")
    logger.info("DATA_DIR: %s", DATA_DIR)
    logger.info("OUTPUT_DIR: %s", OUTPUT_DIR)
    logger.info("Features (%d): %s", len(FEATURE_NAMES), FEATURE_NAMES)
    logger.info("=" * 60)

    jobhop_df = load_jobhop()
    karrierewege_df = load_karrierewege()

    all_pairs: list[dict[str, Any]] = []

    if not jobhop_df.empty:
        jh_pairs = _build_pairs_from_jobhop(jobhop_df)
        all_pairs.extend(jh_pairs)
        logger.info("JobHop contributed %d pairs.", len(jh_pairs))

    if not karrierewege_df.empty:
        kw_pairs = _build_pairs_from_karrierewege(karrierewege_df)
        all_pairs.extend(kw_pairs)
        logger.info("Karrierewege contributed %d pairs.", len(kw_pairs))

    # Always generate classification pairs (independent of other data)
    cls_pairs = _build_pairs_from_classification()
    all_pairs.extend(cls_pairs)
    logger.info("Classification contributed %d pairs.", len(cls_pairs))

    if not all_pairs:
        logger.error("No feature vectors produced -- aborting.")
        return

    logger.info("Total feature vectors: %d", len(all_pairs))

    dataset = build_dataset(all_pairs)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(OUTPUT_DIR))
    logger.info("Dataset saved to %s", OUTPUT_DIR)

    # Save feature metadata
    meta_path = OUTPUT_DIR / "feature_metadata.json"
    with open(meta_path, "w") as f:
        json.dump({
            "feature_names": FEATURE_NAMES,
            "num_features": len(FEATURE_NAMES),
            "edu_levels": EDU_LEVELS,
        }, f, indent=2)
    logger.info("Feature metadata saved to %s", meta_path)

    # Summary statistics
    for split_name in dataset:
        ds = dataset[split_name]
        labels = [row["label"] for row in ds]
        logger.info(
            "  %s: %d examples, label mean=%.3f, std=%.3f, min=%.3f, max=%.3f",
            split_name,
            len(ds),
            np.mean(labels),
            np.std(labels),
            np.min(labels),
            np.max(labels),
        )


if __name__ == "__main__":
    main()
