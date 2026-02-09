"""Evaluation: Vacancy-Resume ranking against human annotator judgments.

Uses the vacancy-resume dataset with 30 CVs, 5 vacancies, and
two independent annotator rankings.

Run with: pytest -m evaluation -v -s tests/test_vacancy_resume_eval.py
"""

import ast
import csv
import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.evaluation

DATA_DIR = Path(__file__).resolve().parents[2] / "Research" / "data" / "vacancy-resume"
CV_DIR = DATA_DIR / "CV"
VACANCIES_CSV = DATA_DIR / "5_vacancies.csv"
ANNOTATIONS_FILE = DATA_DIR / "annotations-for-the-first-30-vacancies.txt"


def _load_vacancies() -> dict[int, str]:
    """Load vacancy texts from CSV. Returns {1-based index: jd_text}."""
    vacancies = {}
    with open(VACANCIES_CSV, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, start=1):
            vacancies[i] = row["job_description"]
    return vacancies


def _load_annotations() -> tuple[list[list[int]], list[list[int]]]:
    """Parse annotator rankings from the annotations file."""
    text = ANNOTATIONS_FILE.read_text()

    # Extract ANNOTATOR_1_RANKINGS
    a1_start = text.index("ANNOTATOR_1_RANKINGS=") + len("ANNOTATOR_1_RANKINGS=")
    a1_end = text.index("ANNOTATOR_2_RANKINGS=")
    a1_str = text[a1_start:a1_end].strip().rstrip()
    # Clean up comments at line ends
    import re
    a1_str = re.sub(r"#.*", "", a1_str).strip()
    a1 = ast.literal_eval(a1_str)

    # Extract ANNOTATOR_2_RANKINGS
    a2_start = text.index("ANNOTATOR_2_RANKINGS=") + len("ANNOTATOR_2_RANKINGS=")
    a2_str = text[a2_start:].strip()
    a2_str = re.sub(r"#.*", "", a2_str).strip()
    a2 = ast.literal_eval(a2_str)

    return a1, a2


def _load_cv_text(cv_num: int) -> str:
    """Load CV text from .docx file."""
    from services.pdf_parser import extract_text_docx
    docx_path = CV_DIR / f"{cv_num}.docx"
    if not docx_path.exists():
        return ""
    return extract_text_docx(docx_path.read_bytes())


@pytest.fixture(scope="module")
def vacancies():
    if not VACANCIES_CSV.exists():
        pytest.skip("Vacancy-resume dataset not found")
    return _load_vacancies()


@pytest.fixture(scope="module")
def annotations():
    if not ANNOTATIONS_FILE.exists():
        pytest.skip("Annotations file not found")
    return _load_annotations()


@pytest.mark.asyncio
async def test_vacancy_resume_ranking(vacancies, annotations):
    """Score 30 CVs x 5 vacancies and compare with human rankings."""
    from scipy.stats import spearmanr
    from services.resume_analyzer import analyze

    a1_rankings, a2_rankings = annotations
    num_cvs = min(30, len(a1_rankings))
    num_vacancies = len(vacancies)

    spearman_scores = []
    ndcg_scores = []

    for cv_idx in range(num_cvs):
        cv_num = cv_idx + 1
        cv_text = _load_cv_text(cv_num)
        if not cv_text.strip():
            continue

        # Score this CV against all 5 vacancies
        system_scores = []
        for vac_idx in range(1, num_vacancies + 1):
            try:
                result = await analyze(cv_text, vacancies[vac_idx])
                system_scores.append(result.overall_score)
            except Exception:
                system_scores.append(0)

        if len(system_scores) != 5:
            continue

        # Human rankings (1=best, 5=worst) — convert to scores (higher=better)
        human_ranks_a1 = a1_rankings[cv_idx]
        # Convert rank to score: rank 1 -> score 5, rank 5 -> score 1
        human_scores_a1 = [6 - r for r in human_ranks_a1]

        # Spearman correlation
        corr, _ = spearmanr(system_scores, human_scores_a1)
        if not (corr != corr):  # skip NaN
            spearman_scores.append(corr)

        # NDCG@5
        ndcg = _compute_ndcg(system_scores, human_scores_a1, k=5)
        ndcg_scores.append(ndcg)

    # Report metrics
    avg_spearman = sum(spearman_scores) / len(spearman_scores) if spearman_scores else 0
    avg_ndcg = sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0

    # Inter-annotator agreement (ceiling metric)
    iaa_scores = []
    for cv_idx in range(num_cvs):
        human_a1 = [6 - r for r in a1_rankings[cv_idx]]
        human_a2 = [6 - r for r in a2_rankings[cv_idx]]
        corr, _ = spearmanr(human_a1, human_a2)
        if not (corr != corr):
            iaa_scores.append(corr)
    avg_iaa = sum(iaa_scores) / len(iaa_scores) if iaa_scores else 0

    print(f"\n{'='*60}")
    print(f"Vacancy-Resume Ranking Evaluation")
    print(f"{'='*60}")
    print(f"CVs evaluated:              {len(spearman_scores)}")
    print(f"Avg Spearman correlation:   {avg_spearman:.4f}")
    print(f"Avg NDCG@5:                 {avg_ndcg:.4f}")
    print(f"Inter-annotator agreement:  {avg_iaa:.4f}")
    print(f"{'='*60}")

    assert avg_spearman > 0.0, f"Spearman {avg_spearman:.4f} should be > 0"
    assert avg_ndcg > 0.5, f"NDCG@5 {avg_ndcg:.4f} should be > 0.5"


def _compute_ndcg(predicted_scores: list[float], relevance_scores: list[float], k: int = 5) -> float:
    """Compute NDCG@k.

    predicted_scores: system scores (higher = better match)
    relevance_scores: ground truth relevance (higher = better)
    """
    import math

    # Rank items by predicted scores (descending)
    ranked_indices = sorted(range(len(predicted_scores)), key=lambda i: -predicted_scores[i])

    # DCG@k
    dcg = 0.0
    for i, idx in enumerate(ranked_indices[:k]):
        dcg += relevance_scores[idx] / math.log2(i + 2)

    # Ideal DCG@k (sort by relevance descending)
    ideal_indices = sorted(range(len(relevance_scores)), key=lambda i: -relevance_scores[i])
    idcg = 0.0
    for i, idx in enumerate(ideal_indices[:k]):
        idcg += relevance_scores[idx] / math.log2(i + 2)

    return dcg / idcg if idcg > 0 else 0.0
