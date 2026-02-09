"""Evaluation: Kaggle Resume.csv category validation.

Uses Resume.csv with Category column (HR, IT, Engineering, etc.)
and Resume_str text. Scores resumes against matching and non-matching
reference JDs to validate category discrimination.

Run with: pytest -m evaluation -v -s tests/test_kaggle_category_eval.py
"""

import csv
from collections import defaultdict
from pathlib import Path

import pytest

pytestmark = pytest.mark.evaluation

KAGGLE_CSV = Path(__file__).resolve().parents[2] / "Research" / "kaggle" / "Resume.csv"

# Reference JDs for category validation
# NOTE: Category names must match Resume.csv exactly (uppercase with hyphens)
CATEGORY_JDS = {
    "INFORMATION-TECHNOLOGY": """Senior IT Specialist

Requirements:
- 5+ years of experience in IT infrastructure and support
- Strong knowledge of networking (TCP/IP, DNS, DHCP, VPN)
- Experience with Windows Server, Active Directory, and Group Policy
- Experience with Linux/Unix system administration
- Knowledge of cloud platforms (AWS, Azure, or GCP)
- Experience with virtualization (VMware, Hyper-V)
- Familiarity with IT security best practices and compliance
- Strong troubleshooting and problem-solving skills
- ITIL framework knowledge
- Bachelor's degree in Information Technology or related field
""",
    "HR": """Senior Human Resources Manager

Requirements:
- 5+ years of HR management experience
- Strong knowledge of employment law and labor regulations
- Experience with HRIS systems (Workday, SAP SuccessFactors, or ADP)
- Expertise in talent acquisition, onboarding, and retention strategies
- Experience with performance management and employee relations
- Knowledge of compensation and benefits administration
- Strong interpersonal and communication skills
- Experience with organizational development and change management
- PHR, SPHR, or SHRM-CP certification preferred
- Bachelor's degree in Human Resources or related field
""",
    "ENGINEERING": """Mechanical Engineer

Requirements:
- 5+ years of engineering experience
- Proficiency in CAD/CAM software (AutoCAD, SolidWorks, or CATIA)
- Experience with FEA/CFD analysis tools
- Knowledge of manufacturing processes and materials science
- Experience with project management and cross-functional teams
- Strong analytical and problem-solving abilities
- Knowledge of GD&T and engineering standards
- Experience with product lifecycle management
- Professional Engineer (PE) license preferred
- Bachelor's degree in Mechanical or related Engineering
""",
    "SALES": """Senior Sales Manager

Requirements:
- 5+ years of B2B sales experience
- Proven track record of meeting and exceeding sales quotas
- Experience with CRM systems (Salesforce, HubSpot)
- Strong negotiation and closing skills
- Experience with sales pipeline management and forecasting
- Knowledge of consultative and solution-based selling
- Ability to develop and maintain client relationships
- Experience with territory management and account planning
- Strong presentation and communication skills
- Bachelor's degree in Business or related field
""",
}

RESUMES_PER_CATEGORY = 10


def _load_kaggle_resumes() -> dict[str, list[str]]:
    """Load resumes grouped by category from Resume.csv."""
    if not KAGGLE_CSV.exists():
        return {}

    resumes: dict[str, list[str]] = defaultdict(list)
    with open(KAGGLE_CSV, encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            category = row.get("Category", "").strip()
            text = row.get("Resume_str", "").strip()
            if category and text and len(text) > 100:
                resumes[category].append(text)

    return dict(resumes)


@pytest.fixture(scope="module")
def kaggle_resumes():
    resumes = _load_kaggle_resumes()
    if not resumes:
        pytest.skip("Resume.csv not found or empty")
    return resumes


@pytest.mark.asyncio
async def test_same_category_scores_higher(kaggle_resumes):
    """Same-category resumes should score higher than cross-category on average."""
    from services.resume_analyzer import analyze

    results = {}

    for category, jd in CATEGORY_JDS.items():
        if category not in kaggle_resumes:
            print(f"  Skipping {category}: not in dataset")
            continue

        same_resumes = kaggle_resumes[category][:RESUMES_PER_CATEGORY]
        same_scores = []
        for text in same_resumes:
            try:
                result = await analyze(text, jd)
                same_scores.append(result.overall_score)
            except Exception:
                pass

        # Pick a different category for cross-comparison
        cross_categories = [c for c in kaggle_resumes if c != category and c in CATEGORY_JDS]
        cross_scores = []
        if cross_categories:
            cross_cat = cross_categories[0]
            cross_resumes = kaggle_resumes[cross_cat][:RESUMES_PER_CATEGORY]
            for text in cross_resumes:
                try:
                    result = await analyze(text, jd)
                    cross_scores.append(result.overall_score)
                except Exception:
                    pass

        same_avg = sum(same_scores) / len(same_scores) if same_scores else 0
        cross_avg = sum(cross_scores) / len(cross_scores) if cross_scores else 0
        results[category] = {
            "same_avg": same_avg,
            "cross_avg": cross_avg,
            "cross_cat": cross_categories[0] if cross_categories else "N/A",
            "n_same": len(same_scores),
            "n_cross": len(cross_scores),
        }

    # Report
    print(f"\n{'='*70}")
    print(f"Kaggle Category Validation")
    print(f"{'='*70}")
    print(f"{'Category':<25} {'Same Avg':>10} {'Cross Avg':>10} {'Cross Cat':<20} {'Delta':>8}")
    print(f"{'-'*70}")

    wins = 0
    total = 0
    for cat, r in results.items():
        delta = r["same_avg"] - r["cross_avg"]
        print(f"{cat:<25} {r['same_avg']:>10.1f} {r['cross_avg']:>10.1f} {r['cross_cat']:<20} {delta:>+8.1f}")
        if r["n_same"] > 0 and r["n_cross"] > 0:
            total += 1
            if r["same_avg"] > r["cross_avg"]:
                wins += 1

    print(f"{'='*70}")
    print(f"Category wins: {wins}/{total}")

    assert wins > 0, "At least one category should score higher on same-category JD"
    if total >= 2:
        assert wins >= total // 2, f"Only {wins}/{total} categories scored higher on matching JD"
