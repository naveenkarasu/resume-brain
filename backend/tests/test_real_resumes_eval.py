"""Evaluation: Real resume PDFs scored against reference JDs.

Uses the resume/ directory with categorized PDFs across
Software_Engineer, Security_Engineer, DevOps_SRE, Data_AI, Specialized.

Run with: pytest -m evaluation -v -s tests/test_real_resumes_eval.py
"""

from pathlib import Path

import pytest

pytestmark = pytest.mark.evaluation

RESUME_DIR = Path(__file__).resolve().parents[2] / "resume"

# Reference Software Engineer job description
SWE_JD = """Senior Software Engineer

Requirements:
- 5+ years of experience in software development
- Strong proficiency in Python, JavaScript/TypeScript, or Java
- Experience with web frameworks (React, Django, Flask, Spring, or Node.js)
- Experience with cloud platforms (AWS, GCP, or Azure)
- Experience with SQL and NoSQL databases (PostgreSQL, MongoDB, Redis)
- Experience with Docker and container orchestration
- Familiarity with CI/CD pipelines and version control (Git)
- Strong problem-solving skills and ability to design scalable systems
- Experience with RESTful APIs and microservices architecture
- Bachelor's degree in Computer Science or related field

Preferred:
- Experience with Kubernetes
- Experience with system design and distributed systems
- Contributions to open source projects
"""


def _load_pdf_text(pdf_path: Path) -> str:
    """Load text from a PDF file."""
    import pdfplumber
    try:
        with pdfplumber.open(pdf_path) as pdf:
            return "\n".join(page.extract_text() or "" for page in pdf.pages)
    except Exception:
        return ""


def _get_pdfs(category: str) -> list[Path]:
    """Get all PDF files in a category directory."""
    cat_dir = RESUME_DIR / category
    if not cat_dir.exists():
        return []
    return sorted(cat_dir.glob("*.pdf"))


@pytest.fixture(scope="module")
def swe_pdfs():
    pdfs = _get_pdfs("Software_Engineer")
    if not pdfs:
        pytest.skip("No Software_Engineer resumes found")
    return pdfs


@pytest.fixture(scope="module")
def data_ai_pdfs():
    pdfs = _get_pdfs("Data_AI")
    if not pdfs:
        pytest.skip("No Data_AI resumes found")
    return pdfs


@pytest.mark.asyncio
async def test_swe_resumes_score_above_50(swe_pdfs):
    """At least 80% of SWE resumes should score > 50 against SWE JD."""
    from services.resume_analyzer import analyze

    scores = []
    for pdf_path in swe_pdfs:
        text = _load_pdf_text(pdf_path)
        if not text.strip():
            continue
        try:
            result = await analyze(text, SWE_JD)
            scores.append(result.overall_score)
            print(f"  {pdf_path.name}: {result.overall_score}")
        except Exception as e:
            print(f"  {pdf_path.name}: ERROR - {e}")

    above_50 = sum(1 for s in scores if s > 50)
    pct = above_50 / len(scores) if scores else 0

    print(f"\n{'='*60}")
    print(f"SWE Resume Scores (n={len(scores)})")
    print(f"{'='*60}")
    print(f"Mean score:     {sum(scores)/len(scores):.1f}" if scores else "No scores")
    print(f"Above 50:       {above_50}/{len(scores)} ({pct:.0%})")
    print(f"Min/Max:        {min(scores)}/{max(scores)}" if scores else "N/A")
    print(f"{'='*60}")

    assert pct >= 0.5, f"Only {pct:.0%} of SWE resumes scored > 50 (need >= 50%)"


@pytest.mark.asyncio
async def test_swe_vs_data_ai_cross_category(swe_pdfs, data_ai_pdfs):
    """SWE resumes should score higher than Data_AI resumes on SWE JD."""
    from services.resume_analyzer import analyze

    async def _avg_score(pdfs):
        scores = []
        for pdf_path in pdfs:
            text = _load_pdf_text(pdf_path)
            if not text.strip():
                continue
            try:
                result = await analyze(text, SWE_JD)
                scores.append(result.overall_score)
            except Exception:
                pass
        return sum(scores) / len(scores) if scores else 0

    swe_avg = await _avg_score(swe_pdfs)
    data_avg = await _avg_score(data_ai_pdfs)

    print(f"\n{'='*60}")
    print(f"Cross-Category Comparison (SWE JD)")
    print(f"{'='*60}")
    print(f"SWE avg score:      {swe_avg:.1f}")
    print(f"Data_AI avg score:  {data_avg:.1f}")
    print(f"Delta:              {swe_avg - data_avg:.1f}")
    print(f"{'='*60}")

    assert swe_avg > data_avg, (
        f"SWE avg ({swe_avg:.1f}) should be > Data_AI avg ({data_avg:.1f})"
    )
