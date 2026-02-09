# Resume Brain - Accuracy Improvement Plan

## Current State

The pipeline has 7 layers: section parsing → TF-IDF cosine → SBERT semantic → keyword extraction → matching → Gemini LLM → score blending. It works, but has critical accuracy gaps in keyword matching, synonym handling, experience scoring, and model quality.

## Critical Gaps Identified

| Gap | Severity | Example Failure |
|-----|----------|-----------------|
| Binary keyword matching (exact only) | CRITICAL | "Kubernetes" vs "K8s" → MISS |
| No synonym awareness | CRITICAL | "Machine Learning" vs "ML" → MISS |
| Weak SBERT model (22M params) | HIGH | Can't distinguish "React" from "React Native" |
| `experience_match` is fake (just semantic score) | HIGH | No actual years-of-experience comparison |
| Whole-document similarity (not section-level) | MEDIUM | Strong skills mask weak experience |
| No bullet quality scoring | MEDIUM | Bullets sent to LLM without local quality check |
| Normalization loses "C++", "C#" edge cases | MEDIUM | `re.sub` strips special chars from language names |

---

## Phase 1: Quick Wins (< 1 day)

### 1.1 Upgrade SBERT Model
**File:** `services/similarity.py` line 22
**Change:** Replace `all-MiniLM-L6-v2` (22M params, 384-dim) with `all-mpnet-base-v2` (109M params, 768-dim)
**Why:** 5-10% better semantic similarity benchmarks. One-line change.
**Trade-off:** Model is ~420MB vs ~80MB, 3-5x slower inference. Still fast enough for single-request analysis.
**Test:** Existing `test_similarity.py` tests should still pass. Semantic scores will change slightly.

### 1.2 Add Fuzzy Keyword Matching (RapidFuzz)
**File:** `services/keyword_extractor.py` → `match_keywords()` function
**New dependency:** `rapidfuzz` in `requirements.txt`
**Change:** Replace exact `kw in resume_terms` with fuzzy matching using Levenshtein distance (threshold ~80):
```python
from rapidfuzz import fuzz
# For each keyword, check if any resume term scores > 80
if any(fuzz.ratio(kw.lower(), term) > 80 for term in resume_terms):
    matched.append(kw)
```
**Why:** Catches typos, partial matches ("Postgres" → "PostgreSQL"), abbreviation variants. Estimated 15-25% fewer false negatives.
**Test:** Update `test_keyword_extractor.py` — add tests for fuzzy matches ("Postgres" matching "PostgreSQL", "K8s" matching "Kubernetes" via synonym layer).

### 1.3 Add Skill Synonym Dictionary
**File:** `services/keyword_extractor.py` — new `SKILL_SYNONYMS` dict + `_canonicalize()` function
**Change:** Add a ~200-entry mapping of aliases → canonical forms:
```python
SKILL_SYNONYMS = {
    "js": "javascript", "ts": "typescript", "k8s": "kubernetes",
    "postgres": "postgresql", "react.js": "react", "reactjs": "react",
    "node": "node.js", "nodejs": "node.js", "gcp": "google cloud platform",
    "ml": "machine learning", "dl": "deep learning", "tf": "tensorflow",
    "py": "python", "rb": "ruby", "c#": "csharp", "c++": "cpp",
    # ... ~200 entries covering common tech aliases
}
```
Apply canonicalization BEFORE matching: both resume terms and JD keywords get normalized to canonical forms.
**Why:** Eliminates the entire class of "same skill, different name" false negatives.
**Test:** Add tests for synonym resolution ("ML" → matches "machine learning", "K8s" → matches "kubernetes").

---

## Phase 2: High-Value Enhancements (1-2 days)

### 2.1 Section-Level SBERT Matching
**Files:** `services/resume_analyzer.py`, `services/similarity.py`
**Change:** Instead of one `hybrid_similarity(resume_text, job_description)`, compute per-section similarity:
- `skills_sim = sbert_cosine_similarity(sections["skills"], jd_text)`
- `experience_sim = sbert_cosine_similarity(sections["experience"], jd_text)`
- `education_sim = sbert_cosine_similarity(sections["education"], jd_text)`

Map these to `score_breakdown` categories directly:
- `skills_match` ← skills section similarity
- `experience_match` ← experience section similarity
- `education_match` ← education section similarity
- `keywords_match` ← keyword overlap ratio

**Why:** Current whole-document similarity dilutes signal. Section-level matching gives 3-4 separate scores that map meaningfully to breakdown categories. Resume2Vec paper showed 15.85% improvement in ranking accuracy.
**Test:** Add tests verifying section-level scores differ from whole-document scores.

### 2.2 Years-of-Experience Extraction
**File:** `services/section_parser.py` — new `extract_experience_years()` function
**Change:** Add regex patterns to extract:
1. Explicit claims: "5+ years of experience", "3 years managing teams"
2. Date ranges: "Jan 2019 - Present", "2020 - 2023"
3. Compute total tenure from date ranges using `dateutil.relativedelta`

```python
EXP_YEARS_RE = re.compile(r'(\d+)\+?\s*(?:years?|yrs?)\s*(?:of\s+)?(?:experience|exp)', re.I)
DATE_RANGE_RE = re.compile(
    r'((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\.?\s*\d{4})'
    r'\s*[-–to]+\s*'
    r'((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\.?\s*\d{4}|[Pp]resent)',
    re.I
)
```

**Files also changed:** `models/responses.py` (add `experience_years: float`), `resume_analyzer.py` (use extracted years for `experience_match`), frontend types
**Why:** Currently `experience_match = semantic_score` — not measuring experience at all. This makes it a real factual comparison.
**Test:** Test extraction from sample resumes with various date formats.

### 2.3 Bullet Quality Scoring
**File:** `services/pdf_parser.py` — new `score_bullet()` and `score_bullets()` functions
**Change:** Score each bullet on a rubric:
- Starts with strong action verb (from curated set of ~100 verbs)
- Contains quantified metrics (regex: `\d+[%$KMB]?|\$\d+`)
- Appropriate length (15-30 words)
- Contains JD-relevant keywords

```python
ACTION_VERBS = {"led", "built", "developed", "designed", "implemented", "optimized",
                "architected", "deployed", "managed", "increased", "reduced", ...}

def score_bullet(bullet: str, jd_keywords: set[str] = set()) -> dict:
    words = bullet.split()
    return {
        "has_action_verb": words[0].lower() in ACTION_VERBS if words else False,
        "has_metrics": bool(re.search(r'\d+[%$KMB]?|\$\d+', bullet)),
        "length_ok": 15 <= len(words) <= 30,
        "keyword_count": sum(1 for kw in jd_keywords if kw.lower() in bullet.lower()),
        "quality_score": <weighted average 0-100>,
    }
```

**Files also changed:** `models/responses.py` (add `bullet_scores` to response), `resume_analyzer.py` (call scoring), frontend `BulletRewritesCard`
**Why:** Gives users immediate, actionable feedback on bullet quality without waiting for LLM. Also provides LLM with quality context.
**Test:** Test scoring on sample bullets with/without action verbs, metrics, etc.

---

## Phase 3: Structural Improvements (2-3 days)

### 3.1 Expanded Section Scoring with Weights
**File:** `services/section_parser.py`
**Change:** Expand `EXPECTED_SECTIONS` from 3 to 8 sections with ATS-weighted importance:

| Section | Weight | Points |
|---------|--------|--------|
| Experience | 20 | Required |
| Skills | 15 | Required |
| Education | 12 | Required |
| Projects | 12 | Recommended |
| Summary/Objective | 10 | Recommended |
| Certifications | 8 | Optional |
| Achievements | 5 | Optional |

Completeness = weighted sum of present sections / total weight
**Why:** Current scoring only checks 3 sections with equal weight (0.33 each). This gives finer granularity.

### 3.2 Lemmatization in Keyword Matching
**File:** `services/keyword_extractor.py`
**New dependency:** `nltk` in `requirements.txt` (WordNetLemmatizer)
**Change:** Add lemmatization to `_normalize()` so "developing", "developed", "developer" all match "develop":
```python
from nltk.stem import WordNetLemmatizer
_lemmatizer = WordNetLemmatizer()
def _normalize(text: str) -> str:
    # ... existing normalization ...
    words = normalized.split()
    return " ".join(_lemmatizer.lemmatize(w) for w in words)
```
**Why:** Increases recall for verb/noun form variations without false positives.

### 3.3 Education Level Detection
**File:** `services/section_parser.py` — new `extract_education_level()` function
**Change:** Parse degree abbreviations from the education section:
```python
DEGREE_PATTERNS = {
    "phd": ["ph.d", "phd", "doctorate", "doctoral"],
    "masters": ["m.s.", "ms", "m.sc", "msc", "m.e.", "me", "mtech", "m.tech", "mba", "ma"],
    "bachelors": ["b.s.", "bs", "b.sc", "bsc", "b.e.", "be", "btech", "b.tech", "ba", "bba"],
    "associate": ["a.s.", "as", "a.a.", "aa", "associate"],
}
```
Compare detected level against JD education requirements.
**Files also changed:** `models/responses.py` (add `education_level: str`), `resume_analyzer.py`
**Why:** Makes `education_match` a factual comparison instead of just section-completeness proxy.

### 3.4 Enhanced PDF Extraction
**File:** `services/pdf_parser.py`
**Changes:**
1. Add DOCX support via `python-docx` (~30 min)
2. Add more bullet markers (unicode: `◆`, `⚫`, `→`, `▸`)
3. Better multi-digit numbered bullets (`10.`, `12)`)
4. Handle two-digit line numbers

**New dependency:** `python-docx` in `requirements.txt`

---

## Phase 4: Advanced (Future Sprint)

### 4.1 spaCy EntityRuler for Structured Skill Extraction
**Effort:** 4-8 hours
Add a spaCy EntityRuler with a JSONL patterns file of 500+ skills. Extract structured entities (SKILL, DEGREE, COMPANY, JOB_TITLE) from resume text. Enables richer comparison than keyword-level matching.

### 4.2 O*NET/ESCO Taxonomy Integration
**Effort:** 2-4 days
Map extracted skills to hierarchical taxonomy where "React" belongs to "Frontend Development" → "Software Engineering". Enables conceptual matching: "Vue.js experience" partially satisfies "frontend development" requirement.

### 4.3 Domain-Specific SBERT Fine-Tuning
**Effort:** 3-5 days
Fine-tune `all-mpnet-base-v2` on resume-vacancy pairs (conSultantBERT approach). Requires accumulating training data. Would give the best possible semantic matching quality.

---

## Dependency Summary

| Phase | New Dependencies | Size Impact |
|-------|-----------------|-------------|
| Phase 1 | `rapidfuzz` | ~2MB |
| Phase 2 | `python-dateutil` (likely already installed) | ~0 |
| Phase 3 | `nltk` (WordNetLemmatizer + data), `python-docx` | ~50MB |
| Phase 4 | `spacy` + `en_core_web_sm` model | ~50MB |

## Files Changed Per Phase

| Phase | New Files | Modified Files |
|-------|-----------|---------------|
| **Phase 1** | — | `similarity.py`, `keyword_extractor.py`, `requirements.txt`, `test_keyword_extractor.py` |
| **Phase 2** | — | `section_parser.py`, `pdf_parser.py`, `resume_analyzer.py`, `responses.py`, `test_section_parser.py`, `test_api.py`, frontend types + components |
| **Phase 3** | — | `section_parser.py`, `keyword_extractor.py`, `pdf_parser.py`, `requirements.txt`, tests |
| **Phase 4** | `services/skill_extractor.py` | `resume_analyzer.py`, `requirements.txt` |

## Verification

After each phase:
1. `cd backend && source .venv/bin/activate && python -m pytest tests/ -v` — all tests pass
2. `cd frontend && npm run build` — 0 TypeScript errors
3. `npm run lint` — 0 ESLint errors
4. Manual test: upload a resume + JD, verify scores are more accurate and differentiated

## Success Metrics

| Metric | Current | After Phase 1 | After Phase 2 | After Phase 3 |
|--------|---------|---------------|---------------|---------------|
| Keyword false negatives | ~25% (synonyms, abbreviations) | ~5% | ~3% | ~2% |
| `experience_match` accuracy | Proxy (semantic score) | Proxy | Factual (years) | Factual + level |
| Semantic model quality | 22M params | 109M params | 109M + section-level | Same |
| Score breakdown validity | Arbitrary mappings | Same | Maps to real signals | Maps to real signals |
| Bullet feedback | LLM-only | LLM-only | Local + LLM | Local + LLM |
