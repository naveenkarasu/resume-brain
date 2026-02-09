# ConFit-Style Contrastive Training Pipeline for Resume-JD Matching

## Why This Approach

Current pipeline uses 5 hand-tuned weights (semantic, TF-IDF, keyword, skill, section) scoring avg 52.4 on SWE resumes. ConFit (ACM RecSys 2024) achieved 64.3 MAP vs BM25's 34.7 using contrastive learning on resume-JD pairs. ConFit v2 (Feb 2025) improved further with +13.8% recall and +17.5% nDCG.

**Goal:** Replace hand-tuned multi-signal pipeline with a single contrastive fine-tuned encoder.

```
Current:  0.30*semantic + 0.20*tfidf + 0.20*keyword + 0.20*skill + 0.10*section = score
New:      encoder(resume) . encoder(jd) = score  (end-to-end learned)
```

## What Was Already Built (Phases 3-5)

All code is in `resume-brain/backend/`:

### Phase 3 — Resume NER Parser
- `services/resume_ner.py` (NEW) — Lazy-loads `yashpwr/resume-ner-bert-v2`, chunks text, extracts 11 entity categories
- `models/responses.py` — Added `entities: dict` to `SectionAnalysis`
- `services/section_parser.py` — Added optional `ner_entities` param to `extract_contact_info()`, `extract_experience_years()`, `extract_education_level()`
- `services/resume_analyzer.py` — Wired NER into pipeline
- `tests/test_resume_ner.py` (NEW) — 12 unit tests

### Phase 4 — Scoring Recalibration
- `services/resume_analyzer.py` — Flipped LLM/local blend from 0.6/0.4 to 0.4/0.6, wired JD priority sections + required_skills
- `services/skill_extractor.py` — Added `SKILL_IMPLICATIONS` (25 entries), `_expand_skills()`, implied=0.5 credit, required=2x weight, NER chunking fix
- `services/keyword_extractor.py` — Added `_is_negated()` with clause-boundary windowing, `extract_jd_priority_sections()`
- `tests/test_keyword_extractor.py` — 6 new tests (negation, JD priority)
- `tests/test_skill_extractor.py` — 6 new tests (skill hierarchy)

### Phase 5 — Evaluation Infrastructure
- `requirements.txt` — Added `scipy>=1.12.0`
- `tests/conftest.py` (NEW) — Pytest markers: `integration`, `evaluation`
- `tests/test_vacancy_resume_eval.py` (NEW) — 30 CVs x 5 vacancies vs human rankings
- `tests/test_real_resumes_eval.py` (NEW) — SWE resumes vs reference JD
- `tests/test_kaggle_category_eval.py` (NEW) — Category discrimination validation

### Test Status
- 93 unit tests pass, 0 regressions
- SWE resumes eval: PASSED (mean=52.4, 62% above 50)
- Models used: `TechWolf/JobBERT-v2`, `jjzha/jobbert_skill_extraction`, `yashpwr/resume-ner-bert-v2` (all cached in `~/.cache/huggingface/hub/`)

---

## Contrastive Training Pipeline — 21 Sub-Steps

### Phase 1: Data Pipeline

**Step 1 — Build positive pairs from Kaggle**
- Match 66K categorized resumes (`Research/kaggle/Resume.csv`, columns: `Category`, `Resume_str`) with 38K JDs (`Research/kaggle/training_data.csv`, columns: `position_title`, `job_description`) by category/role similarity
- Categories in Resume.csv: INFORMATION-TECHNOLOGY (120), BUSINESS-DEVELOPMENT (120), ENGINEERING (118), FINANCE (118), HR (110), SALES (116), HEALTHCARE (115), etc.

**Step 2 — Generate synthetic JDs**
- For resume categories without matching JDs, use LLM to generate reference JDs from resume text
- ConFit v2's approach: use GPT-4o-mini to generate "hypothetical reference resumes" from JDs (can do reverse too)

**Step 3 — Build hard negatives**
- Pair resumes with wrong-category JDs (e.g., HR resume + Engineering JD)
- After initial training, use model itself to find near-miss pairs (top 3-4% similarity but wrong match)

**Step 4 — Augment with paraphrasing**
- Paraphrase resume/JD sections using EDA (Easy Data Augmentation) to multiply training pairs
- ConFit used ChatGPT paraphrasing on specific sections

**Step 5 — Create train/val/test splits**
- Hold out vacancy-resume dataset (150 human-ranked pairs) as test set — NEVER train on it
- Hold out real resume PDFs as test set
- Split remaining pairs 80/10/10

### Phase 2: Model Architecture

**Step 6 — Choose base encoder**
- Start from `TechWolf/JobBERT-v2` (already cached locally, domain-specialized for job postings)
- Alternative: `intfloat/e5-large-v2` (ConFit's best performer)

**Step 7 — Build dual-encoder architecture**
- Shared encoder that independently encodes resume and JD into dense vectors
- ConFit encodes each field independently, applies self-attention for internal interactions, then merges

**Step 8 — Add projection head**
- Linear layer mapping encoder output to fixed-dim embedding space (e.g., 256-dim)

**Step 9 — Scoring function**
- Cosine similarity or dot product between resume and JD embeddings

### Phase 3: Training Loop

**Step 10 — Implement contrastive loss**
- InfoNCE loss: push positive pairs together, push negatives apart
- B pairs per batch -> B^2 training signals
- Loss: -log(exp(s(R+,J+)) / (exp(s(R+,J+)) + sum(exp(s(R+,J-)))))
- Applied bidirectionally (resume->job and job->resume)

**Step 11 — In-batch negative sampling**
- Every other resume/JD in the batch serves as a negative
- Increases training signals quadratically with batch size

**Step 12 — Hard negative mining**
- After initial training, use model to find near-miss pairs (runner-up mining from top 3-4% similarity)
- Retrain with these hard negatives mixed in
- ConFit v2's Runner-Up Mining (RUM): sample from top percentile ranges, not absolute top-k

**Step 13 — Training hyperparameters**
- Learning rate: 2e-5 (typical for BERT fine-tuning)
- Batch size: 32-64 (larger = more in-batch negatives)
- Temperature: 0.05-0.1 for contrastive loss
- Epochs: 3-10 with early stopping
- Warmup: 10% of total steps

**Step 14 — Checkpointing and early stopping**
- Save best model based on validation nDCG
- Early stop if val metric doesn't improve for 3 epochs

### Phase 4: Evaluation

**Step 15 — Evaluate on vacancy-resume dataset**
- Rank 5 vacancies per CV, compute Spearman correlation and NDCG@5 against human annotators
- Data: `Research/data/vacancy-resume/` — 30 CVs (.docx), 5 vacancies (CSV), 2 annotator rankings
- Target: Spearman > 0.3, NDCG@5 > 0.7

**Step 16 — Evaluate on real resume PDFs**
- Score SWE/Data_AI/Security resumes against reference JDs
- Data: `resume/` directory — 60+ PDFs across 6 categories
- Compare with current pipeline scores (baseline: mean 52.4)

**Step 17 — A/B comparison**
- Run same inputs through old pipeline and new model
- Compare score distributions and ranking quality
- Decision: replace, blend, or keep both

### Phase 5: Integration

**Step 18 — Export trained model**
- Save fine-tuned encoder weights to `backend/models/contrastive/`

**Step 19 — Create inference service**
- New file: `backend/services/contrastive_scorer.py`
- Lazy-load fine-tuned model, same pattern as `similarity.py`

**Step 20 — Replace or blend with current pipeline**
- Option A: Replace `hybrid_similarity()` entirely
- Option B: Add contrastive score as 6th signal with learned weight
- Option C: Use contrastive for ranking, keep old pipeline for interpretable breakdown

**Step 21 — Run full regression tests**
- All 93 existing tests must still pass
- Evaluation tests should show improvement over baseline

---

## File Structure

```
resume-brain/
├── backend/
│   ├── training/
│   │   ├── build_pairs.py          # Steps 1-4: data pipeline
│   │   ├── contrastive_dataset.py  # PyTorch Dataset class
│   │   ├── contrastive_model.py    # Steps 6-9: dual encoder
│   │   ├── contrastive_loss.py     # Steps 10-12: InfoNCE + hard negatives
│   │   ├── train.py                # Steps 13-14: training loop
│   │   └── evaluate.py             # Steps 15-17: evaluation metrics
│   ├── services/
│   │   └── contrastive_scorer.py   # Steps 19-20: inference service
│   └── models/
│       └── contrastive/            # Step 18: saved model weights
├── Research/
│   ├── kaggle/
│   │   ├── Resume.csv              # 66K resumes with Category labels
│   │   ├── training_data.csv       # 38K JDs
│   │   └── resumefilteration.ipynb # Existing similarity matching notebook
│   └── data/vacancy-resume/
│       ├── CV/                     # 65 CVs (.docx)
│       ├── 5_vacancies.csv         # 5 JDs
│       └── annotations-for-the-first-30-vacancies.txt  # Human rankings
└── resume/                         # 60+ real resume PDFs in 6 categories
```

## Data Inventory

| Data | Location | Size | Role |
|------|----------|------|------|
| Kaggle Resume.csv | Research/kaggle/Resume.csv | 66K resumes, 24 categories | Training pairs |
| Kaggle training_data.csv | Research/kaggle/training_data.csv | 38K JDs | Training pairs |
| SkillSpan | Research/data/skillspan/ | 11.5K NER samples | Skill extraction (already used) |
| Vacancy-Resume CVs | Research/data/vacancy-resume/CV/ | 65 .docx files | **TEST ONLY** |
| Vacancy-Resume annotations | Research/data/vacancy-resume/annotations-*.txt | 30 CVs x 5 JDs ranked | **TEST ONLY** |
| Real resume PDFs | resume/ | 60+ PDFs, 6 categories | **TEST ONLY** |

## Dependencies to Add

```
torch>=2.0.0
accelerate>=0.25.0
```

Already have: `transformers`, `sentence-transformers`, `scipy`, `scikit-learn`

## Key Research References

- ConFit v1: https://arxiv.org/abs/2401.16349 (ACM RecSys 2024)
- ConFit v2: https://arxiv.org/abs/2502.12361 (Feb 2025)
- DACL: https://link.springer.com/chapter/10.1007/978-981-95-3055-7_26
- Resume2Vec: https://www.mdpi.com/2079-9292/14/4/794

## How to Run Existing Tests

```bash
cd resume-brain/backend

# Unit tests (fast, ~1.5 min)
.venv/bin/python -m pytest tests/ -v --ignore=tests/test_vacancy_resume_eval.py --ignore=tests/test_real_resumes_eval.py --ignore=tests/test_kaggle_category_eval.py

# Integration test (loads real NER model, ~30s)
.venv/bin/python -m pytest -m integration -v -s

# Evaluation tests (slow on CPU, run separately)
.venv/bin/python -m pytest -m evaluation -v -s tests/test_real_resumes_eval.py      # ~15 min
.venv/bin/python -m pytest -m evaluation -v -s tests/test_kaggle_category_eval.py   # ~40 min
.venv/bin/python -m pytest -m evaluation -v -s tests/test_vacancy_resume_eval.py    # ~2.5 hrs
```
