# Plan 2: Resume Brain - Backend NLP Pipeline Overhaul

## Context
The current local NLP pipeline scores too low (38 for a real resume against a real JD). Root causes:
- Generic SBERT model (all-mpnet-base-v2) not trained on job domain
- Keyword extraction uses hardcoded dictionaries instead of ML skill extraction
- No real training data — everything is rule-based
- Heavy dependency on Gemini LLM (60% weight) with conservative local fallback

## Downloads Required (run after restart)

```bash
cd ~/Desktop/ai-log-investigator/resume-brain/backend
source .venv/bin/activate

# 1. Clone test dataset
mkdir -p data
git clone https://github.com/NataliaVanetik/vacancy-resume-matching-dataset.git data/vacancy-resume

# 2. Install new dependencies
pip install datasets transformers[torch]

# 3. Pre-cache models (largest downloads ~2GB total)
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('TechWolf/JobBERT-v2'); print('JobBERT-v2 ready')"
python -c "from transformers import pipeline; pipeline('token-classification', model='jjzha/jobbert_skill_extraction'); print('Skill NER ready')"

# 4. Download SkillSpan dataset
python -c "from datasets import load_dataset; load_dataset('jjzha/skillspan'); print('SkillSpan ready')"
```

**Kaggle dataset** (manual): https://www.kaggle.com/datasets/shamimhasan8/resume-vs-job-description-matching-dataset
→ Place CSV in `backend/data/kaggle-resume-jd/`

## Implementation Plan (Backend Only)

### Phase 1: Model Upgrades (Drop-in replacements)
**Files**: `services/similarity.py`, `requirements.txt`

1. Replace `all-mpnet-base-v2` with `TechWolf/JobBERT-v2` in similarity.py
   - Same SentenceTransformer API, just different model name
   - 1024-dim embeddings (vs 768) — trained on millions of job postings
   - Expected: +15-20% accuracy on job-domain similarity

### Phase 2: ML Skill Extraction (Replace keyword dictionary)
**Files**: `services/skill_extractor.py` (NEW), `services/keyword_extractor.py`

2. Add `jjzha/jobbert_skill_extraction` NER pipeline
   - Extract actual skill spans from both resume AND JD text
   - Replace COMMON_KEYWORDS dictionary with ML-extracted skills
   - Keep dictionary as fallback layer only
3. Train/fine-tune on SkillSpan dataset for hard/soft skill distinction
4. Add skill normalization (canonical form mapping)

### Phase 3: Resume NER Parser
**Files**: `services/resume_ner.py` (NEW), `services/section_parser.py`

5. Add `yashpwr/resume-ner-bert-v2` for structured field extraction
   - Extract: Name, Skills, Experience, Education, Companies, Designations
   - Replace regex-based section parsing with ML extraction
   - Keep regex as fallback

### Phase 4: Scoring Recalibration
**Files**: `services/resume_analyzer.py`

6. Rebalance scoring weights based on real data testing
7. Reduce Gemini dependency: change from 60/40 to 40/60 (local/LLM)
8. Add skill-overlap scoring (ML-extracted skills comparison)
9. Add weighted skill importance (required vs preferred in JD)

### Phase 5: Real-World Testing & Validation
**Files**: `tests/test_real_world.py` (NEW), `tests/test_scoring_accuracy.py` (NEW)

10. Load Vacancy-Resume Matching Dataset (65 resumes, 5 vacancies, human rankings)
11. Compute NDCG and MRR against human HR rankings
12. Test against user's 65 real resumes in resume/ folder
13. Iterative tuning: adjust weights until metrics improve

### Phase 6: Training Pipeline (if needed)
**Files**: `training/` directory (NEW)

14. Load Kaggle resume-JD matching dataset
15. Split 80/10/10 (train/val/test)
16. Fine-tune JobBERT on resume-JD pairs with contrastive loss
17. Evaluate on vacancy-resume matching dataset

## Frontend (DEFERRED — TODO)
- Score ring: Full 360-degree gradient ring (green→red based on score)
- Will implement AFTER backend is satisfactory

## Verification
1. Run scoring on user's real resumes against real JDs
2. Compare before/after scores on same inputs
3. NDCG/MRR on vacancy-resume dataset should improve
4. All existing 57 tests still pass
