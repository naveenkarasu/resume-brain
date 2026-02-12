# Resume-Brain Project Context

## What This Is
A 6-model specialized ML pipeline for resume-JD (job description) matching that replaces a hand-tuned 7-layer hybrid scoring formula.

## Current State
- **All infrastructure code is complete** - 47 new files, 4 modified files
- **124 tests passing** (54 pipeline + 70 legacy)
- **No models trained yet** - this is the next step
- 1.6GB of training data (26 datasets) is in `training/data/`

## Architecture
6 models in a pipeline: M1 (JD NER) -> M2 (Resume NER) -> M3 (Skills Compare) + M4 (Exp/Edu Compare) -> M5 (Judge) -> M6 (Verdict)

All models have rule-based fallbacks so the pipeline works even without trained models.

## Key Paths
- Pipeline services: `backend/services/pipeline/` (m1 through m6 + orchestrator)
- Schemas: `backend/models/schemas/` (inter-model Pydantic contracts)
- Training scripts: `training/scripts/train_m1.py` through `train_m6.py`
- Training configs: `training/configs/*.yaml`
- Data prep: `training/data_prep/m1_data.py` through `m6_data.py`
- Training data: `training/data/model1_jd_extractor/` through `model6_verdict/`
- Tests: `backend/tests/test_pipeline/`
- Full guide: `EXPLAIN.md`

## What Needs To Be Done
1. Train all 6 models (see EXPLAIN.md for commands and details)
2. Data prep scripts may need debugging when run on actual downloaded datasets
3. After training: verify with `PIPELINE_MODE=v2` pytest
4. A/B evaluation: legacy vs v2 pipeline

## Training Commands (from resume-brain/ root)
```
backend\.venv\Scripts\python training\scripts\train_m1.py   # ~2-3h GPU
backend\.venv\Scripts\python training\scripts\train_m2.py   # ~3-4h GPU
backend\.venv\Scripts\python training\scripts\train_m3.py   # ~1-2h GPU
backend\.venv\Scripts\python training\scripts\train_m4.py   # ~10min CPU
backend\.venv\Scripts\python training\scripts\train_m5.py   # ~10min CPU
backend\.venv\Scripts\python training\scripts\train_m6.py   # exits immediately (Tier 1)
```

## Pipeline Mode
- `PIPELINE_MODE=legacy` (default) - uses existing scoring formula
- `PIPELINE_MODE=v2` - uses 6-model pipeline
- Auto-fallback to legacy on v2 failure
