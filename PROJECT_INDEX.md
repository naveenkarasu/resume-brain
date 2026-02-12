# Project Index: Resume-Brain

Generated: 2026-02-09

## Overview

AI-powered resume-JD (job description) matching analyzer with 3D brain visualization. Features a hybrid NLP pipeline (TF-IDF + Sentence-BERT + Gemini) and a 6-model specialized ML pipeline (v2) under development. Built with FastAPI + React + Three.js.

## Project Structure

```
resume-brain/
├── backend/                    # Python FastAPI backend
│   ├── main.py                 # FastAPI entry point
│   ├── config.py               # Settings (pipeline_mode, env vars)
│   ├── requirements.txt        # Python dependencies
│   ├── Dockerfile
│   ├── api/
│   │   ├── router.py           # API routes: /health, /analyze, /analyze/quick
│   │   └── dependencies.py     # Shared DI
│   ├── models/
│   │   ├── requests.py         # QuickAnalyzeRequest
│   │   ├── responses.py        # AnalysisResponse, ScoreBreakdown, BulletRewrite
│   │   └── schemas/            # 6-model inter-model Pydantic contracts (6 files)
│   ├── services/
│   │   ├── resume_analyzer.py  # Main orchestrator (legacy + v2 modes)
│   │   ├── gemini_client.py    # Google Gemini 2.5 Flash wrapper
│   │   ├── pdf_parser.py       # PDF text extraction (pdfplumber)
│   │   ├── similarity.py       # TF-IDF + Sentence-BERT similarity
│   │   ├── keyword_extractor.py# TF-IDF keyword extraction
│   │   ├── skill_extractor.py  # Skill overlap computation
│   │   ├── resume_ner.py       # Resume NER (yashpwr/resume-ner-bert-v2)
│   │   ├── section_parser.py   # Resume section detection
│   │   ├── prompt_builder.py   # Gemini prompt templates
│   │   └── pipeline/           # 6-model v2 pipeline (10 files)
│   │       ├── base.py         # Base model class
│   │       ├── model_registry.py # Lazy-loading model registry
│   │       ├── orchestrator.py # v2 pipeline orchestrator
│   │       ├── m1_jd_extractor.py
│   │       ├── m2_resume_extractor.py
│   │       ├── m3_skills_comparator.py
│   │       ├── m4_exp_edu_comparator.py
│   │       ├── m5_judge.py
│   │       └── m6_verdict.py
│   └── tests/                  # 124 tests (54 pipeline + 70 legacy)
│       ├── conftest.py
│       ├── test_api.py
│       ├── test_similarity.py
│       ├── test_section_parser.py
│       ├── test_keyword_extractor.py
│       ├── test_skill_extractor.py
│       ├── test_resume_ner.py
│       ├── test_pdf_parser.py
│       ├── test_vacancy_resume_eval.py   # Evaluation (~2.5h)
│       ├── test_real_resumes_eval.py     # Evaluation (~15min)
│       ├── test_kaggle_category_eval.py  # Evaluation (~40min)
│       └── test_pipeline/      # 54 pipeline unit tests (7 files)
├── frontend/                   # React + TypeScript + Three.js
│   ├── package.json            # React 19, Three.js, Zustand, Tailwind v4
│   ├── vite.config.ts          # Vite + proxy to backend :8000
│   ├── Dockerfile
│   ├── tsconfig.json
│   └── src/
│       ├── main.tsx            # React entry point
│       ├── App.tsx             # Main app layout (upload + results)
│       ├── api/
│       │   ├── client.ts       # API client (analyzeResume, analyzeQuick, checkHealth)
│       │   └── types.ts        # TypeScript interfaces (AnalysisResponse, etc.)
│       ├── store/
│       │   └── analysisStore.ts # Zustand state (phase, result, file, JD)
│       ├── components/
│       │   ├── layout/         # Header, Footer
│       │   ├── upload/         # ResumeUploader, JobDescriptionInput, AnalyzeButton
│       │   ├── results/        # MatchScoreCard, MissingKeywordsCard, BulletRewritesCard, ATSVersionCard
│       │   └── three/          # BrainScene, BrainMesh, ScoreRing, NeuralConnections, ParticleField, BackgroundParticles
│       └── utils/
│           └── demo-data.ts    # Sample data for demo mode
├── training/                   # ML training pipeline
│   ├── scripts/                # Training entry points (train_m1.py - train_m6.py + evaluate_pipeline.py)
│   ├── data_prep/              # Dataset preprocessing (m1_data.py - m6_data.py)
│   ├── configs/                # YAML training configs (6 files)
│   ├── utils/                  # metrics.py, ner_utils.py, taxonomy.py
│   ├── data/                   # 1.6GB training data (26 datasets, 6 model dirs)
│   └── models/                 # Saved model artifacts (after training)
├── .github/workflows/
│   ├── ci.yml                  # Backend pytest + frontend tsc/build
│   └── deploy-pages.yml        # GitHub Pages deploy (demo mode)
├── docker-compose.yml          # Backend :8000, Frontend :3000
├── train_all.ps1               # PowerShell training script
├── train_all.bat               # Windows batch training script
├── train_all.sh                # Linux/Mac training script
├── CLAUDE.md                   # AI assistant project context
├── EXPLAIN.md                  # Full training guide (Windows + GPU)
├── plan.md                     # 6-model architecture spec
└── README.md                   # Project overview + quick start
```

## Entry Points

- **Backend API**: `backend/main.py` - FastAPI app (uvicorn)
- **Frontend**: `frontend/src/main.tsx` - React 19 + Vite
- **Training**: `training/scripts/train_m1.py` through `train_m6.py`
- **Evaluation**: `training/scripts/evaluate_pipeline.py`
- **Tests**: `backend/tests/` (pytest)

## API Endpoints

| Method | Path             | Description                              |
|--------|------------------|------------------------------------------|
| GET    | `/health`        | Health check + Gemini status             |
| POST   | `/analyze`       | Upload PDF + job description (multipart) |
| POST   | `/analyze/quick` | Text-only analysis (JSON body)           |

## Core Modules

### Backend Services (Legacy Pipeline)
- `resume_analyzer.py` - Main orchestrator, routes to legacy or v2 pipeline
- `gemini_client.py` - Google Gemini 2.5 Flash API (JSON generation, temp=0.3)
- `similarity.py` - TF-IDF cosine + Sentence-BERT (all-MiniLM-L6-v2) similarity
- `keyword_extractor.py` - TF-IDF + dictionary-based keyword extraction with density
- `skill_extractor.py` - Skill overlap computation
- `resume_ner.py` - Resume NER using yashpwr/resume-ner-bert-v2
- `section_parser.py` - Resume section detection + completeness scoring
- `pdf_parser.py` - PDF text extraction via pdfplumber

### Backend Services (v2 Pipeline - 6 Models)
- `pipeline/orchestrator.py` - Runs M1-M6 in sequence, converts to AnalysisResponse
- `pipeline/model_registry.py` - Lazy-loading model artifact registry
- `pipeline/m1_jd_extractor.py` - JD NER (BERT, 9 entity types)
- `pipeline/m2_resume_extractor.py` - Resume NER (BERT, 14 entity types)
- `pipeline/m3_skills_comparator.py` - Skill embedding comparison (JobBERT contrastive)
- `pipeline/m4_exp_edu_comparator.py` - Experience/education matching (LightGBM)
- `pipeline/m5_judge.py` - Overall score regression (LightGBM)
- `pipeline/m6_verdict.py` - Template-based verdict generation

### Inter-Model Schemas (`models/schemas/`)
- `jd_extracted.py` - M1 output: skills, experience, education requirements
- `resume_extracted.py` - M2 output: skills, experience, education profile
- `skills_comparison.py` - M3 output: matched/missing/partial skills
- `exp_edu_comparison.py` - M4 output: experience/education/domain scores
- `judge_result.py` - M5 output: overall score + breakdown
- `verdict_result.py` - M6 output: summary, strengths, weaknesses, rewrites

### Frontend Components
- `store/analysisStore.ts` - Zustand store (phases: idle/loading/results)
- `api/client.ts` - HTTP client with `/api` proxy
- `components/three/` - Three.js brain visualization (R3F)
- `components/upload/` - Resume upload + JD input
- `components/results/` - Score cards, keywords, bullet rewrites, ATS resume

## Configuration

- `backend/config.py` - Pydantic Settings (env-based)
  - `GEMINI_API_KEY` - Google Gemini API key (optional)
  - `PIPELINE_MODE` - "legacy" (default) or "v2"
  - `PIPELINE_V2_STRICT` - Disable fallback to legacy
  - `PIPELINE_MODEL_DIR` - Trained model artifact path
  - `MAX_UPLOAD_SIZE_MB` - PDF upload limit (default: 5)
  - `CORS_ORIGINS` - Allowed origins
- `frontend/vite.config.ts` - Vite dev proxy: `/api` -> `localhost:8000`
- `training/configs/*.yaml` - Per-model training hyperparameters (6 files)

## Tech Stack

| Layer      | Technologies                                                              |
|------------|---------------------------------------------------------------------------|
| Backend    | Python 3.12, FastAPI, Pydantic, Google Gemini 2.5 Flash                   |
| NLP        | scikit-learn (TF-IDF), sentence-transformers (SBERT), transformers, NLTK  |
| ML Models  | PyTorch (BERT fine-tuning), LightGBM, accelerate, seqeval                 |
| Frontend   | React 19, TypeScript 5.9, Vite 7, Three.js/R3F, Tailwind CSS 4, Zustand  |
| Deploy     | Docker Compose, GitHub Actions CI, GitHub Pages (demo mode)               |

## Key Dependencies

### Backend (requirements.txt)
- `fastapi==0.115.0` + `uvicorn==0.30.6` - Web framework
- `google-genai==1.0.0` - Gemini API client
- `scikit-learn==1.5.2` - TF-IDF vectorization
- `sentence-transformers==3.3.1` - SBERT embeddings
- `transformers>=4.40.0` - Hugging Face model loading
- `torch>=2.0` - PyTorch (GPU for training, CPU for inference)
- `lightgbm>=4.0` - Gradient boosting for M4/M5
- `pdfplumber==0.11.4` - PDF parsing
- `rapidfuzz==3.14.3` - Fuzzy string matching

### Frontend (package.json)
- `react@^19.2.0` + `react-dom` - UI framework
- `three@^0.182.0` + `@react-three/fiber@^9.5.0` + `@react-three/drei` - 3D visualization
- `zustand@^5.0.11` - State management
- `tailwindcss@^4.1.18` - Styling
- `react-dropzone@^14.4.0` - File upload

## Pipeline Modes

```
PIPELINE_MODE=legacy (default)
  Resume + JD -> Section Parser + TF-IDF + SBERT + Keywords + Gemini -> Weighted Score
  Formula: 0.35*semantic + 0.30*tfidf + 0.20*keywords + 0.15*sections
  Blended: 60% LLM + 40% local NLP

PIPELINE_MODE=v2
  Resume + JD -> M1(JD NER) + M2(Resume NER) -> M3(Skills) + M4(Exp/Edu) -> M5(Judge) -> M6(Verdict)
  All models have rule-based fallbacks (works without trained models)
  Auto-fallback to legacy on failure (unless PIPELINE_V2_STRICT=true)
```

## Training Data

| Model | Dir                          | Size   | Datasets |
|-------|------------------------------|--------|----------|
| M1    | training/data/model1_jd_extractor/     | 200 MB | 7 (SkillSpan, Green, JD2Skills, Google, Djinni, Sayfullina, SkillBench) |
| M2    | training/data/model2_resume_extractor/ | 943 MB | 5 (yashpwr, DataTurks, Mehyaar, datasetmaster, Djinni) |
| M3    | training/data/model3_skills_comparator/| 61 MB  | 7 (TechWolf, MIND, Tabiya, Nesta, StackLite, RelatedSkills, JobSkillSet) |
| M4    | training/data/model4_exp_edu_comparator/| 50 MB | 6 (JobHop, Karrierewege, JobBERT, titles dedup/normalized, classification) |
| M5    | training/data/model5_judge/            | 169 MB | 6 (netsol, ATS score, resume-JD fit, atlas, AI screening, AI recruitment) |
| M6    | training/data/model6_verdict/          | 149 MB | 4 (MikePfunk critiques, Grammarly CoEdIT, IteraTeR, OpenRewriteEval) |
| **Total** |                                    | **1.6 GB** | **26 datasets** |

## Tests

- **Unit tests**: 70 legacy + 54 pipeline = **124 total**
- **Pipeline tests**: `backend/tests/test_pipeline/` (7 files, 54 tests)
- **Evaluation tests** (slow, run separately):
  - `test_real_resumes_eval.py` (~15min) - 60+ real PDF resumes
  - `test_kaggle_category_eval.py` (~40min) - Category discrimination
  - `test_vacancy_resume_eval.py` (~2.5h) - Gold standard human rankings

## Quick Start

### Backend
```bash
cd resume-brain/backend
python -m venv .venv && .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
# Add GEMINI_API_KEY to .env (optional)
uvicorn main:app --reload
```

### Frontend
```bash
cd resume-brain/frontend
npm install && npm run dev
```

### Docker
```bash
cd resume-brain
docker compose up --build  # Backend :8000, Frontend :3000
```

### Train Models (Windows GPU)
```powershell
cd resume-brain
backend\.venv\Scripts\python training\scripts\train_m1.py  # ~2-3h GPU
backend\.venv\Scripts\python training\scripts\train_m2.py  # ~3-4h GPU
backend\.venv\Scripts\python training\scripts\train_m3.py  # ~1-2h GPU
backend\.venv\Scripts\python training\scripts\train_m4.py  # ~10min CPU
backend\.venv\Scripts\python training\scripts\train_m5.py  # ~10min CPU
backend\.venv\Scripts\python training\scripts\train_m6.py  # instant (template)
```

## Current State

- All infrastructure code complete (47+ files)
- 124 tests passing (rule-based fallbacks)
- No models trained yet - next step
- v2 pipeline works end-to-end with fallbacks
- Legacy pipeline fully operational with/without Gemini API key
