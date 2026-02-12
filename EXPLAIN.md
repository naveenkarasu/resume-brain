# Resume-Brain: 6-Model Training Guide (Windows + GPU)

## Quick Reference

| What | Details |
|------|---------|
| **Project Size** | 4.5GB total (1.6GB training data, 2.5GB venv, rest is code) |
| **Python** | 3.12+ |
| **Current Status** | All code complete, 124 tests passing, **no models trained yet** |
| **GPU Needed For** | M1 (BERT NER), M2 (BERT NER), M3 (contrastive JobBERT) |
| **CPU Only** | M4 (LightGBM), M5 (LightGBM), M6 (template, no training) |

---

## 1. Moving the Project to Windows

### What to Copy
Copy the entire `resume-brain/` folder. **Skip the Linux venv** - you'll rebuild it on Windows.

```
resume-brain/           # ~2GB without venv
├── backend/            # Application code + tests
├── training/
│   ├── data/           # 1.6GB - ALL 26 datasets already downloaded
│   ├── data_prep/      # Data preprocessing scripts
│   ├── scripts/        # Training entry points
│   ├── configs/        # YAML training configs
│   └── utils/          # Shared utilities
├── train_all.bat       # Windows batch training script
├── train_all.ps1       # PowerShell training script (recommended)
└── Research/           # Reference papers/notes
```

**Copy everything EXCEPT `.venv/`** (Linux venv won't work on Windows):
```
# From Linux, use a USB drive, network share, or zip:
# Exclude .venv since it's Linux-specific
```

Or zip it:
```bash
cd /home/kappa/Desktop/ai-log-investigator
zip -r resume-brain.zip resume-brain/ -x "resume-brain/backend/.venv/*"
```

---

## 2. Windows Environment Setup

### Step 1: Install Python 3.12+
Download from https://www.python.org/downloads/ and install. Check "Add to PATH" during install.

### Step 2: Create venv and install dependencies
Open **PowerShell** or **Command Prompt** in the `resume-brain` folder:

```powershell
cd resume-brain\backend

# Create virtual environment
python -m venv .venv

# Activate it
.venv\Scripts\activate

# Install base requirements
pip install -r requirements.txt

# IMPORTANT: Install PyTorch with CUDA (replace CPU-only version)
# For CUDA 12.1 (most common):
pip install torch --index-url https://download.pytorch.org/whl/cu121

# For CUDA 11.8:
# pip install torch --index-url https://download.pytorch.org/whl/cu118

# Install training dependencies
pip install lightgbm>=4.0 seqeval>=1.2.2 accelerate>=0.25
```

### Step 3: Verify GPU
```powershell
.venv\Scripts\python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}') if torch.cuda.is_available() else print('NO GPU')"
```

Expected output:
```
CUDA: True
GPU: NVIDIA GeForce RTX XXXX
```

### Step 4: Verify Tests Pass
```powershell
cd resume-brain\backend
.venv\Scripts\python -m pytest tests\test_pipeline\ -v --tb=short
# Expected: 54 passed
```

---

## 3. Train All 6 Models

### Option A: One-command PowerShell script (recommended)
```powershell
cd resume-brain
powershell -ExecutionPolicy Bypass -File train_all.ps1
```

### Option B: One-command batch script
```cmd
cd resume-brain
train_all.bat
```

### Option C: Train each model individually

Run all commands from the `resume-brain\` root folder:

```powershell
cd resume-brain
```

**Model 1 - JD Extractor** (~2-3h GPU, ~4GB VRAM):
```powershell
backend\.venv\Scripts\python training\scripts\train_m1.py
```

**Model 2 - Resume Extractor** (~3-4h GPU, ~4GB VRAM):
```powershell
backend\.venv\Scripts\python training\scripts\train_m2.py
```

**Model 3 - Skills Comparator** (~1-2h GPU, ~6GB VRAM):
```powershell
backend\.venv\Scripts\python training\scripts\train_m3.py
```

**Model 4 - Exp/Edu Comparator** (~5-10min CPU):
```powershell
backend\.venv\Scripts\python training\scripts\train_m4.py
```

**Model 5 - Judge** (~5-10min CPU):
```powershell
backend\.venv\Scripts\python training\scripts\train_m5.py
```

**Model 6 - Verdict** (no training, exits immediately):
```powershell
backend\.venv\Scripts\python training\scripts\train_m6.py
```

### Training Order
Train in this order (dependencies):
```
1. M1 and M2 (independent, train one after the other)
2. M3 and M4 (independent, can go in any order)
3. M5 (after M3+M4 are done)
4. M6 (no training needed)
```

---

## 4. Model Details

| Model | Base | Task | Data | Target | Time (GPU) |
|-------|------|------|------|--------|------------|
| M1 | `bert-base-cased` | NER (9 entity types) | 200MB | F1 > 85% | ~2-3h |
| M2 | `yashpwr/resume-ner-bert-v2` | NER (14 entity types) | 943MB | F1 > 90% | ~3-4h |
| M3 | `TechWolf/JobBERT-v2` | Contrastive skill embeddings | 61MB | Accuracy > 80% | ~1-2h |
| M4 | LightGBM | 14 features -> 4 scores | 50MB | Spearman > 0.6 | ~10min CPU |
| M5 | LightGBM | 13 features -> overall score | 169MB | Spearman > 0.5 | ~10min CPU |
| M6 | Templates | Rule-based verdict | N/A | N/A | 0 (no training) |

**Total estimated training time: ~6-8 hours sequential**

---

## 5. After Training: Verification

### Run pipeline tests with trained models
```powershell
cd resume-brain\backend
$env:PIPELINE_MODE="v2"
.venv\Scripts\python -m pytest tests\test_pipeline\ -v --tb=short
# Expected: 54 passed
```

### Run full test suite
```powershell
.venv\Scripts\python -m pytest tests\ -v --tb=short --ignore=tests\test_vacancy_resume_eval.py
# Expected: 124 passed
```

### Activate v2 Pipeline for Production
```powershell
# Set environment variable:
$env:PIPELINE_MODE="v2"

# Run the app:
cd resume-brain\backend
.venv\Scripts\python -m uvicorn main:app --reload
```

Or add to `.env` file in backend folder:
```
PIPELINE_MODE=v2
```

---

## 6. Architecture Overview

```
resume_text + jd_text
  |-- M1.predict(jd_text)              -> JDExtracted (skills, requirements)
  |-- M2.predict(resume_text)          -> ResumeExtracted (skills, experience, edu)
  |       |                                    |
  |-- M3.predict(resume_skills, jd_skills)     -> SkillsComparison
  |-- M4.predict(resume_exp/edu, jd_reqs)      -> ExpEduComparison
  |               |                                    |
  |-- M5.predict(skills_comparison, exp_edu)   -> JudgeResult (score 0-100)
  |                                    |
  +-- M6.predict(all_outputs + text)           -> VerdictResult (summary, strengths)
                   |
     _to_analysis_response()           -> AnalysisResponse (backward compatible)
```

### Pipeline Mode Switch
- `PIPELINE_MODE=legacy` (default): Uses existing 7-layer hybrid formula
- `PIPELINE_MODE=v2`: Uses 6-model pipeline
- On v2 failure: auto-falls back to legacy (unless `PIPELINE_V2_STRICT=true`)

### All Models Have Fallbacks
Every model works **without trained models** using rule-based fallbacks. This is why the 54 tests pass right now even before training.

---

## 7. Troubleshooting (Windows)

### "CUDA out of memory"
Edit the YAML config to reduce batch size:
```
training\configs\m1_jd_extractor.yaml (or m2, m3)
  training:
    batch_size: 8    # reduce from 16
```

### "No module named X"
```powershell
cd resume-brain\backend
.venv\Scripts\pip install lightgbm seqeval accelerate pyyaml
```

### torch CPU-only (CUDA: False)
```powershell
.venv\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### "ModuleNotFoundError: No module named 'services'"
You must run training scripts from the `resume-brain\` root, not from `training\`:
```powershell
cd resume-brain
backend\.venv\Scripts\python training\scripts\train_m1.py
```

### Windows path issues with forward slashes
The training scripts use forward slashes in paths. Python handles this fine on Windows. The YAML configs also use forward slashes - this is normal and works.

### "LongPathsEnabled" error
If you get path-too-long errors on Windows:
```powershell
# Run as Administrator:
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
```

---

## 8. File Structure

```
resume-brain\
|-- backend\
|   |-- .venv\                          # Python virtualenv (rebuild on Windows)
|   |-- config.py                       # Settings (pipeline_mode, etc.)
|   |-- main.py                         # FastAPI entry point
|   |-- requirements.txt                # All dependencies
|   |-- models\
|   |   |-- responses.py                # AnalysisResponse schema
|   |   +-- schemas\                    # Inter-model contracts (6 files)
|   |-- services\
|   |   |-- resume_analyzer.py          # Main orchestrator (legacy + v2)
|   |   |-- pipeline\                   # 6-model inference pipeline (9 files)
|   |   +-- (legacy services)
|   +-- tests\
|       |-- test_pipeline\              # 54 pipeline tests (7 files)
|       +-- (70 legacy tests)
|-- training\
|   |-- data\                           # 1.6GB datasets (26 total)
|   |-- data_prep\                      # Dataset preprocessing (6 files)
|   |-- scripts\                        # Training scripts (7 files)
|   |-- configs\                        # YAML configs (6 files)
|   |-- utils\                          # Metrics, NER utils, taxonomy
|   +-- models\                         # Saved model artifacts (after training)
|-- train_all.bat                       # Windows batch training script
|-- train_all.ps1                       # PowerShell training script
|-- train_all.sh                        # Linux/Mac training script
+-- EXPLAIN.md                          # This file
```

---

## 9. Continuing with Claude Code on Windows

### Install Claude Code on Windows
```powershell
npm install -g @anthropic-ai/claude-code
```

### Start a new session
```powershell
cd resume-brain
claude
```

### Paste this to give Claude context:

```
Read EXPLAIN.md for the full project architecture and training guide.

I'm working on the resume-brain project - a 6-model specialized ML pipeline for
resume-JD matching. Current state:

COMPLETED:
- All infrastructure code: 47 new files, 4 modified files
- 6 Pydantic inter-model schemas (backend\models\schemas\)
- 6 model service classes with rule-based fallbacks (backend\services\pipeline\)
- Lazy-loading model registry, orchestrator, pipeline mode switch
- Training scripts, data_prep, configs, utilities (training\)
- 1.6GB training data (26 datasets) downloaded (training\data\)
- 124 tests passing (54 pipeline + 70 legacy)

NEEDS TO BE DONE NOW:
- Train all 6 models using the training scripts
- Run: backend\.venv\Scripts\python training\scripts\train_m1.py
  (and train_m2 through train_m6)
- Data prep scripts may need debugging when run on actual data
- After training: set PIPELINE_MODE=v2 and run pytest to verify
- A/B evaluation: legacy vs v2 pipeline

Key paths:
- Training scripts: training\scripts\train_m1.py through train_m6.py
- Configs: training\configs\*.yaml
- Data prep: training\data_prep\m1_data.py through m6_data.py
- Pipeline services: backend\services\pipeline\
- Tests: backend\tests\test_pipeline\

Help me train the models one at a time, starting with M1. Debug any data prep
issues that come up.
```

### Alternative: Use the CLAUDE.md file
You can also put the context prompt above into a `CLAUDE.md` file in the `resume-brain\` root. Claude Code reads it automatically at session start:

```powershell
# Create CLAUDE.md with the context above, then:
cd resume-brain
claude
# Claude will automatically read CLAUDE.md and have full context
```

---

## 10. Expected Outcomes After Training

| Model | Target Metric | What It Replaces |
|-------|--------------|-----------------|
| M1 | F1 > 85% | `keyword_extractor.extract_keywords_combined()` |
| M2 | F1 > 90% | `resume_ner.extract_resume_entities()` |
| M3 | Accuracy > 80% | `skill_extractor.compute_skill_overlap()` |
| M4 | Spearman > 0.6 | `section_parser.compute_experience_match()` |
| M5 | Spearman > 0.5, NDCG@5 > 0.7 | Hand-tuned 30/20/20/20/10 formula |
| M6 | N/A (templates) | `gemini_client.py` for deterministic output |

**Total estimated training time on GPU: ~6-8 hours sequential**
