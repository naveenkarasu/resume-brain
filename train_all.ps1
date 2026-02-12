# Train all 6 models for the resume-brain pipeline
# Run from: resume-brain\ root directory
# Usage: powershell -ExecutionPolicy Bypass -File train_all.ps1

$ErrorActionPreference = "Stop"
$PYTHON = "backend\.venv\Scripts\python.exe"

Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host " Resume-Brain: Training All 6 Models" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# Verify GPU
& $PYTHON -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else 'WARNING: No GPU!')"
Write-Host ""

# Phase 1: M1 + M2
Write-Host "Phase 1: M1 JD Extractor + M2 Resume Extractor" -ForegroundColor Yellow
Write-Host "================================================" -ForegroundColor Yellow

Write-Host "[M1] Training JD Extractor..." -ForegroundColor Green
& $PYTHON training/scripts/train_m1.py
if ($LASTEXITCODE -ne 0) { Write-Host "[M1] FAILED!" -ForegroundColor Red; exit 1 }
Write-Host "[M1] Done!" -ForegroundColor Green

Write-Host "[M2] Training Resume Extractor..." -ForegroundColor Green
& $PYTHON training/scripts/train_m2.py
if ($LASTEXITCODE -ne 0) { Write-Host "[M2] FAILED!" -ForegroundColor Red; exit 1 }
Write-Host "[M2] Done!" -ForegroundColor Green

# Phase 2: M3 + M4
Write-Host ""
Write-Host "Phase 2: M3 Skills Comparator + M4 Exp/Edu Comparator" -ForegroundColor Yellow
Write-Host "======================================================" -ForegroundColor Yellow

Write-Host "[M3] Training Skills Comparator..." -ForegroundColor Green
& $PYTHON training/scripts/train_m3.py
if ($LASTEXITCODE -ne 0) { Write-Host "[M3] FAILED!" -ForegroundColor Red; exit 1 }
Write-Host "[M3] Done!" -ForegroundColor Green

Write-Host "[M4] Training Exp/Edu Comparator..." -ForegroundColor Green
& $PYTHON training/scripts/train_m4.py
if ($LASTEXITCODE -ne 0) { Write-Host "[M4] FAILED!" -ForegroundColor Red; exit 1 }
Write-Host "[M4] Done!" -ForegroundColor Green

# Phase 3: M5
Write-Host ""
Write-Host "Phase 3: M5 Judge" -ForegroundColor Yellow
Write-Host "==================" -ForegroundColor Yellow

Write-Host "[M5] Training Judge..." -ForegroundColor Green
& $PYTHON training/scripts/train_m5.py
if ($LASTEXITCODE -ne 0) { Write-Host "[M5] FAILED!" -ForegroundColor Red; exit 1 }
Write-Host "[M5] Done!" -ForegroundColor Green

# Phase 4: M6
Write-Host ""
Write-Host "Phase 4: M6 Verdict (no training needed)" -ForegroundColor Yellow
Write-Host "==========================================" -ForegroundColor Yellow

& $PYTHON training/scripts/train_m6.py
Write-Host "[M6] Ready!" -ForegroundColor Green

# Done
Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host " ALL MODELS TRAINED SUCCESSFULLY" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Models saved to: training\models\"
Get-ChildItem training\models\ -Directory | ForEach-Object { Write-Host "  $_" }
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host '  1. cd backend'
Write-Host '  2. $env:PIPELINE_MODE="v2"'
Write-Host '  3. .venv\Scripts\python -m pytest tests\test_pipeline\ -v'
Write-Host '  4. .venv\Scripts\python -m uvicorn main:app --reload'
