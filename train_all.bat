@echo off
REM Train all 6 models for the resume-brain pipeline
REM Run from: resume-brain\ root directory
REM Usage: train_all.bat
REM    or: train_all.bat --sequential

setlocal enabledelayedexpansion

set PYTHON=backend\.venv\Scripts\python.exe
set SEQUENTIAL=false

if "%1"=="--sequential" set SEQUENTIAL=true

echo.
echo ==========================================
echo  Resume-Brain: Training All 6 Models
echo ==========================================
echo.

REM Verify GPU
%PYTHON% -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else 'WARNING: No GPU detected!')"
echo.

REM ============================================
REM Phase 1: M1 + M2 (BERT NER models)
REM ============================================
echo ============================================
echo Phase 1: M1 JD Extractor + M2 Resume Extractor
echo ============================================

if "%SEQUENTIAL%"=="true" (
    echo [M1] Training JD Extractor...
    %PYTHON% training\scripts\train_m1.py
    if errorlevel 1 (echo [M1] FAILED & exit /b 1)
    echo [M1] Done!

    echo [M2] Training Resume Extractor...
    %PYTHON% training\scripts\train_m2.py
    if errorlevel 1 (echo [M2] FAILED & exit /b 1)
    echo [M2] Done!
) else (
    echo [M1+M2] Training in parallel...
    start /b "M1" %PYTHON% training\scripts\train_m1.py
    start /b "M2" %PYTHON% training\scripts\train_m2.py
    echo Waiting for M1 and M2 to finish...
    echo NOTE: On Windows, parallel training may compete for GPU memory.
    echo       If you get CUDA OOM errors, use: train_all.bat --sequential
    REM Windows doesn't have a clean way to wait for background processes
    REM So we use sequential as the reliable path
    echo.
    echo WARNING: Parallel mode on Windows is unreliable. Switching to sequential.
    echo.

    echo [M1] Training JD Extractor...
    %PYTHON% training\scripts\train_m1.py
    if errorlevel 1 (echo [M1] FAILED & exit /b 1)
    echo [M1] Done!

    echo [M2] Training Resume Extractor...
    %PYTHON% training\scripts\train_m2.py
    if errorlevel 1 (echo [M2] FAILED & exit /b 1)
    echo [M2] Done!
)

REM ============================================
REM Phase 2: M3 (contrastive GPU) + M4 (LightGBM CPU)
REM ============================================
echo.
echo ============================================
echo Phase 2: M3 Skills Comparator + M4 Exp/Edu Comparator
echo ============================================

echo [M3] Training Skills Comparator...
%PYTHON% training\scripts\train_m3.py
if errorlevel 1 (echo [M3] FAILED & exit /b 1)
echo [M3] Done!

echo [M4] Training Exp/Edu Comparator...
%PYTHON% training\scripts\train_m4.py
if errorlevel 1 (echo [M4] FAILED & exit /b 1)
echo [M4] Done!

REM ============================================
REM Phase 3: M5 (LightGBM CPU)
REM ============================================
echo.
echo ============================================
echo Phase 3: M5 Judge
echo ============================================

echo [M5] Training Judge...
%PYTHON% training\scripts\train_m5.py
if errorlevel 1 (echo [M5] FAILED & exit /b 1)
echo [M5] Done!

REM ============================================
REM Phase 4: M6 (no training)
REM ============================================
echo.
echo ============================================
echo Phase 4: M6 Verdict (template-based, no training)
echo ============================================

%PYTHON% training\scripts\train_m6.py
echo [M6] Ready!

REM ============================================
REM Done
REM ============================================
echo.
echo ==========================================
echo  ALL MODELS TRAINED SUCCESSFULLY
echo ==========================================
echo.
echo Models saved to: training\models\
dir /b training\models\
echo.
echo Next steps:
echo   1. Verify:
echo      cd backend
echo      set PIPELINE_MODE=v2
echo      .venv\Scripts\python -m pytest tests\test_pipeline\ -v
echo   2. Run app:
echo      set PIPELINE_MODE=v2
echo      .venv\Scripts\python -m uvicorn main:app --reload
echo.

endlocal
