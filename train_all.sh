#!/bin/bash
# Train all 6 models for the resume-brain pipeline
# Run from: resume-brain/ root directory
# Usage: bash train_all.sh [--sequential]
set -e

PYTHON="backend/.venv/bin/python"
SEQUENTIAL=false

if [ "$1" == "--sequential" ]; then
    SEQUENTIAL=true
    echo "[INFO] Running in sequential mode (for low VRAM GPUs)"
fi

echo ""
echo "=========================================="
echo " Resume-Brain: Training All 6 Models"
echo "=========================================="
echo ""

# Verify GPU
$PYTHON -c "
import torch
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f}GB')
else:
    print('WARNING: No GPU detected! BERT training will be very slow.')
    print('M4 and M5 (LightGBM) will still run fine on CPU.')
"
echo ""

# ============================================
# Phase 1: M1 + M2 (BERT NER models)
# ============================================
echo "============================================"
echo "Phase 1: M1 JD Extractor + M2 Resume Extractor"
echo "============================================"

if [ "$SEQUENTIAL" = true ]; then
    echo "[M1] Training JD Extractor..."
    $PYTHON training/scripts/train_m1.py
    echo "[M1] Done!"

    echo "[M2] Training Resume Extractor..."
    $PYTHON training/scripts/train_m2.py
    echo "[M2] Done!"
else
    echo "[M1+M2] Training in parallel..."
    $PYTHON training/scripts/train_m1.py &
    M1_PID=$!
    $PYTHON training/scripts/train_m2.py &
    M2_PID=$!

    wait $M1_PID
    echo "[M1] JD Extractor training complete!"
    wait $M2_PID
    echo "[M2] Resume Extractor training complete!"
fi

# ============================================
# Phase 2: M3 (contrastive GPU) + M4 (LightGBM CPU)
# ============================================
echo ""
echo "============================================"
echo "Phase 2: M3 Skills Comparator + M4 Exp/Edu Comparator"
echo "============================================"

if [ "$SEQUENTIAL" = true ]; then
    echo "[M3] Training Skills Comparator..."
    $PYTHON training/scripts/train_m3.py
    echo "[M3] Done!"

    echo "[M4] Training Exp/Edu Comparator..."
    $PYTHON training/scripts/train_m4.py
    echo "[M4] Done!"
else
    echo "[M3+M4] Training in parallel..."
    $PYTHON training/scripts/train_m3.py &
    M3_PID=$!
    $PYTHON training/scripts/train_m4.py &
    M4_PID=$!

    wait $M3_PID
    echo "[M3] Skills Comparator training complete!"
    wait $M4_PID
    echo "[M4] Exp/Edu Comparator training complete!"
fi

# ============================================
# Phase 3: M5 (LightGBM CPU)
# ============================================
echo ""
echo "============================================"
echo "Phase 3: M5 Judge"
echo "============================================"

echo "[M5] Training Judge..."
$PYTHON training/scripts/train_m5.py
echo "[M5] Done!"

# ============================================
# Phase 4: M6 (no training - template based)
# ============================================
echo ""
echo "============================================"
echo "Phase 4: M6 Verdict (template-based, no training)"
echo "============================================"

$PYTHON training/scripts/train_m6.py
echo "[M6] Ready!"

# ============================================
# Done!
# ============================================
echo ""
echo "=========================================="
echo " ALL MODELS TRAINED SUCCESSFULLY"
echo "=========================================="
echo ""
echo "Models saved to: training/models/"
ls -la training/models/*/
echo ""
echo "Next steps:"
echo "  1. Verify: cd backend && PIPELINE_MODE=v2 .venv/bin/python -m pytest tests/test_pipeline/ -v"
echo "  2. Run app: cd backend && PIPELINE_MODE=v2 .venv/bin/python -m uvicorn main:app --reload"
echo ""
