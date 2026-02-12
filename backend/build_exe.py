#!/usr/bin/env python3
"""Build script for Resume Brain backend executable.

Creates a standalone distributable using PyInstaller.
Run from the backend/ directory:
    python build_exe.py

Output: dist/resume-brain/resume-brain.exe

Prerequisites:
    pip install pyinstaller
"""

import os
import subprocess
import sys
from pathlib import Path


def ensure_nltk_data():
    """Download NLTK wordnet data if not present."""
    try:
        import nltk
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
        print("[OK] NLTK data ready")
    except ImportError:
        print("[WARN] nltk not installed, skipping NLTK data download")


def check_pyinstaller():
    """Ensure PyInstaller is installed."""
    try:
        import PyInstaller
        print(f"[OK] PyInstaller {PyInstaller.__version__} found")
        return True
    except ImportError:
        print("[ERROR] PyInstaller not installed. Run: pip install pyinstaller")
        return False


def clean_previous_build():
    """Remove previous build artifacts."""
    import shutil
    backend_dir = Path(__file__).parent

    for folder in ['build', 'dist']:
        target = backend_dir / folder / 'resume-brain'
        if target.exists():
            print(f"[CLEAN] Removing {target}")
            shutil.rmtree(target)


def build():
    """Run PyInstaller with the spec file."""
    backend_dir = Path(__file__).parent
    spec_file = backend_dir / 'resume-brain.spec'

    if not spec_file.exists():
        print(f"[ERROR] Spec file not found: {spec_file}")
        sys.exit(1)

    print("\n=== Building Resume Brain Backend ===\n")

    # Step 1: NLTK data
    ensure_nltk_data()

    # Step 2: Check PyInstaller
    if not check_pyinstaller():
        sys.exit(1)

    # Step 3: Clean
    clean_previous_build()

    # Step 4: Build
    print("\n[BUILD] Running PyInstaller (this may take several minutes)...\n")
    result = subprocess.run(
        [sys.executable, '-m', 'PyInstaller', str(spec_file), '--noconfirm'],
        cwd=str(backend_dir),
    )

    if result.returncode != 0:
        print("\n[ERROR] PyInstaller build failed!")
        sys.exit(1)

    # Step 5: Verify
    exe_path = backend_dir / 'dist' / 'resume-brain' / 'resume-brain.exe'
    if exe_path.exists():
        size_mb = exe_path.stat().st_size / (1024 * 1024)
        print(f"\n[OK] Build successful!")
        print(f"     Executable: {exe_path}")
        print(f"     Size: {size_mb:.1f} MB")

        # Check total dist size
        dist_dir = backend_dir / 'dist' / 'resume-brain'
        total_size = sum(f.stat().st_size for f in dist_dir.rglob('*') if f.is_file())
        total_mb = total_size / (1024 * 1024)
        print(f"     Total dist: {total_mb:.0f} MB")
        print(f"\n     Test with: dist\\resume-brain\\resume-brain.exe")
    else:
        print("\n[ERROR] Expected exe not found at:", exe_path)
        sys.exit(1)


if __name__ == '__main__':
    build()
