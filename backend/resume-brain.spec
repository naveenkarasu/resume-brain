# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for Resume Brain backend.

Bundles the FastAPI + ML backend into a one-directory distributable.
Run with: pyinstaller resume-brain.spec

Output: dist/resume-brain/resume-brain.exe
"""

import os
import sys
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

# Paths
backend_dir = os.path.abspath('.')
project_root = os.path.abspath('..')
models_dir = os.path.join(project_root, 'training', 'models')

# --- Hidden imports ---
# PyInstaller misses many dynamic imports in ML libraries
hiddenimports = [
    # FastAPI / Uvicorn
    'uvicorn.logging',
    'uvicorn.loops',
    'uvicorn.loops.auto',
    'uvicorn.protocols',
    'uvicorn.protocols.http',
    'uvicorn.protocols.http.auto',
    'uvicorn.protocols.websockets',
    'uvicorn.protocols.websockets.auto',
    'uvicorn.lifespan',
    'uvicorn.lifespan.on',
    'uvicorn.lifespan.off',
    'multipart',
    'multipart.multipart',

    # Pydantic
    'pydantic',
    'pydantic_settings',
    'pydantic.deprecated.decorator',

    # ML stack
    'torch',
    'torch.nn',
    'torch.utils',
    'torch.utils.data',
    'transformers',
    'sentence_transformers',
    'sklearn',
    'sklearn.utils._cython_blas',
    'sklearn.neighbors._typedefs',
    'sklearn.neighbors._quad_tree',
    'sklearn.tree._utils',
    'lightgbm',
    'accelerate',
    'scipy',
    'scipy.special',
    'scipy.special._cdflib',

    # NLP
    'nltk',
    'nltk.stem',
    'nltk.stem.wordnet',
    'rapidfuzz',
    'rapidfuzz.fuzz',

    # PDF / Doc
    'pdfplumber',
    'pdfminer',
    'pdfminer.high_level',
    'docx',

    # Other
    'seqeval',
    'seqeval.metrics',
    'yaml',
    'httpx',
    'slowapi',
    'datasets',

    # App modules
    'api',
    'api.router',
    'api.dependencies',
    'api.requests',
    'config',
    'models',
    'models.responses',
    'models.requests',
    'models.schemas',
    'models.schemas.jd_extracted',
    'models.schemas.resume_extracted',
    'models.schemas.skills_comparison',
    'models.schemas.exp_edu_comparison',
    'models.schemas.judge_result',
    'models.schemas.verdict_result',
    'services',
    'services.pdf_parser',
    'services.resume_analyzer',
    'services.keyword_extractor',
    'services.resume_ner',
    'services.section_parser',
    'services.skill_extractor',
    'services.similarity',
    'services.prompt_builder',
    'services.gemini_client',
    'services.pipeline',
    'services.pipeline.base',
    'services.pipeline.model_registry',
    'services.pipeline.orchestrator',
    'services.pipeline.m1_jd_extractor',
    'services.pipeline.m2_resume_extractor',
    'services.pipeline.m3_skills_comparator',
    'services.pipeline.m4_exp_edu_comparator',
    'services.pipeline.m5_judge',
    'services.pipeline.m6_verdict',
]

# Collect all submodules for tricky packages
for pkg in ['transformers', 'sentence_transformers', 'sklearn', 'lightgbm', 'pdfminer']:
    try:
        hiddenimports += collect_submodules(pkg)
    except Exception:
        pass

# --- Data files ---
datas = []

# Trained ML models (if they exist)
if os.path.isdir(models_dir):
    datas.append((models_dir, 'training/models'))

# NLTK data - wordnet lemmatizer
nltk_data_candidates = [
    os.path.join(os.path.expanduser('~'), 'nltk_data'),
    os.path.join(sys.prefix, 'nltk_data'),
    os.path.join(sys.prefix, 'share', 'nltk_data'),
    os.path.join(backend_dir, 'nltk_data'),
]
for nltk_path in nltk_data_candidates:
    wordnet_path = os.path.join(nltk_path, 'corpora', 'wordnet')
    if os.path.isdir(wordnet_path):
        datas.append((os.path.join(nltk_path, 'corpora', 'wordnet'), 'nltk_data/corpora/wordnet'))
        # Also grab omw-1.4 if present (needed by wordnet)
        omw_path = os.path.join(nltk_path, 'corpora', 'omw-1.4')
        if os.path.isdir(omw_path):
            datas.append((omw_path, 'nltk_data/corpora/omw-1.4'))
        break

# Collect data files for key packages
for pkg in ['transformers', 'sentence_transformers', 'pdfplumber']:
    try:
        datas += collect_data_files(pkg)
    except Exception:
        pass

# --- Analysis ---
a = Analysis(
    ['main.py'],
    pathex=[backend_dir],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tkinter',
        'matplotlib',
        'IPython',
        'jupyter',
        'notebook',
        'pytest',
        'pytest_asyncio',
        'pip',
        'setuptools',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='resume-brain',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,  # Keep console for logging; Tauri hides it anyway
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='resume-brain',
)
