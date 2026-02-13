# Resume Brain

AI-powered resume analyzer with 3D brain visualization. Upload your resume PDF and a job description to get match scores, missing keywords, optimized bullet points, and an ATS-ready resume.

Built with FastAPI + Google Gemini AI + scikit-learn + Sentence-BERT + React + Three.js.

## Features

- **Hybrid Match Scoring**: Multi-layer NLP pipeline (TF-IDF + Sentence-BERT + Gemini) produces meaningful scores even without an API key
- **Keyword Analysis**: TF-IDF-based dynamic keyword extraction from job descriptions with density analysis
- **Score Breakdown**: Skills, experience, education, and keyword match percentages
- **Bullet Rewrites**: AI-optimized resume bullet points with explanations
- **ATS Resume**: Full ATS-optimized resume text ready to copy
- **3D Brain Visualization**: Interactive Three.js brain that reacts to analysis state
- **Section Parsing**: Detects resume sections (Experience, Education, Skills, etc.) and computes completeness
- **Demo Mode**: Try it without a backend using sample data
- **Graceful Degradation**: Falls back to local NLP pipeline if Gemini is unavailable

## Analysis Pipeline

The hybrid analysis pipeline combines local NLP with LLM-based analysis:

```
Resume + JD
    │
    ├─── Section Parser ─── Structural completeness (0-1)
    │
    ├─── TF-IDF Cosine ─── Lexical similarity (scikit-learn)
    │
    ├─── Sentence-BERT ─── Semantic similarity (all-MiniLM-L6-v2)
    │
    ├─── Keyword Extractor ─── TF-IDF + dictionary matching
    │    └── Density analysis (ATS optimal: 1-3% per keyword)
    │
    └─── Gemini 2.5 Flash ─── Qualitative scoring + rewrites
         │
         ▼
    Weighted Hybrid Score
    (0.35×semantic + 0.30×tfidf + 0.20×keywords + 0.15×sections)
    Blended with LLM: 60% LLM + 40% local NLP
```

When Gemini is unavailable, the local NLP pipeline still produces differentiated, meaningful scores.

## Architecture

```
┌──────────────┐     ┌──────────────────┐     ┌──────────────┐
│   React +    │────▶│   FastAPI         │────▶│   Gemini     │
│   Three.js   │◀────│   Backend         │◀────│   2.5 Flash  │
└──────────────┘     └──────────────────┘     └──────────────┘
     Vite              scikit-learn              google-genai
     Zustand           sentence-transformers
     R3F               pdfplumber
                       section_parser
```

## Desktop App (v2.0.0)

Resume Brain is also available as a standalone desktop app for Windows and macOS. Everything runs locally — no server setup needed. Internet is only required for Gemini LLM calls when you provide an API key.

### Install

Download the latest release from [GitHub Releases](../../releases):
- **Windows**: `.msi` (recommended) or `.exe` (NSIS installer)
- **macOS**: `.dmg`

### Setup

1. Launch Resume Brain after installing
2. Wait for the loading screen to finish (the backend starts automatically)
3. Click the **gear icon** in the header to open Settings
4. (Optional) Enter your **Gemini API key** and click Save — the backend restarts with the key
   - Get a free key at https://aistudio.google.com/apikey
   - Without a key, analysis still works using local NLP (TF-IDF + Sentence-BERT), just no AI-powered bullet rewrites or ATS resume generation

### System Tray

- Closing the window minimizes to the system tray (the backend stays running)
- Click the tray icon to show the window again
- Right-click the tray icon → **Quit** to fully exit and stop the backend

### Build from Source

Prerequisites: Python 3.12, Node.js 20+, Rust (stable), PyInstaller

```bash
# 1. Build backend executable
cd resume-brain/backend
python -m venv .venv && .venv/Scripts/activate  # or source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt && pip install pyinstaller
python build_exe.py

# 2. Setup sidecar
cd ../desktop
powershell -ExecutionPolicy Bypass -File setup-sidecar.ps1  # Windows
# bash setup-sidecar.sh                                      # macOS/Linux

# 3. Install dependencies
cd ../frontend && npm install
cd ../desktop && npm install

# 4. Run in dev mode or build
npx tauri dev          # dev mode with hot reload
npx tauri build        # produce installer
```

---

## Web App

### Quick Start

#### Backend
```bash
cd resume-brain/backend
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
cp ../.env.example .env  # Add your GEMINI_API_KEY
uvicorn main:app --reload
```

Note: First run downloads the Sentence-BERT model (~80MB). Use the CPU-only PyTorch index to save ~1.5GB of disk space.

#### Frontend
```bash
cd resume-brain/frontend
npm install
npm run dev
```

#### Docker
```bash
cd resume-brain
cp .env.example .env  # Add your GEMINI_API_KEY
docker compose up --build
```
Open http://localhost:3000

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check + Gemini status |
| POST | `/analyze` | Upload PDF + job description (multipart) |
| POST | `/analyze/quick` | Text-only analysis (JSON body) |

### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `overall_score` | int | 0-100 blended match score |
| `score_breakdown` | object | Skills, experience, education, keywords percentages |
| `matched_keywords` | string[] | Keywords found in resume |
| `missing_keywords` | string[] | Keywords missing from resume |
| `keyword_density` | object | Keyword frequency percentage in resume text |
| `tfidf_score` | float | TF-IDF cosine similarity (0-1) |
| `semantic_score` | float | Sentence-BERT cosine similarity (0-1) |
| `scoring_method` | string | "hybrid", "local_only", or "llm_only" |
| `section_analysis` | object | Detected sections and completeness score |
| `bullet_rewrites` | array | Original + rewritten bullets with reasoning |
| `ats_optimized_resume` | string | Full ATS-optimized resume text |

## Tech Stack

**Backend**: Python 3.12, FastAPI, Google Gemini 2.5 Flash, scikit-learn (TF-IDF), sentence-transformers (SBERT), pdfplumber, pydantic-settings

**Frontend**: React 18, TypeScript, Vite, Three.js, @react-three/fiber, Tailwind CSS, Zustand, react-dropzone

**Desktop**: Tauri v2, Rust, PyInstaller (sidecar), system tray

**Deploy**: Docker Compose, GitHub Actions, GitHub Pages

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GEMINI_API_KEY` | No* | Google Gemini API key |

*Without it, the app uses the local NLP pipeline (TF-IDF + SBERT + keyword matching) and returns results with `scoring_method: "local_only"`.

Get a free Gemini API key at https://aistudio.google.com/apikey

## License

MIT
