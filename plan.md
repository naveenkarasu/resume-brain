# Resume-Brain: 6-Model Specialized Architecture Plan

## Vision

Replace the hand-tuned multi-signal pipeline (`0.30*semantic + 0.20*tfidf + 0.20*keyword + 0.20*skill + 0.10*section`) with a **6-model specialized architecture** where each model handles one step of resume-JD analysis, coordinated by a master pipeline.

```
Current:  hand-tuned weights → blended score
New:      6 specialized models → learned scoring → interpretable feedback
```

## Why 6 Models Instead of 1

- A single monolithic model (like ConFit) is a black box that gives one score
- 6 specialized models give: structured extraction, granular comparison, interpretable scoring, AND actionable feedback
- Each model can be independently improved, tested, and swapped
- Portfolio story: "Systematic improvement with measurable impact at each stage"

## Architecture Overview

```
Raw Resume Text ──→ [Model 2: Resume Extractor] ──→ Structured Resume Profile
                                                          │
Raw JD Text ──────→ [Model 1: JD Extractor] ──────→ Structured JD Requirements
                                                          │
                    ┌─────────────────────────────────────┘
                    ▼
        ┌──────────────────────┐    ┌──────────────────────────┐
        │ Model 3: Skills      │    │ Model 4: Exp/Edu         │
        │ Comparator           │    │ Comparator               │
        │ (matched/missing/    │    │ (experience gap,         │
        │  partial skills)     │    │  education fit, domain)  │
        └──────────┬───────────┘    └────────────┬─────────────┘
                   │                              │
                   ▼                              ▼
              ┌────────────────────────────────────────┐
              │ Model 5: Judge                         │
              │ (combines all signals → quality score) │
              └────────────────┬───────────────────────┘
                               │
                               ▼
              ┌────────────────────────────────────────┐
              │ Model 6: Verdict                       │
              │ (score breakdown, rewritten bullets,   │
              │  actionable feedback, missing keywords)│
              └────────────────────────────────────────┘
```

---

## Model Specifications

### Model 1: JD Extractor

**Purpose:** Extract structured requirements from raw job description text.

**Input:** Raw JD text (string)

**Output:**
```json
{
  "required_skills": ["Python", "Kubernetes", "PostgreSQL"],
  "preferred_skills": ["Go", "GraphQL"],
  "years_experience": 5,
  "education_level": "bachelors",
  "education_field": "Computer Science",
  "seniority_level": "senior",
  "certifications": ["AWS Certified"],
  "soft_skills": ["leadership", "communication"],
  "domain": "backend engineering"
}
```

**Approach:** Token-level NER (BIO tagging) fine-tuned on job posting text. Multi-label classification for seniority and domain.

**Base Model Candidates:**
- `jjzha/esco-xlm-roberta-large` (ESCO-pretrained, ACL 2023)
- `BERT-base` fine-tuned on skill extraction

**Datasets (200 MB, downloaded):**

| Dataset | Size | What It Provides |
|---------|------|------------------|
| Green et al. Skill Extraction Benchmark | 18,600 entities | 5 NER types: Skill, Qualification, Experience, Occupation, Domain |
| JOBS-Information-Extraction | ~2K annotated JDs | Years of experience + certifications as NER entities |
| JD2Skills-BERT-XMLC | 20,298 postings | 16 structured fields including seniority, experience |
| SkillSpan (in Research/) | 14,500 sentences | Hard/soft skill spans, BIO tags |
| Skill-Extraction-Benchmark | 3,894 sentences | ESCO-linked skill spans |
| Google Job Skills | ~1,250 postings | Minimum vs Preferred Qualifications split |
| Djinni JDs | 142K JDs | Experience years (5 classes), primary keyword (45 classes) |
| Sayfullina Soft Skills | 7,411 sentences | Soft skill BIO tags |

**Taxonomies for normalization:** O*NET (923 occupations), ESCO (13,939 skills)

---

### Model 2: Resume Extractor

**Purpose:** Extract structured profile from raw resume text.

**Input:** Raw resume text (string, from PDF extraction)

**Output:**
```json
{
  "skills": [{"name": "Python", "proficiency": "expert", "years": 5}],
  "experience": [
    {"title": "Senior SWE", "company": "Google", "start": "2020-01", "end": "2024-06", "responsibilities": [...]}
  ],
  "education": [
    {"degree": "MS", "field": "Computer Science", "institution": "Stanford", "year": 2020}
  ],
  "certifications": ["AWS Solutions Architect"],
  "projects": [{"name": "...", "technologies": [...], "impact": "..."}],
  "total_years_experience": 8.5
}
```

**Approach:** Token-level NER (BIO tagging) for entity extraction + section classification. Build on existing `resume_ner.py` (yashpwr/resume-ner-bert-v2).

**Base Model Candidates:**
- `yashpwr/resume-ner-bert-v2` (already in use, 90.87% F1)
- Fine-tune with additional data for project/certification extraction

**Datasets (943 MB, downloaded):**

| Dataset | Size | What It Provides |
|---------|------|------------------|
| yashpwr/resume-ner-training-data | 22,855 samples | 25 entity types (skills, degree, company, title, years) |
| DataTurks Resume NER | 220 resumes | High-quality manual annotations |
| Mehyaar NER Annotated CVs | 5,029 CVs | IT skills focus |
| datasetmaster/resumes | 1K-10K | Pre-structured JSON (skills, projects, education, roles) |
| Djinni Candidate Profiles | 230K profiles | Structured IT candidate profiles |

---

### Model 3: Skills Comparator

**Purpose:** Compare skills between extracted resume profile and JD requirements. Determine which skills match, which are missing, and which are partial matches (synonyms/related).

**Input:** Resume skills list + JD required/preferred skills lists

**Output:**
```json
{
  "matched_skills": [{"resume": "PyTorch", "jd": "PyTorch", "confidence": 1.0}],
  "partial_matches": [{"resume": "TensorFlow", "jd": "PyTorch", "confidence": 0.7, "reason": "both deep learning frameworks"}],
  "missing_required": ["Kubernetes", "Go"],
  "missing_preferred": ["GraphQL"],
  "extra_skills": ["Redis", "Docker"],
  "skill_coverage_score": 0.72
}
```

**Approach:** Skill embedding similarity using taxonomy-aware representations. Skills mapped to ESCO/O*NET taxonomy, then compared using learned embeddings that encode skill relationships.

**Base Model Candidates:**
- Contrastive encoder trained on TechWolf ESCO-skill-sentences (138K pairs)
- Skill2Vec from ESCO/O*NET co-occurrence matrices

**Datasets (61 MB, downloaded):**

| Dataset | Size | What It Provides |
|---------|------|------------------|
| TechWolf Synthetic-ESCO-Skill-Sentences | 138,260 pairs | Skill-sentence contrastive training pairs |
| MIND Tech Ontology | 19 tech categories | Tech-specific skill relationships |
| Tabiya ESCO Open Dataset | Full ESCO tabular | Skill synonyms/alt labels |
| Nesta UK Skills Taxonomy | 10,500 skills | Data-driven skill clustering (143 clusters) |
| StackLite | Millions Q&A | Tech skill co-occurrence |
| Related Job Skills | Direct pairs | Skill-to-skill relatedness |
| Job-Skill-Set | Per job | Hard/soft skills per job title |

**Taxonomy backbone:** O*NET (923 occupations x 120 skill dimensions) + ESCO (13,939 skills, hierarchical)

---

### Model 4: Exp/Edu Comparator

**Purpose:** Compare experience level, education credentials, and domain relevance between resume and JD.

**Input:** Resume experience/education + JD requirements

**Output:**
```json
{
  "experience_match": {
    "resume_years": 8.5,
    "required_years": 5,
    "score": 0.95,
    "assessment": "exceeds_requirement"
  },
  "education_match": {
    "resume_degree": "MS Computer Science",
    "required_degree": "BS Computer Science",
    "score": 0.90,
    "assessment": "exceeds_requirement"
  },
  "domain_relevance": {
    "resume_domain": "backend engineering",
    "jd_domain": "backend engineering",
    "score": 0.95
  },
  "career_trajectory_score": 0.85
}
```

**Approach:** Learn-to-rank on structured features. Job title normalization via JobBERT, education matching via O*NET education distributions, experience gap scoring.

**Base Model Candidates:**
- `TechWolf/JobBERT-v2` or `v3` for title embeddings
- Gradient boosted trees (XGBoost/LightGBM) on structured features

**Datasets (50 MB downloaded + 100K streaming sample):**

| Dataset | Size | What It Provides |
|---------|------|------------------|
| JobHop | 1.68M experiences (391K resumes) | Career trajectories with ESCO codes |
| Karrierewege (100K sample) | 100K rows | Career paths with skills per role |
| O*NET Education/Training | 923 occupations | Education level distributions per occupation |
| JobBERT Evaluation Dataset | 30,926 titles | Job title → ESCO normalization ground truth |
| Job Titles (deduplicated) | 65,248 titles | Standardized titles from ESCO/O*NET/OSCA |
| Job Titles (normalized) | ~70K titles | Cleaned flat reference list |
| Job Classification | Per job | Classification levels and paygrades |

---

### Model 5: Judge

**Purpose:** Combine all comparison signals from Models 3 and 4 into an overall match quality score with confidence.

**Input:** Outputs from Models 3 (skill comparison) and 4 (exp/edu comparison)

**Output:**
```json
{
  "overall_score": 78,
  "confidence": 0.85,
  "score_breakdown": {
    "skills_match": 72,
    "experience_match": 95,
    "education_match": 90,
    "domain_relevance": 85
  },
  "fit_category": "good_fit"
}
```

**Approach:** Learn-to-rank model trained on labeled resume-JD pairs. Takes structured features from Models 3 & 4 as input. Strategy A+B: combine existing labeled data (15K pairs) + Claude-labeled additional pairs.

**Base Model Candidates:**
- LightGBM/XGBoost on structured features (simple, interpretable)
- Small neural network (MLP) for non-linear signal combination
- Cross-encoder fine-tuned on resume-JD pairs (for text-based scoring)

**Datasets (169 MB, downloaded):**

| Dataset | Size | What It Provides |
|---------|------|------------------|
| netsol/resume-score-details | 1,031 pairs | Multi-dimensional scores + natural language justifications |
| resume-ats-score-v1-en | 6,370 pairs | Continuous ATS scores (19.2-90.1) |
| resume-job-description-fit | 8,000 pairs | 3-class fit labels (Fit/Partial/No Fit) |
| resume-atlas | 13,389 resumes | 43-category classification |
| AI Resume Screening 2025 | Structured | AI scores + hiring predictions |
| AI Recruitment Pipeline | 1K+ profiles | AI scores + screening outcomes |

**Gap-fill strategy:** Claude labels additional pairs from Djinni data (Strategy A+B). Validate against 150 gold-standard human-ranked pairs (vacancy-resume dataset in Research/).

---

### Model 6: Verdict

**Purpose:** Generate the final user-facing output: score breakdown, rewritten bullet points, and actionable feedback.

**Input:** All outputs from Models 1-5 + original resume/JD text

**Output:**
```json
{
  "overall_score": 78,
  "score_breakdown": {"skills": 72, "experience": 95, "education": 90, "keywords": 68},
  "bullet_rewrites": [
    {"original": "Worked on Python projects", "rewritten": "Developed 3 Python microservices reducing API latency by 40%", "reason": "Added metrics, action verb, and specific impact"}
  ],
  "missing_keywords": ["Kubernetes", "CI/CD", "Go"],
  "actionable_feedback": [
    "Add a DevOps section highlighting deployment experience",
    "Quantify your Python project impacts with metrics",
    "Include Kubernetes experience from your Docker work"
  ],
  "strengths": ["Strong Python/backend experience", "Exceeds education requirements"],
  "weaknesses": ["No cloud certification mentioned", "Missing container orchestration skills"]
}
```

**Approach:** Fine-tuned language model for structured generation. Takes structured signals from all models and generates natural language feedback.

**Base Model Candidates:**
- Fine-tuned T5/FLAN-T5 for structured output generation
- Fine-tuned Mistral-7B or Phi-3 for richer feedback (if GPU allows)
- Alternatively: template-based generation from Model 5 signals (no LLM needed)

**Datasets (149 MB, downloaded):**

| Dataset | Size | What It Provides |
|---------|------|------------------|
| MikePfunk28/resume-training-dataset | 22,855 conversations | Expert resume critiques + improvement suggestions |
| Grammarly CoEdIT | 69,783 pairs | Instruction-tuned text editing (6 task types) |
| IteraTeR | 4,018 pairs | Before/after edits with intent labels (Clarity, Style) |
| OpenRewriteEval | 1,629 pairs | Cross-sentence rewrites with instructions |

**Gap-fill strategy:** Generate resume-specific bullet rewrite pairs using data from Models 1 & 2 datasets + Claude labeling.

---

## Data Inventory

### Downloaded Datasets

| Model | Disk Size | Data Files | Status |
|-------|-----------|------------|--------|
| Model 1 (JD Extractor) | 200 MB | 15 | Complete |
| Model 2 (Resume Extractor) | 943 MB | 5,033 | Complete |
| Model 3 (Skills Comparator) | 61 MB | 47 | Complete |
| Model 4 (Exp/Edu Comparator) | 50 MB | 9 | Complete |
| Model 5 (Judge) | 169 MB | 1,038 | Complete |
| Model 6 (Verdict) | 149 MB | 6 | Complete |
| **Total** | **1.6 GB** | **6,148** | **All data-rich** |

### Test Data (NEVER train on these)

| Data | Location | Size | Purpose |
|------|----------|------|---------|
| Vacancy-Resume CVs | Research/data/vacancy-resume/CV/ | 65 .docx | Gold standard test set |
| Vacancy-Resume annotations | Research/data/vacancy-resume/annotations-*.txt | 150 ranked pairs | Human rankings (2 annotators) |
| Real resume PDFs | resume/ | 60+ PDFs, 6 categories | End-to-end eval |

### Data Location

```
resume-brain/training/data/
├── model1_jd_extractor/
│   ├── skillspan -> ../../Research/data/skillspan  (symlink)
│   ├── green_skill_extraction/     (18.6K entities, 5 NER types)
│   ├── jobs_information_extraction/ (2K JDs, years+certs NER)
│   ├── jd2skills/                  (20K structured postings)
│   ├── skill_extraction_benchmark/ (3.9K ESCO-linked sentences)
│   ├── google_job_skills/          (1.2K with min/preferred quals)
│   ├── djinni_jds/                 (142K JDs, parquet)
│   └── sayfullina_soft_skills/     (7.4K soft skill BIO tags)
├── model2_resume_extractor/
│   ├── yashpwr_resume_ner/         (22.8K samples, 25 entity types)
│   ├── dataturks_resume_ner/       (220 annotated resumes)
│   ├── mehyaar_ner_cvs/            (5K annotated CVs)
│   ├── datasetmaster_resumes/      (structured JSON resumes)
│   └── djinni_candidates/          (230K candidate profiles)
├── model3_skills_comparator/
│   ├── techwolf_esco_sentences/    (138K skill-sentence pairs)
│   ├── mind_tech_ontology/         (19 tech category relationships)
│   ├── tabiya_esco/                (ESCO in tabular format)
│   ├── nesta_skills_taxonomy/      (10.5K skills, 143 clusters)
│   ├── stacklite/                  (SO tag co-occurrence)
│   ├── related_job_skills/         (skill-to-skill pairs)
│   └── job_skill_set/              (skills per job title)
├── model4_exp_edu_comparator/
│   ├── jobhop/                     (1.68M experiences, CC BY 4.0)
│   ├── karrierewege/               (100K sample, career paths)
│   ├── jobbert_evaluation/         (31K titles → ESCO)
│   ├── job_titles_dedup/           (65K standardized titles)
│   ├── job_titles_normalized/      (70K normalized titles)
│   └── job_classification/         (job levels + paygrades)
├── model5_judge/
│   ├── netsol_score_details/       (1K pairs, multi-dim scores)
│   ├── ats_score/                  (6.4K continuous ATS scores)
│   ├── resume_jd_fit/              (8K fit/no-fit labels)
│   ├── resume_atlas/               (13K resumes, 43 categories)
│   ├── ai_resume_screening/        (AI scores + predictions)
│   └── ai_recruitment_pipeline/    (1K+ screening outcomes)
└── model6_verdict/
    ├── mikepfunk_resume_critique/  (22.8K critique conversations)
    ├── grammarly_coedit/           (70K text editing pairs)
    ├── iterater_human_sent/        (4K before/after edits)
    └── open_rewrite_eval/          (1.6K rewrite pairs)
```

---

## Evaluation Strategy

### Per-Model Metrics

| Model | Primary Metric | Target |
|-------|---------------|--------|
| Model 1 | F1 score on NER entities | > 85% |
| Model 2 | F1 score on NER entities | > 90% (baseline: 90.87%) |
| Model 3 | Skill match accuracy vs ESCO ground truth | > 80% |
| Model 4 | Spearman correlation on experience/education scores | > 0.6 |
| Model 5 | Spearman correlation with human rankings | > 0.3 (current baseline), target > 0.5 |
| Model 6 | BLEU/ROUGE on bullet rewrites, human eval on feedback | Qualitative |

### End-to-End Evaluation

- **Vacancy-Resume dataset**: 30 CVs x 5 vacancies, 2 human annotators
  - Metric: Spearman correlation, NDCG@5
  - Target: Spearman > 0.5, NDCG@5 > 0.7
- **Real resume PDFs**: 60+ PDFs across 6 categories
  - Metric: Mean score, category discrimination
  - Baseline: mean=52.4, 62% above 50
- **A/B comparison**: New pipeline vs current pipeline on same inputs

---

## Build Order

**Phase 1: Extractors (Models 1 & 2)** — Foundation
- Most data-rich, clear NER task, existing models to build on
- Model 2 has head start (resume_ner.py already exists)

**Phase 2: Comparators (Models 3 & 4)** — Core Logic
- Depend on extractor outputs
- Model 3 uses taxonomy data, Model 4 uses career trajectory data

**Phase 3: Judge (Model 5)** — Scoring
- Depends on comparator outputs
- Needs Claude labeling for additional training pairs

**Phase 4: Verdict (Model 6)** — User-Facing Output
- Depends on all other models
- Can start with template-based, upgrade to fine-tuned LM

**Phase 5: Integration & Evaluation**
- Wire all 6 models into pipeline
- Run against vacancy-resume gold standard
- A/B compare with current pipeline

---

## Technical Requirements

### Already Available
- `transformers`, `sentence-transformers`, `scikit-learn`, `scipy`
- `TechWolf/JobBERT-v2`, `jjzha/jobbert_skill_extraction`, `yashpwr/resume-ner-bert-v2` (cached)

### To Add
```
torch>=2.0.0
accelerate>=0.25.0
```

### Hardware
- GPU: Local (consumer GPU)
- Training: Fine-tuning BERT-sized models (110M-340M params)
- Inference: All models must run on CPU for deployment

---

## Key Research References

- ConFit v1: https://arxiv.org/abs/2401.16349 (ACM RecSys 2024)
- ConFit v2: https://arxiv.org/abs/2502.12361 (ACL 2025 Findings)
- SkillSpan: https://arxiv.org/abs/2204.12811 (NAACL 2022)
- ESCOXLM-R: https://aclanthology.org/2023.acl-long.901 (ACL 2023)
- JobHop: https://arxiv.org/abs/2505.07653
- Karrierewege: https://arxiv.org/abs/2412.14612
- CMap: https://www.nature.com/articles/s41597-025-05526-3

## How to Run Existing Tests

```bash
cd resume-brain/backend

# Unit tests (fast, ~1.5 min)
.venv/bin/python -m pytest tests/ -v --ignore=tests/test_vacancy_resume_eval.py --ignore=tests/test_real_resumes_eval.py --ignore=tests/test_kaggle_category_eval.py

# Integration test (loads real NER model, ~30s)
.venv/bin/python -m pytest -m integration -v -s

# Evaluation tests (slow on CPU, run separately)
.venv/bin/python -m pytest -m evaluation -v -s tests/test_real_resumes_eval.py      # ~15 min
.venv/bin/python -m pytest -m evaluation -v -s tests/test_kaggle_category_eval.py   # ~40 min
.venv/bin/python -m pytest -m evaluation -v -s tests/test_vacancy_resume_eval.py    # ~2.5 hrs
```

## Baseline Commit

`fe023c2` — Phases 3-5 complete (NER, scoring recalibration, eval infra). Return to this if anything breaks:
```bash
git reset --hard fe023c2
```
