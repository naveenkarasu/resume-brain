#!/usr/bin/env python3
"""Download all training datasets for the 6-model pipeline.

Usage: .venv/bin/python training/download_datasets.py
"""

import os
import subprocess
import sys
from pathlib import Path

BASE = Path("/home/kappa/Desktop/ai-log-investigator/resume-brain/training/data")

# Ensure HF datasets library is available
try:
    from datasets import load_dataset
    print("[OK] datasets library available")
except ImportError:
    print("[ERROR] datasets library not found")
    sys.exit(1)


def save_hf_dataset(name, outdir, trust_remote_code=False):
    """Download a HuggingFace dataset and save as parquet."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded
    existing = list(outdir.glob("*.parquet"))
    if existing:
        print(f"  [SKIP] {name} already downloaded ({len(existing)} parquet files)")
        return True

    try:
        print(f"  [DL] {name} ...")
        kwargs = {}
        if trust_remote_code:
            kwargs["trust_remote_code"] = True
        ds = load_dataset(name, **kwargs)
        print(f"  [OK] {ds}")

        if hasattr(ds, 'keys'):
            # DatasetDict with splits
            for split in ds:
                path = outdir / f"{split}.parquet"
                ds[split].to_parquet(str(path))
                size_mb = os.path.getsize(path) / (1024 * 1024)
                print(f"  [SAVED] {split}.parquet ({size_mb:.1f} MB)")
        else:
            # Single dataset
            path = outdir / "data.parquet"
            ds.to_parquet(str(path))
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"  [SAVED] data.parquet ({size_mb:.1f} MB)")
        return True
    except Exception as e:
        print(f"  [ERROR] {name}: {e}")
        return False


def git_clone(url, outdir):
    """Clone a git repo if not already cloned."""
    outdir = Path(outdir)
    if outdir.exists() and any(outdir.iterdir()):
        print(f"  [SKIP] {url} already cloned to {outdir.name}")
        return True

    outdir.mkdir(parents=True, exist_ok=True)
    try:
        print(f"  [CLONE] {url} ...")
        subprocess.run(
            ["git", "clone", "--depth=1", url, str(outdir)],
            check=True, capture_output=True, text=True, timeout=120
        )
        print(f"  [OK] Cloned to {outdir.name}")
        return True
    except Exception as e:
        print(f"  [ERROR] Clone {url}: {e}")
        return False


def kaggle_download(dataset_slug, outdir):
    """Download a Kaggle dataset."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded
    existing = list(outdir.glob("*"))
    if any(f.suffix in ('.csv', '.json', '.parquet', '.zip') for f in existing):
        print(f"  [SKIP] {dataset_slug} already downloaded")
        return True

    try:
        print(f"  [DL] Kaggle: {dataset_slug} ...")
        kaggle_cli = "/home/kappa/Desktop/ai-log-investigator/resume-brain/backend/.venv/bin/kaggle"
        env = os.environ.copy()
        env["KAGGLE_API_TOKEN"] = "KGAT_29fcf746e043fa066c6d81d60105d487"
        subprocess.run(
            [kaggle_cli, "datasets", "download", "-d", dataset_slug, "-p", str(outdir), "--unzip"],
            check=True, capture_output=True, text=True, timeout=300, env=env
        )
        print(f"  [OK] Downloaded {dataset_slug}")
        return True
    except Exception as e:
        print(f"  [ERROR] Kaggle {dataset_slug}: {e}")
        return False


def symlink(src, dst):
    """Create a symlink."""
    dst = Path(dst)
    src = Path(src)
    if dst.exists():
        print(f"  [SKIP] Symlink {dst.name} already exists")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(src, dst)
    print(f"  [OK] Symlinked {dst.name} -> {src}")


def main():
    results = {}

    # ========== MODEL 1: JD EXTRACTOR ==========
    print("\n" + "=" * 60)
    print("MODEL 1: JD EXTRACTOR")
    print("=" * 60)
    m1 = BASE / "model1_jd_extractor"

    # SkillSpan (already in Research/)
    skillspan_src = Path("/home/kappa/Desktop/ai-log-investigator/resume-brain/Research/data/skillspan")
    if skillspan_src.exists():
        symlink(skillspan_src, m1 / "skillspan")

    results["m1_green"] = git_clone(
        "https://github.com/acp19tag/skill-extraction-dataset",
        m1 / "green_skill_extraction"
    )
    results["m1_jobs_ie"] = git_clone(
        "https://github.com/ahmednabil950/JOBS-Information-Extraction",
        m1 / "jobs_information_extraction"
    )
    results["m1_jd2skills"] = git_clone(
        "https://github.com/WING-NUS/JD2Skills-BERT-XMLC",
        m1 / "jd2skills"
    )
    results["m1_skill_bench"] = git_clone(
        "https://github.com/jensjorisdecorte/Skill-Extraction-benchmark",
        m1 / "skill_extraction_benchmark"
    )
    results["m1_google"] = kaggle_download("niyamatalmass/google-job-skills", m1 / "google_job_skills")
    results["m1_djinni"] = save_hf_dataset(
        "lang-uk/recruitment-dataset-job-descriptions-english",
        m1 / "djinni_jds"
    )
    results["m1_sayfullina"] = save_hf_dataset("jjzha/sayfullina", m1 / "sayfullina_soft_skills")

    # ========== MODEL 2: RESUME EXTRACTOR ==========
    print("\n" + "=" * 60)
    print("MODEL 2: RESUME EXTRACTOR")
    print("=" * 60)
    m2 = BASE / "model2_resume_extractor"

    results["m2_yashpwr"] = save_hf_dataset("yashpwr/resume-ner-training-data", m2 / "yashpwr_resume_ner")
    results["m2_dataturks"] = kaggle_download("dataturks/resume-entities-for-ner", m2 / "dataturks_resume_ner")
    results["m2_mehyaar"] = kaggle_download("mehyarmlaweh/ner-annotated-cvs", m2 / "mehyaar_ner_cvs")
    results["m2_datasetmaster"] = save_hf_dataset("datasetmaster/resumes", m2 / "datasetmaster_resumes")
    results["m2_djinni"] = save_hf_dataset(
        "lang-uk/recruitment-dataset-candidate-profiles-english",
        m2 / "djinni_candidates"
    )

    # ========== MODEL 3: SKILLS COMPARATOR ==========
    print("\n" + "=" * 60)
    print("MODEL 3: SKILLS COMPARATOR")
    print("=" * 60)
    m3 = BASE / "model3_skills_comparator"

    results["m3_techwolf"] = save_hf_dataset(
        "TechWolf/Synthetic-ESCO-skill-sentences",
        m3 / "techwolf_esco_sentences"
    )
    results["m3_mind"] = git_clone(
        "https://github.com/MIND-TechAI/MIND-tech-ontology",
        m3 / "mind_tech_ontology"
    )
    results["m3_tabiya"] = git_clone(
        "https://github.com/tabiya-tech/tabiya-open-dataset",
        m3 / "tabiya_esco"
    )
    results["m3_nesta"] = git_clone(
        "https://github.com/nestauk/skills-taxonomy-v2",
        m3 / "nesta_skills_taxonomy"
    )
    results["m3_stacklite"] = git_clone(
        "https://github.com/dgrtwo/StackLite",
        m3 / "stacklite"
    )
    results["m3_related"] = kaggle_download("ulrikthygepedersen/related-job-skills", m3 / "related_job_skills")
    results["m3_jobskillset"] = kaggle_download("batuhanmutlu/job-skill-set", m3 / "job_skill_set")

    # ========== MODEL 4: EXP/EDU COMPARATOR ==========
    print("\n" + "=" * 60)
    print("MODEL 4: EXP/EDU COMPARATOR")
    print("=" * 60)
    m4 = BASE / "model4_exp_edu_comparator"

    results["m4_jobhop"] = save_hf_dataset("aida-ugent/JobHop", m4 / "jobhop")
    results["m4_karrierewege"] = save_hf_dataset("ElenaSenger/Karrierewege", m4 / "karrierewege")
    results["m4_jobbert_eval"] = git_clone(
        "https://github.com/jensjorisdecorte/JobBERT-evaluation-dataset",
        m4 / "jobbert_evaluation"
    )
    results["m4_jobtitles"] = save_hf_dataset("gpriday/job-titles", m4 / "job_titles_dedup")
    results["m4_jobtitles_norm"] = git_clone(
        "https://github.com/jneidel/job-titles",
        m4 / "job_titles_normalized"
    )
    results["m4_jobclass"] = kaggle_download(
        "HRAnalyticRepository/job-classification-dataset",
        m4 / "job_classification"
    )

    # ========== MODEL 5: JUDGE ==========
    print("\n" + "=" * 60)
    print("MODEL 5: JUDGE")
    print("=" * 60)
    m5 = BASE / "model5_judge"

    results["m5_netsol"] = save_hf_dataset("netsol/resume-score-details", m5 / "netsol_score_details")
    results["m5_ats"] = save_hf_dataset("0xnbk/resume-ats-score-v1-en", m5 / "ats_score")
    results["m5_fit"] = save_hf_dataset(
        "cnamuangtoun/resume-job-description-fit",
        m5 / "resume_jd_fit"
    )
    results["m5_atlas"] = save_hf_dataset("ahmedheakl/resume-atlas", m5 / "resume_atlas")
    results["m5_screening"] = kaggle_download(
        "mdtalhask/ai-powered-resume-screening-dataset-2025",
        m5 / "ai_resume_screening"
    )
    results["m5_recruitment"] = kaggle_download(
        "yaswanthkumary/ai-recruitment-pipeline-dataset",
        m5 / "ai_recruitment_pipeline"
    )

    # ========== MODEL 6: VERDICT ==========
    print("\n" + "=" * 60)
    print("MODEL 6: VERDICT")
    print("=" * 60)
    m6 = BASE / "model6_verdict"

    results["m6_mikepfunk"] = save_hf_dataset(
        "MikePfunk28/resume-training-dataset",
        m6 / "mikepfunk_resume_critique"
    )
    results["m6_coedit"] = save_hf_dataset("grammarly/coedit", m6 / "grammarly_coedit")
    results["m6_iterater"] = save_hf_dataset("wanyu/IteraTeR_human_sent", m6 / "iterater_human_sent")
    results["m6_rewriteeval"] = save_hf_dataset(
        "gabrielmbmb/OpenRewriteEval",
        m6 / "open_rewrite_eval"
    )

    # ========== SUMMARY ==========
    print("\n" + "=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)

    success = sum(1 for v in results.values() if v)
    total = len(results)
    print(f"\n{success}/{total} downloads succeeded\n")

    for key, ok in sorted(results.items()):
        status = "OK" if ok else "FAILED"
        print(f"  [{status}] {key}")

    # Print total disk usage
    print("\nDisk usage per model:")
    for d in sorted(BASE.iterdir()):
        if d.is_dir():
            size = subprocess.run(
                ["du", "-sh", str(d)], capture_output=True, text=True
            ).stdout.strip()
            print(f"  {size}")


if __name__ == "__main__":
    main()
