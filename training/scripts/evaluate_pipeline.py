"""End-to-end evaluation: compare v2 pipeline vs legacy pipeline.

Runs both pipelines on the same inputs and compares:
- Spearman correlation with gold standard scores
- Score distributions
- Latency

Usage:
    python training/scripts/evaluate_pipeline.py [--data-dir tests/fixtures]
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

# Add backend to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "backend"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


async def run_pipeline(
    resume_text: str,
    jd_text: str,
    mode: str,
) -> tuple[dict, float]:
    """Run a pipeline and return (result_dict, latency_seconds)."""
    os.environ["PIPELINE_MODE"] = mode

    # Re-import to pick up mode change
    from config import Settings
    import config
    config.settings = Settings()

    from services.resume_analyzer import analyze

    start = time.perf_counter()
    result = await analyze(resume_text, jd_text)
    elapsed = time.perf_counter() - start

    return result.model_dump(), elapsed


async def main(data_dir: str = "tests/fixtures") -> None:
    data_path = Path(data_dir)

    # Load test pairs: list of {resume_text, jd_text, gold_score}
    pairs_file = data_path / "eval_pairs.json"
    if not pairs_file.exists():
        logger.error("Evaluation pairs not found at %s", pairs_file)
        logger.info("Create a JSON file with format: [{resume_text, jd_text, gold_score}, ...]")
        return

    with open(pairs_file) as f:
        pairs = json.load(f)

    logger.info("Loaded %d evaluation pairs", len(pairs))

    legacy_scores = []
    v2_scores = []
    gold_scores = []
    legacy_times = []
    v2_times = []

    for i, pair in enumerate(pairs):
        logger.info("Evaluating pair %d/%d...", i + 1, len(pairs))
        resume = pair["resume_text"]
        jd = pair["jd_text"]
        gold = pair.get("gold_score", None)

        # Run legacy
        try:
            legacy_result, legacy_time = await run_pipeline(resume, jd, "legacy")
            legacy_scores.append(legacy_result["overall_score"])
            legacy_times.append(legacy_time)
        except Exception as e:
            logger.warning("Legacy pipeline failed on pair %d: %s", i, e)
            legacy_scores.append(None)
            legacy_times.append(None)

        # Run v2
        try:
            v2_result, v2_time = await run_pipeline(resume, jd, "v2")
            v2_scores.append(v2_result["overall_score"])
            v2_times.append(v2_time)
        except Exception as e:
            logger.warning("V2 pipeline failed on pair %d: %s", i, e)
            v2_scores.append(None)
            v2_times.append(None)

        if gold is not None:
            gold_scores.append(gold)

    # --- Report ---
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 60)

    # Filter out failures
    valid = [
        i for i in range(len(pairs))
        if legacy_scores[i] is not None and v2_scores[i] is not None
    ]
    logger.info("Valid pairs: %d / %d", len(valid), len(pairs))

    l_scores = np.array([legacy_scores[i] for i in valid])
    v_scores = np.array([v2_scores[i] for i in valid])
    l_times = [legacy_times[i] for i in valid if legacy_times[i] is not None]
    v_times = [v2_times[i] for i in valid if v2_times[i] is not None]

    # Score statistics
    logger.info("\nScore Statistics:")
    logger.info("  Legacy - mean: %.1f, std: %.1f, min: %d, max: %d",
                l_scores.mean(), l_scores.std(), l_scores.min(), l_scores.max())
    logger.info("  V2     - mean: %.1f, std: %.1f, min: %d, max: %d",
                v_scores.mean(), v_scores.std(), v_scores.min(), v_scores.max())

    # Correlation between pipelines
    rho, pval = spearmanr(l_scores, v_scores)
    logger.info("\nPipeline Agreement:")
    logger.info("  Spearman(legacy, v2) = %.4f (p=%.6f)", rho, pval)
    logger.info("  Mean absolute diff = %.1f", np.mean(np.abs(l_scores - v_scores)))

    # Gold standard comparison
    if gold_scores:
        g_scores = np.array([gold_scores[i] for i in valid if i < len(gold_scores)])
        if len(g_scores) == len(valid):
            rho_legacy, _ = spearmanr(g_scores, l_scores)
            rho_v2, _ = spearmanr(g_scores, v_scores)
            logger.info("\nGold Standard Correlation:")
            logger.info("  Spearman(gold, legacy) = %.4f", rho_legacy)
            logger.info("  Spearman(gold, v2)     = %.4f", rho_v2)

    # Latency
    logger.info("\nLatency:")
    if l_times:
        logger.info("  Legacy - mean: %.2fs, p95: %.2fs",
                    np.mean(l_times), np.percentile(l_times, 95))
    if v_times:
        logger.info("  V2     - mean: %.2fs, p95: %.2fs",
                    np.mean(v_times), np.percentile(v_times, 95))

    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate pipeline A/B comparison")
    parser.add_argument("--data-dir", default="tests/fixtures")
    args = parser.parse_args()
    asyncio.run(main(args.data_dir))
