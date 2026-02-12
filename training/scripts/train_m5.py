"""Train Model 5: Judge (LightGBM scoring model).

Combines M3 + M4 signals into a single 0-100 overall score.
Uses 13 features from skills comparison and experience/education comparison.

Usage:
    python training/scripts/train_m5.py [--config training/configs/m5_judge.yaml]
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import yaml
from scipy.stats import spearmanr

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def ndcg_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int = 5) -> float:
    """Compute NDCG@k for ranking evaluation."""
    # Group by query (assume all items are in same query for now)
    order = np.argsort(-y_pred)[:k]
    dcg = np.sum(y_true[order] / np.log2(np.arange(2, k + 2)))
    ideal_order = np.argsort(-y_true)[:k]
    idcg = np.sum(y_true[ideal_order] / np.log2(np.arange(2, k + 2)))
    return float(dcg / idcg) if idcg > 0 else 0.0


def main(config_path: str = "training/configs/m5_judge.yaml") -> None:
    config = load_config(config_path)
    logger.info("Training M5 Judge with config: %s", config["model"]["name"])

    # --- 1. Load data ---
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from data_prep.m5_data import (
        load_netsol, load_ats_score, load_resume_jd_fit,
        normalize_scores, build_dataset, FEATURE_NAMES,
    )

    logger.info("Loading scoring data sources...")
    all_records = []
    for name, loader in [("netsol", load_netsol), ("ats_score", load_ats_score), ("resume_jd_fit", load_resume_jd_fit)]:
        try:
            records = loader()
            all_records.extend(records)
            logger.info("[%s] contributed %d records (total: %d).", name, len(records), len(all_records))
        except Exception:
            logger.exception("Loader '%s' raised an error -- skipping.", name)

    if not all_records:
        logger.error("No records produced -- aborting.")
        return

    all_records = normalize_scores(all_records)
    logger.info("After normalization: %d records.", len(all_records))

    dd = build_dataset(all_records)
    feature_names = FEATURE_NAMES

    # Convert DatasetDict to numpy arrays
    def _to_numpy(split):
        X = np.column_stack([np.array(split[f], dtype=np.float64) for f in feature_names])
        y = np.array(split["score"], dtype=np.float64)
        return X, y

    X_train, y_train = _to_numpy(dd["train"])
    X_val, y_val = _to_numpy(dd["validation"])
    X_test, y_test = _to_numpy(dd["test"])
    logger.info(
        "Train: %d, Val: %d, Test: %d, Features: %d",
        len(X_train), len(X_val), len(X_test), len(feature_names),
    )

    # --- 2. Train LightGBM ---
    import lightgbm as lgb

    output_dir = Path(config["output"]["model_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    params = {
        "objective": "regression",
        "metric": config["training"]["metric"],
        "learning_rate": config["training"]["learning_rate"],
        "max_depth": config["training"]["max_depth"],
        "num_leaves": config["training"]["num_leaves"],
        "subsample": config["training"]["subsample"],
        "colsample_bytree": config["training"]["colsample_bytree"],
        "verbose": -1,
    }

    callbacks = [
        lgb.early_stopping(config["training"]["early_stopping_rounds"]),
        lgb.log_evaluation(config["training"]["verbose"]),
    ]

    booster = lgb.train(
        params,
        train_data,
        num_boost_round=config["training"]["num_estimators"],
        valid_sets=[val_data],
        callbacks=callbacks,
    )

    # Save model
    model_path = output_dir / "model.txt"
    booster.save_model(str(model_path))
    logger.info("Model saved to %s", model_path)

    # --- 3. Evaluate ---
    logger.info("Evaluating on test set...")
    preds = booster.predict(X_test)

    # Spearman correlation
    rho, pval = spearmanr(y_test, preds)
    logger.info("Spearman correlation: %.4f (p=%.6f)", rho, pval)

    # NDCG@5
    ndcg5 = ndcg_at_k(y_test, preds, k=5)
    logger.info("NDCG@5: %.4f", ndcg5)

    # RMSE
    rmse = np.sqrt(np.mean((y_test - preds) ** 2))
    logger.info("RMSE: %.4f", rmse)

    # Check targets
    targets = config["evaluation"]["targets"]
    if rho >= targets["spearman"]:
        logger.info("Spearman target %.2f ACHIEVED", targets["spearman"])
    else:
        logger.warning("Spearman target %.2f NOT MET (got %.4f)", targets["spearman"], rho)

    if ndcg5 >= targets["ndcg_at_5"]:
        logger.info("NDCG@5 target %.2f ACHIEVED", targets["ndcg_at_5"])
    else:
        logger.warning("NDCG@5 target %.2f NOT MET (got %.4f)", targets["ndcg_at_5"], ndcg5)

    # Feature importance
    logger.info("Feature importances (gain):")
    imp = booster.feature_importance(importance_type="gain")
    sorted_imp = sorted(zip(feature_names, imp), key=lambda x: -x[1])
    for name, val in sorted_imp:
        logger.info("  %s: %.2f", name, val)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train M5 Judge")
    parser.add_argument("--config", default="training/configs/m5_judge.yaml")
    args = parser.parse_args()
    main(args.config)
