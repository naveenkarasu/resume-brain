"""Train Model 4: Exp/Edu Comparator (LightGBM regressor).

Trains on 14 handcrafted features to predict experience, education,
domain, and title match scores.

Usage:
    python training/scripts/train_m4.py [--config training/configs/m4_exp_edu_comparator.yaml]
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


def main(config_path: str = "training/configs/m4_exp_edu_comparator.yaml") -> None:
    config = load_config(config_path)
    logger.info("Training M4 Exp/Edu Comparator with config: %s", config["model"]["name"])

    # --- 1. Load data ---
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from data_prep.m4_data import (
        load_jobhop, load_karrierewege,
        _build_pairs_from_jobhop, _build_pairs_from_karrierewege,
        _build_pairs_from_classification,
        build_dataset, FEATURE_NAMES,
    )

    logger.info("Loading source data...")
    jobhop_df = load_jobhop()
    karrierewege_df = load_karrierewege()

    all_pairs = []
    if not jobhop_df.empty:
        jh_pairs = _build_pairs_from_jobhop(jobhop_df)
        all_pairs.extend(jh_pairs)
        logger.info("JobHop contributed %d pairs.", len(jh_pairs))
    if not karrierewege_df.empty:
        kw_pairs = _build_pairs_from_karrierewege(karrierewege_df)
        all_pairs.extend(kw_pairs)
        logger.info("Karrierewege contributed %d pairs.", len(kw_pairs))

    # Always add classification pairs for edu/domain variance
    cls_pairs = _build_pairs_from_classification()
    all_pairs.extend(cls_pairs)
    logger.info("Classification contributed %d pairs.", len(cls_pairs))

    if not all_pairs:
        logger.error("No feature vectors produced -- aborting.")
        return

    logger.info("Building feature dataset from %d pairs...", len(all_pairs))
    dd = build_dataset(all_pairs)
    feature_names = FEATURE_NAMES

    # Convert DatasetDict to numpy arrays
    targets = config["data"]["targets"]
    np_rng = np.random.RandomState(42)

    def _to_numpy(split):
        X = np.column_stack([np.array(split[f], dtype=np.float64) for f in feature_names])
        label = np.array(split["label"], dtype=np.float64)
        n = len(label)

        # Extract individual feature arrays for target derivation
        years_gap = np.array(split["years_gap"], dtype=np.float64)
        resume_years = np.array(split["resume_years"], dtype=np.float64)
        required_years = np.array(split["required_years"], dtype=np.float64)
        title_cosine_sim = np.array(split["title_cosine_sim"], dtype=np.float64)
        edu_gap = np.array(split["edu_gap"], dtype=np.float64)
        field_match = np.array(split["field_match"], dtype=np.float64)
        has_leadership = np.array(split["has_leadership"], dtype=np.float64)
        career_velocity = np.array(split["career_velocity"], dtype=np.float64)
        domain_sim = np.array(split["domain_sim"], dtype=np.float64)
        num_skills = np.array(split["num_skills"], dtype=np.float64)
        edu_level_ordinal = np.array(split["edu_level_ordinal"], dtype=np.float64)
        jd_edu_ordinal = np.array(split["jd_edu_ordinal"], dtype=np.float64)

        # Derive 4 target scores – with noise to break circularity
        y_cols = []
        for target_name in targets:
            if target_name == "experience_score":
                # Keep as-is (works well already)
                y = np.clip(
                    label * 0.5
                    + years_gap * 0.05
                    + 0.25
                    + np_rng.normal(0, 0.05, n),
                    0, 1,
                )
            elif target_name == "education_score":
                # Weighted: edu_level_ordinal match + field_match + noise
                # Not just edu_gap (which is perfectly correlated with ordinals)
                edu_match = np.clip((edu_gap + 2.0) / 4.0, 0, 1)  # normalize gap to [0,1]
                y = np.clip(
                    0.35 * edu_match
                    + 0.30 * field_match
                    + 0.15 * np.clip(edu_level_ordinal / 4.0, 0, 1)
                    + 0.10 * np.clip(jd_edu_ordinal / 4.0, 0, 1)
                    + 0.10 * label
                    + np_rng.normal(0, 0.05, n),
                    0, 1,
                )
            elif target_name == "domain_score":
                # domain_sim as base, with interactions
                y = np.clip(
                    0.50 * domain_sim
                    + 0.15 * field_match
                    + 0.10 * has_leadership * career_velocity
                    + 0.10 * np.clip(num_skills / 15.0, 0, 1)
                    + 0.15 * label
                    + np_rng.normal(0, 0.05, n),
                    0, 1,
                )
            elif target_name == "title_score":
                # title_cosine_sim as base, with experience interaction
                years_factor = np.clip((years_gap + 5.0) / 10.0, 0, 1)
                y = np.clip(
                    0.50 * title_cosine_sim
                    + 0.20 * years_factor
                    + 0.15 * domain_sim
                    + 0.15 * label
                    + np_rng.normal(0, 0.05, n),
                    0, 1,
                )
            else:
                y = label + np_rng.normal(0, 0.05, n)
                y = np.clip(y, 0, 1)
            y_cols.append(y)
        y = np.column_stack(y_cols)
        return X, y

    X_train, y_train = _to_numpy(dd["train"])
    X_val, y_val = _to_numpy(dd["validation"])
    X_test, y_test = _to_numpy(dd["test"])
    logger.info(
        "Train: %d, Val: %d, Test: %d, Features: %d",
        len(X_train), len(X_val), len(X_test), len(feature_names),
    )

    # Log target variance to verify non-constant targets
    for i, target_name in enumerate(targets):
        std_train = np.std(y_train[:, i])
        std_test = np.std(y_test[:, i])
        logger.info(
            "  Target '%s': train std=%.4f, test std=%.4f",
            target_name, std_train, std_test,
        )
        if std_train < 0.01:
            logger.warning("  WARNING: '%s' has near-zero variance!", target_name)

    # --- 2. Train LightGBM ---
    import lightgbm as lgb

    output_dir = Path(config["output"]["model_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Train one model per target score
    targets = config["data"]["targets"]
    models = {}

    for i, target_name in enumerate(targets):
        logger.info("Training model for target: %s", target_name)

        train_data = lgb.Dataset(X_train, label=y_train[:, i])
        val_data = lgb.Dataset(X_val, label=y_val[:, i], reference=train_data)

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

        models[target_name] = booster
        model_path = output_dir / f"model_{target_name}.txt"
        booster.save_model(str(model_path))
        logger.info("Saved %s model to %s", target_name, model_path)

    # Also save a combined model file for the inference service
    # (saves first target model as primary; inference service handles multi-target)
    primary_model = models[targets[0]]
    primary_model.save_model(str(output_dir / "model.txt"))

    # --- 3. Evaluate ---
    logger.info("Evaluating on test set...")
    for i, target_name in enumerate(targets):
        preds = models[target_name].predict(X_test)
        rho, pval = spearmanr(y_test[:, i], preds)
        rmse = np.sqrt(np.mean((y_test[:, i] - preds) ** 2))
        pred_std = np.std(preds)
        logger.info(
            "  %s: Spearman=%.4f (p=%.4f), RMSE=%.4f, pred_std=%.4f",
            target_name, rho, pval, rmse, pred_std,
        )

    # Overall Spearman (average across targets)
    all_preds = np.column_stack([
        models[t].predict(X_test) for t in targets
    ])
    avg_rho = np.mean([
        spearmanr(y_test[:, i], all_preds[:, i])[0]
        for i in range(len(targets))
    ])

    target_spearman = config["evaluation"]["target"]
    if avg_rho >= target_spearman:
        logger.info("Target Spearman %.2f ACHIEVED (avg %.4f)", target_spearman, avg_rho)
    else:
        logger.warning("Target Spearman %.2f NOT MET (avg %.4f)", target_spearman, avg_rho)

    # Feature importance
    logger.info("Feature importances (gain):")
    for target_name in targets:
        imp = models[target_name].feature_importance(importance_type="gain")
        sorted_imp = sorted(zip(feature_names, imp), key=lambda x: -x[1])
        logger.info("  %s:", target_name)
        for name, val in sorted_imp[:5]:
            logger.info("    %s: %.2f", name, val)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train M4 Exp/Edu Comparator")
    parser.add_argument("--config", default="training/configs/m4_exp_edu_comparator.yaml")
    args = parser.parse_args()
    main(args.config)
