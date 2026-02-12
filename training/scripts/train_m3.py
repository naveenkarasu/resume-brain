"""Train Model 3: Skills Comparator (contrastive learning on JobBERT-v2).

Trains a 256-dim projection head on top of JobBERT-v2 using
MultipleNegativesRankingLoss with hard negatives from ESCO/O*NET.

Usage:
    python training/scripts/train_m3.py [--config training/configs/m3_skills_comparator.yaml]
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main(config_path: str = "training/configs/m3_skills_comparator.yaml") -> None:
    config = load_config(config_path)
    logger.info("Training M3 Skills Comparator with config: %s", config["model"]["name"])

    # --- 1. Load triplet data ---
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from data_prep.m3_data import (
        load_esco_sentences,
        load_tabiya_synonyms,
        load_nesta_clusters,
        load_stacklite_cooccurrence,
        build_triplets,
        create_hard_negatives,
    )

    logger.info("Loading skill data sources...")
    esco_pairs = load_esco_sentences()
    synonyms = load_tabiya_synonyms()
    clusters = load_nesta_clusters()
    cooccur = load_stacklite_cooccurrence()

    logger.info("Building skill triplets...")
    triplets = build_triplets(
        esco_pairs, synonyms, clusters, cooccur,
        max_triplets=config["data"].get("max_triplets", 500000),
    )
    triplets = create_hard_negatives(triplets, cooccur)
    logger.info("Total triplets: %d", len(triplets))

    # --- 2. Set up sentence-transformers ---
    from sentence_transformers import SentenceTransformer, InputExample, losses
    from sentence_transformers.evaluation import TripletEvaluator
    from torch.utils.data import DataLoader

    model = SentenceTransformer(config["model"]["base"])

    # Add projection head (dense layer reducing to 256-dim)
    from sentence_transformers import models
    projection = models.Dense(
        in_features=model.get_sentence_embedding_dimension(),
        out_features=config["model"]["projection_dim"],
        activation_function=None,
    )
    model.add_module("projection", projection)

    # --- 3. Prepare data ---
    # Split triplets: 90% train, 10% eval
    np.random.shuffle(triplets)
    split = int(len(triplets) * 0.9)
    train_triplets = triplets[:split]
    eval_triplets = triplets[split:]

    train_examples = [
        InputExample(texts=[t["anchor"], t["positive"], t["negative"]])
        for t in train_triplets
    ]
    train_dataloader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=config["training"]["batch_size"],
    )

    # Loss
    train_loss = losses.MultipleNegativesRankingLoss(model)

    # Evaluator
    evaluator = TripletEvaluator(
        anchors=[t["anchor"] for t in eval_triplets],
        positives=[t["positive"] for t in eval_triplets],
        negatives=[t["negative"] for t in eval_triplets],
        name="skill_triplet_eval",
    )

    # --- 4. Training ---
    output_dir = config["output"]["model_dir"]
    warmup_steps = int(
        len(train_dataloader) * config["training"]["epochs"] * config["training"]["warmup_ratio"]
    )

    logger.info("Starting contrastive training...")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=config["training"]["epochs"],
        warmup_steps=warmup_steps,
        output_path=output_dir,
        evaluation_steps=len(train_dataloader) // 2,
        save_best_model=True,
        use_amp=config["training"].get("fp16", False),
    )

    # --- 5. Export projection weights for lightweight inference ---
    logger.info("Exporting projection head weights...")
    projection_weights = {}
    for name, param in model.named_parameters():
        if "projection" in name:
            projection_weights[name.split(".")[-1]] = param.detach().cpu().numpy()

    projection_path = Path(output_dir) / "projection.npy"
    np.save(str(projection_path), projection_weights)
    logger.info("Projection weights saved to %s", projection_path)

    # --- 6. Evaluate on ESCO synonym test set ---
    logger.info("Evaluating on ESCO synonym pairs...")
    eval_score = evaluator(model, output_path=output_dir)
    target = config["evaluation"]["target"]
    if eval_score >= target:
        logger.info("Target accuracy %.2f ACHIEVED (got %.4f)", target, eval_score)
    else:
        logger.warning("Target accuracy %.2f NOT MET (got %.4f)", target, eval_score)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train M3 Skills Comparator")
    parser.add_argument("--config", default="training/configs/m3_skills_comparator.yaml")
    args = parser.parse_args()
    main(args.config)
