"""Train Model 6: Verdict (Tier 2 - FLAN-T5-base seq2seq).

Tier 1 (template-based) requires no training.
This script is for future Tier 2: fine-tuning google/flan-t5-base on
resume critique and rewrite data.

Usage:
    python training/scripts/train_m6.py [--config training/configs/m6_verdict.yaml]
"""

import argparse
import logging
import sys
from pathlib import Path

import yaml
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main(config_path: str = "training/configs/m6_verdict.yaml") -> None:
    config = load_config(config_path)
    logger.info("Training M6 Verdict (Tier 2) with config: %s", config["model"]["name"])

    if config["model"]["tier"] == 1:
        logger.info("Tier 1 (template-based) requires no training. Exiting.")
        return

    # --- 1. Load data ---
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from data_prep.m6_data import build_dataset

    logger.info("Building seq2seq dataset...")
    dataset = build_dataset()
    train_ds = dataset["train"]
    val_ds = dataset["validation"]
    test_ds = dataset["test"]
    logger.info("Train: %d, Val: %d, Test: %d", len(train_ds), len(val_ds), len(test_ds))

    # --- 2. Tokenize ---
    from transformers import AutoTokenizer

    base_model = config["model"]["tier2_base"]
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    max_input = config["data"]["max_length"]
    max_target = config["data"]["target_max_length"]

    def tokenize_fn(examples):
        model_inputs = tokenizer(
            examples["input"],
            max_length=max_input,
            truncation=True,
            padding="max_length",
        )
        labels = tokenizer(
            examples["target"],
            max_length=max_target,
            truncation=True,
            padding="max_length",
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    train_tok = train_ds.map(tokenize_fn, batched=True, remove_columns=train_ds.column_names)
    val_tok = val_ds.map(tokenize_fn, batched=True, remove_columns=val_ds.column_names)

    # --- 3. Model ---
    from transformers import AutoModelForSeq2SeqLM

    model = AutoModelForSeq2SeqLM.from_pretrained(base_model)

    # --- 4. Training ---
    from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer

    output_dir = config["output"]["model_dir"]

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=config["training"]["learning_rate"],
        per_device_train_batch_size=config["training"]["batch_size"],
        per_device_eval_batch_size=config["training"]["batch_size"],
        num_train_epochs=config["training"]["epochs"],
        warmup_ratio=config["training"]["warmup_ratio"],
        fp16=config["training"].get("fp16", False),
        predict_with_generate=True,
        generation_max_length=max_target,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        save_total_limit=2,
        logging_dir=config["output"]["log_dir"],
        logging_steps=100,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
    )

    logger.info("Starting seq2seq training...")
    trainer.train()

    # --- 5. Save ---
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info("Model saved to %s", output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train M6 Verdict")
    parser.add_argument("--config", default="training/configs/m6_verdict.yaml")
    args = parser.parse_args()
    main(args.config)
