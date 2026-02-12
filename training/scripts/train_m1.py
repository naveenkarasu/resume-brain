"""Train Model 1: JD Extractor (BIO NER on job descriptions).

Usage:
    python training/scripts/train_m1.py [--config training/configs/m1_jd_extractor.yaml]
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


def main(config_path: str = "training/configs/m1_jd_extractor.yaml") -> None:
    config = load_config(config_path)
    logger.info("Training M1 JD Extractor with config: %s", config["model"]["name"])

    # --- 1. Load and prepare data ---
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from data_prep.m1_data import unify_all, split_dataset

    logger.info("Loading and unifying datasets...")
    dataset = unify_all()

    # Clean records: ensure tokens are strings, ner_tags are ints, source is string
    cleaned = []
    for rec in dataset:
        tokens = [str(t) if not isinstance(t, str) else t for t in rec.get("tokens", [])]
        tags = [int(t) if not isinstance(t, int) else t for t in rec.get("ner_tags", [])]
        if not tokens or len(tokens) != len(tags):
            continue
        # Skip records with NaN-like tokens
        if any(t in ("nan", "None", "") for t in tokens):
            continue
        cleaned.append({"tokens": tokens, "ner_tags": tags, "source": str(rec.get("source", "unknown"))})
    logger.info("Cleaned: %d -> %d records (dropped %d)", len(dataset), len(cleaned), len(dataset) - len(cleaned))
    dataset = cleaned

    dd = split_dataset(
        dataset,
        train_ratio=config["data"]["train_split"],
        val_ratio=config["data"]["val_split"],
    )
    train_ds, val_ds, test_ds = dd["train"], dd["validation"], dd["test"]
    logger.info("Train: %d, Val: %d, Test: %d", len(train_ds), len(val_ds), len(test_ds))

    # --- 2. Tokenize ---
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config["model"]["base"])

    def tokenize_and_align(examples):
        tokenized = tokenizer(
            examples["tokens"],
            truncation=True,
            is_split_into_words=True,
            max_length=config["data"]["max_length"],
            padding="max_length",
        )
        all_labels = []
        for i, labels in enumerate(examples["ner_tags"]):
            word_ids = tokenized.word_ids(batch_index=i)
            aligned = []
            previous_word_id = None
            for word_id in word_ids:
                if word_id is None:
                    aligned.append(-100)
                elif word_id != previous_word_id:
                    aligned.append(labels[word_id])
                else:
                    aligned.append(-100)
                previous_word_id = word_id
            all_labels.append(aligned)
        tokenized["labels"] = all_labels
        return tokenized

    train_tok = train_ds.map(tokenize_and_align, batched=True, remove_columns=train_ds.column_names)
    val_tok = val_ds.map(tokenize_and_align, batched=True, remove_columns=val_ds.column_names)
    test_tok = test_ds.map(tokenize_and_align, batched=True, remove_columns=test_ds.column_names)

    # --- 3. Model ---
    from transformers import AutoModelForTokenClassification

    # Import label list from the inference service
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "backend"))
    from services.pipeline.m1_jd_extractor import ID2LABEL, LABEL2ID

    model = AutoModelForTokenClassification.from_pretrained(
        config["model"]["base"],
        num_labels=config["model"]["num_labels"],
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    # --- 4. Metrics ---
    from seqeval.metrics import f1_score, precision_score, recall_score, classification_report

    label_list = list(ID2LABEL.values())

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=2)

        true_labels = []
        true_preds = []
        for pred_seq, label_seq in zip(predictions, labels):
            t_labels = []
            t_preds = []
            for p, l in zip(pred_seq, label_seq):
                if l == -100:
                    continue
                t_labels.append(label_list[l])
                t_preds.append(label_list[p])
            true_labels.append(t_labels)
            true_preds.append(t_preds)

        return {
            "precision": precision_score(true_labels, true_preds),
            "recall": recall_score(true_labels, true_preds),
            "f1": f1_score(true_labels, true_preds),
        }

    # --- 5. Training ---
    from transformers import TrainingArguments, Trainer, EarlyStoppingCallback

    output_dir = config["output"]["model_dir"]
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=config["training"]["learning_rate"],
        per_device_train_batch_size=config["training"]["batch_size"],
        per_device_eval_batch_size=config["training"]["batch_size"],
        num_train_epochs=config["training"]["epochs"],
        weight_decay=config["training"]["weight_decay"],
        warmup_ratio=config["training"]["warmup_ratio"],
        fp16=config["training"].get("fp16", False),
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_dir=config["output"]["log_dir"],
        logging_steps=50,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        compute_metrics=compute_metrics,
        # No early stopping -- run all epochs, best model saved via load_best_model_at_end
    )

    logger.info("Starting training...")
    trainer.train()

    # --- 6. Evaluation ---
    logger.info("Evaluating on test set...")
    results = trainer.evaluate(test_tok)
    logger.info("Test results: %s", results)

    test_f1 = results.get("eval_f1", 0)
    target = config["evaluation"]["target"]
    if test_f1 >= target:
        logger.info("Target F1 %.2f ACHIEVED (got %.4f)", target, test_f1)
    else:
        logger.warning("Target F1 %.2f NOT MET (got %.4f)", target, test_f1)

    # --- 7. Save ---
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info("Model saved to %s", output_dir)

    # Full classification report
    predictions = trainer.predict(test_tok)
    preds = np.argmax(predictions.predictions, axis=2)
    true_labels = []
    true_preds = []
    for pred_seq, label_seq in zip(preds, predictions.label_ids):
        t_labels = []
        t_preds = []
        for p, l in zip(pred_seq, label_seq):
            if l == -100:
                continue
            t_labels.append(label_list[l])
            t_preds.append(label_list[p])
        true_labels.append(t_labels)
        true_preds.append(t_preds)

    report = classification_report(true_labels, true_preds)
    logger.info("Classification Report:\n%s", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train M1 JD Extractor")
    parser.add_argument("--config", default="training/configs/m1_jd_extractor.yaml")
    args = parser.parse_args()
    main(args.config)
