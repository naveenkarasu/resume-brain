"""Train Model 2: Resume Extractor (BIO NER on resumes).

Fine-tunes yashpwr/resume-ner-bert-v2 with additional entity types.
Freezes layers 0-8 for first 2 epochs to preserve pretrained knowledge.

Usage:
    python training/scripts/train_m2.py [--config training/configs/m2_resume_extractor.yaml]
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


def freeze_bert_layers(model, layer_indices: list[int]) -> None:
    """Freeze specific BERT encoder layers."""
    for idx in layer_indices:
        for param in model.bert.encoder.layer[idx].parameters():
            param.requires_grad = False
    logger.info("Froze BERT layers: %s", layer_indices)


def unfreeze_all(model) -> None:
    """Unfreeze all parameters."""
    for param in model.parameters():
        param.requires_grad = True
    logger.info("Unfroze all layers")


def main(config_path: str = "training/configs/m2_resume_extractor.yaml") -> None:
    config = load_config(config_path)
    logger.info("Training M2 Resume Extractor with config: %s", config["model"]["name"])

    # --- 1. Load and prepare data ---
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from data_prep.m2_data import unify_all, split_dataset

    logger.info("Loading and unifying datasets...")
    dataset = unify_all()

    # Clean records: ensure tokens are strings, ner_tags are ints, source is string
    cleaned = []
    for rec in dataset:
        tokens = [str(t) if not isinstance(t, str) else t for t in rec.get("tokens", [])]
        tags = [int(t) if not isinstance(t, int) else t for t in rec.get("ner_tags", [])]
        if not tokens or len(tokens) != len(tags):
            continue
        if any(t in ("nan", "None", "") for t in tokens):
            continue
        # Strip surrogate characters that break Arrow/UTF-8
        tokens = [t.encode("utf-8", errors="replace").decode("utf-8") for t in tokens]
        source = str(rec.get("source", "unknown")).encode("utf-8", errors="replace").decode("utf-8")
        cleaned.append({"tokens": tokens, "ner_tags": tags, "source": source})
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

    # 14 entity types: original 11 + CERTIFICATION, PROJECT_NAME, PROJECT_TECHNOLOGY
    label_list = [
        "O",
        "B-NAME", "I-NAME",
        "B-EMAIL", "I-EMAIL",
        "B-PHONE", "I-PHONE",
        "B-SKILLS", "I-SKILLS",
        "B-COMPANIES", "I-COMPANIES",
        "B-DESIGNATION", "I-DESIGNATION",
        "B-COLLEGE", "I-COLLEGE",
        "B-GRADUATION_YEAR", "I-GRADUATION_YEAR",
        "B-LOCATION", "I-LOCATION",
        "B-YEARS_OF_EXPERIENCE", "I-YEARS_OF_EXPERIENCE",
        "B-DEGREE", "I-DEGREE",
        "B-CERTIFICATION", "I-CERTIFICATION",
        "B-PROJECT_NAME", "I-PROJECT_NAME",
        "B-PROJECT_TECHNOLOGY", "I-PROJECT_TECHNOLOGY",
    ]
    id2label = {i: l for i, l in enumerate(label_list)}
    label2id = {l: i for i, l in enumerate(label_list)}

    model = AutoModelForTokenClassification.from_pretrained(
        config["model"]["base"],
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,  # classifier head size changes from 11 to 14
    )

    # --- 4. Metrics ---
    from seqeval.metrics import f1_score, precision_score, recall_score, classification_report

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

    # --- 5. Training (two-phase: frozen then unfrozen) ---
    from transformers import TrainingArguments, Trainer, EarlyStoppingCallback

    output_dir = config["output"]["model_dir"]
    freeze_layers = config["training"].get("freeze_layers", [])
    freeze_epochs = config["training"].get("freeze_epochs", 0)

    # Phase 1: Frozen lower layers
    if freeze_layers and freeze_epochs > 0:
        logger.info("Phase 1: Training with frozen layers %s for %d epochs", freeze_layers, freeze_epochs)
        freeze_bert_layers(model, freeze_layers)

        phase1_args = TrainingArguments(
            output_dir=output_dir + "/phase1",
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=config["training"]["learning_rate"],
            per_device_train_batch_size=config["training"]["batch_size"],
            per_device_eval_batch_size=config["training"]["batch_size"],
            num_train_epochs=freeze_epochs,
            weight_decay=config["training"]["weight_decay"],
            warmup_ratio=config["training"]["warmup_ratio"],
            fp16=config["training"].get("fp16", False),
            logging_steps=50,
            save_total_limit=1,
        )

        trainer = Trainer(
            model=model,
            args=phase1_args,
            train_dataset=train_tok,
            eval_dataset=val_tok,
            compute_metrics=compute_metrics,
        )
        trainer.train()

    # Phase 2: All layers unfrozen
    unfreeze_all(model)
    remaining_epochs = config["training"]["epochs"] - freeze_epochs

    logger.info("Phase 2: Training all layers for %d epochs", remaining_epochs)
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=config["training"]["learning_rate"] * 0.5,  # lower LR for fine-tuning
        per_device_train_batch_size=config["training"]["batch_size"],
        per_device_eval_batch_size=config["training"]["batch_size"],
        num_train_epochs=remaining_epochs,
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train M2 Resume Extractor")
    parser.add_argument("--config", default="training/configs/m2_resume_extractor.yaml")
    args = parser.parse_args()
    main(args.config)
