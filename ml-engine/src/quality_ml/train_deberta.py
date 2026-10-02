"""Fine-tune DeBERTa-v3-base on the cleaned QuRE dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from quality_ml.qure import clean_qure, split_qure

MODEL_NAME = "microsoft/deberta-v3-base"

LABEL2ID = {
    "defect": 0,
    "ok": 1,
}

ID2LABEL = {
    0: "defect",
    1: "ok",
}


def prepare_dataset(csv_path: Path):
    """Load, clean and split QuRE into train/validation/test datasets."""

    raw = pd.read_csv(csv_path)
    clean = clean_qure(raw)

    train_df, validation_df, test_df = split_qure(
        clean,
        test_size=0.10,
        validation_size=0.10,
        random_state=42,
    )

    def convert(df: pd.DataFrame) -> Dataset:
        result = df[["requirement", "defect"]].copy()
        result["labels"] = result["defect"].map(LABEL2ID)

        if result["labels"].isna().any():
            raise ValueError("Found an unknown QuRE label.")

        return Dataset.from_pandas(
            result[["requirement", "labels"]],
            preserve_index=False,
        )

    return (
        convert(train_df),
        convert(validation_df),
        convert(test_df),
        {
            "raw_rows": len(raw),
            "clean_rows": len(clean),
            "train_rows": len(train_df),
            "validation_rows": len(validation_df),
            "test_rows": len(test_df),
        },
    )


def tokenize_dataset(dataset, tokenizer, max_length: int):
    """Tokenize requirement text."""

    def tokenize(batch):
        return tokenizer(
            batch["requirement"],
            truncation=True,
            max_length=max_length,
        )

    return dataset.map(
        tokenize,
        batched=True,
        remove_columns=["requirement"],
    )


def compute_metrics(eval_prediction):
    """Calculate classification metrics."""

    predictions = eval_prediction.predictions
    labels = eval_prediction.label_ids

    if isinstance(predictions, tuple):
        predictions = predictions[0]

    predicted_labels = np.argmax(predictions, axis=-1)

    return {
        "accuracy": accuracy_score(labels, predicted_labels),
        "precision": precision_score(
            labels,
            predicted_labels,
            average="binary",
            pos_label=1,
            zero_division=0,
        ),
        "recall": recall_score(
            labels,
            predicted_labels,
            average="binary",
            pos_label=1,
            zero_division=0,
        ),
        "f1": f1_score(
            labels,
            predicted_labels,
            average="binary",
            pos_label=1,
            zero_division=0,
        ),
    }


class WeightedTrainer(Trainer):
    """Trainer with inverse-frequency class-weighted cross-entropy loss."""

    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)

        if class_weights is None:
            raise ValueError("class_weights must be provided.")

        self.class_weights = torch.tensor(
            class_weights,
            dtype=torch.float32,
        )

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
    ):
        labels = inputs.pop("labels")

        outputs = model(**inputs)

        weights = self.class_weights.to(
            outputs.logits.device
        )

        loss_function = torch.nn.CrossEntropyLoss(
            weight=weights
        )

        loss = loss_function(
            outputs.logits,
            labels,
        )

        return (
            (loss, outputs)
            if return_outputs
            else loss
        )




def configure_trainable_layers(model, trainable_layers: str = "all"):
    """Configure which DeBERTa encoder layers are trainable.

    Options:
        all             -> all model parameters trainable
        last_6          -> encoder layers 6-11 + pooler + classifier
        last_3          -> encoder layers 9-11 + pooler + classifier
        classifier_only -> pooler + classifier only
    """

    valid_options = {
        "all",
        "last_6",
        "last_3",
        "classifier_only",
    }

    if trainable_layers not in valid_options:
        raise ValueError(
            f"Unknown configuration: {trainable_layers}. "
            f"Choose from {sorted(valid_options)}."
        )

    # Start by freezing everything.
    for parameter in model.parameters():
        parameter.requires_grad = False

    if trainable_layers == "all":
        # EXP-05-compatible configuration:
        # every parameter is trainable.
        for parameter in model.parameters():
            parameter.requires_grad = True

    else:
        encoder_layers = model.deberta.encoder.layer

        if trainable_layers == "last_6":
            start_layer = 6

        elif trainable_layers == "last_3":
            start_layer = 9

        else:
            # classifier_only
            start_layer = len(encoder_layers)

        # Unfreeze selected encoder layers.
        for layer in encoder_layers[start_layer:]:
            for parameter in layer.parameters():
                parameter.requires_grad = True

        # Keep task-specific components trainable.
        for parameter in model.pooler.parameters():
            parameter.requires_grad = True

        for parameter in model.classifier.parameters():
            parameter.requires_grad = True

    return model

def train(
    csv_path: Path,
    output_dir: Path,
    learning_rate: float = 2e-5,
    epochs: float = 3,
    train_batch_size: int = 4,
    eval_batch_size: int = 8,
    gradient_accumulation_steps: int = 4,
    max_length: int = 128,
    trainable_layers: str = "all",
):
    """Fine-tune DeBERTa-v3-base with a configurable set of trainable layers."""

    train_dataset, validation_dataset, test_dataset, dataset_info = (
        prepare_dataset(csv_path)
    )

    print("Dataset information:")
    print(json.dumps(dataset_info, indent=2))

    # Calculate inverse-frequency class weights from the training set.
    # Label 0 = defect, Label 1 = ok.
    raw_train = train_dataset["labels"]

    class_counts = np.bincount(
        raw_train,
        minlength=2,
    )

    class_weights = (
        len(raw_train)
        / (2.0 * class_counts)
    )

    print("Class counts:")
    print(f"  defect (0): {class_counts[0]}")
    print(f"  ok     (1): {class_counts[1]}")

    print("Class weights:")
    print(f"  defect (0): {class_weights[0]:.6f}")
    print(f"  ok     (1): {class_weights[1]:.6f}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    tokenized_train = tokenize_dataset(
        train_dataset,
        tokenizer,
        max_length,
    )

    tokenized_validation = tokenize_dataset(
        validation_dataset,
        tokenizer,
        max_length,
    )

    tokenized_test = tokenize_dataset(
        test_dataset,
        tokenizer,
        max_length,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    # Explicitly keep the model in FP32 for stable full-precision training.
    model = model.float()

    # Configure which DeBERTa layers are trainable.
    model = configure_trainable_layers(
        model,
        trainable_layers=trainable_layers,
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_encoder_layers = [
        index
        for index, layer in enumerate(model.deberta.encoder.layer)
        if any(parameter.requires_grad for parameter in layer.parameters())
    ]

    print(f"Trainable layer configuration: {trainable_layers}")
    print(f"Trainable encoder layers: {trainable_encoder_layers}")
    print(f"Total parameters: {total_parameters:,}")
    print(f"Trainable parameters: {trainable_parameters:,}")

    if trainable_layers == "all" and trainable_parameters != total_parameters:
        raise RuntimeError(
            "The 'all' configuration must make every DeBERTa parameter trainable."
        )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=learning_rate,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=epochs,
        weight_decay=0.01,
        warmup_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2,
        logging_steps=25,
        report_to="none",
        fp16=False,
        seed=42,
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_validation,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        class_weights=class_weights,
    )

    print("\nStarting fine-tuning...")
    trainer.train()

    print("\nEvaluating on validation set...")
    validation_metrics = trainer.evaluate(
        eval_dataset=tokenized_validation
    )

    print("\nEvaluating on test set...")
    test_metrics = trainer.evaluate(
        eval_dataset=tokenized_test,
        metric_key_prefix="test",
    )

    final_model_dir = output_dir / "final_model"

    trainer.save_model(str(final_model_dir))
    tokenizer.save_pretrained(str(final_model_dir))

    experiment = {
        "model": MODEL_NAME,
        "task": "QuRE binary requirement quality classification",
        "labels": LABEL2ID,
        "trainable_layers": trainable_layers,
        "trainable_encoder_layers": trainable_encoder_layers,
        "all_model_parameters_trainable": trainable_parameters == total_parameters,
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "train_batch_size": train_batch_size,
        "eval_batch_size": eval_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "max_length": max_length,
        "dataset": dataset_info,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
    }

    metrics_path = output_dir / "experiment_metrics.json"

    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(experiment, file, indent=2, default=float)

    print("\nTraining complete.")
    print(f"Model saved to: {final_model_dir}")
    print(f"Metrics saved to: {metrics_path}")

    print("\nTest metrics:")
    for key, value in test_metrics.items():
        print(f"{key}: {value}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--csv",
        required=True,
        type=Path,
        help="Path to QuRE.csv",
    )

    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Directory where the trained model will be saved.",
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
    )

    parser.add_argument(
        "--epochs",
        type=float,
        default=3,
    )

    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--trainable-layers",
        choices=[
            "all",
            "last_6",
            "last_3",
            "classifier_only",
        ],
        default="all",
        help="Which DeBERTa layers to fine-tune.",
    )

    args = parser.parse_args()

    train(
        csv_path=args.csv,
        output_dir=args.output,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_length=args.max_length,
        trainable_layers=args.trainable_layers,
    )


if __name__ == "__main__":
    main()