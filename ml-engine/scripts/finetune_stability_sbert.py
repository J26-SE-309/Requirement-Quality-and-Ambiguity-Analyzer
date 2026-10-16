
"""Fine-tune SBERT for Interpretation Stability Analysis in Google Colab.

This script:
- Uses only the training split.
- Supports frozen, last-N-layer, and all-layer experiments.
- Trains with a cosine-distance contrastive objective.
- Saves the fine-tuned encoder and experiment metadata.

Do not run training on local Windows.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader, Dataset

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

LABEL_TO_INT = {
    "stable": 1,
    "unstable": 0,
}

EXPERIMENTS = {
    "EXP-01": {
        "trainable_layers": 0,
        "learning_rate": None,
        "epochs": 0,
    },
    "EXP-02": {
        "trainable_layers": 2,
        "learning_rate": 2e-5,
        "epochs": 2,
    },
    "EXP-03": {
        "trainable_layers": 4,
        "learning_rate": 2e-5,
        "epochs": 2,
    },
    "EXP-04": {
        "trainable_layers": "all",
        "learning_rate": 2e-5,
        "epochs": 2,
    },
    "EXP-05": {
        "trainable_layers": "all",
        "learning_rate": 1e-5,
        "epochs": 3,
    },
    "EXP-06": {
        "trainable_layers": 4,
        "learning_rate": 1e-5,
        "epochs": 3,
    },
}


class RequirementPairDataset(Dataset):
    """Store labeled pairs of software requirements."""

    def __init__(self, frame: pd.DataFrame) -> None:
        self.originals = frame["original_text"].tolist()
        self.mutated = frame["mutated_text"].tolist()
        self.labels = [
            LABEL_TO_INT[label]
            for label in frame["stability_label"]
        ]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[str, str, int]:
        return (
            self.originals[index],
            self.mutated[index],
            self.labels[index],
        )


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_training_data(path: Path) -> pd.DataFrame:
    """Load and validate the training split."""

    frame = pd.read_csv(path, keep_default_na=False)

    required = {
        "pair_id",
        "requirement_id",
        "original_text",
        "mutated_text",
        "stability_label",
    }

    missing = required - set(frame.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    if frame.empty:
        raise ValueError("Training dataset is empty.")

    if frame["pair_id"].duplicated().any():
        raise ValueError("Duplicate pair IDs found.")

    if not frame["stability_label"].isin(LABEL_TO_INT).all():
        raise ValueError("Invalid stability labels found.")

    if (
        not frame["original_text"].astype(str).str.strip().all()
        or not frame["mutated_text"].astype(str).str.strip().all()
    ):
        raise ValueError("Blank requirement text found.")

    if (
        "split" in frame.columns
        and not frame["split"].astype(str).str.lower().eq("train").all()
    ):
        raise ValueError(
            "Fine-tuning input must contain only training rows."
        )

    if frame["stability_label"].nunique() != 2:
        raise ValueError("Both stability classes are required.")

    return frame


def configure_trainable_layers(
    model: SentenceTransformer,
    selection: int | str,
) -> tuple[int, int]:
    """Freeze the encoder and unfreeze selected transformer layers."""

    transformer = model[0].auto_model

    for parameter in model.parameters():
        parameter.requires_grad = False

    if not hasattr(transformer, "encoder"):
        raise ValueError("Unsupported transformer architecture.")

    if not hasattr(transformer.encoder, "layer"):
        raise ValueError("Transformer encoder layers not found.")

    layers = transformer.encoder.layer

    if selection == "all":
        for parameter in model.parameters():
            parameter.requires_grad = True

    elif isinstance(selection, int):
        if selection < 0 or selection > len(layers):
            raise ValueError("Invalid number of trainable layers.")

        if selection > 0:
            for layer in layers[-selection:]:
                for parameter in layer.parameters():
                    parameter.requires_grad = True

    else:
        raise ValueError("Unknown layer selection.")

    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    total = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    return trainable, total


def collate_pairs(batch):
    """Prepare a batch without tokenizing on Windows."""

    originals, mutated, labels = zip(*batch, strict=True)

    return (
        list(originals),
        list(mutated),
        torch.tensor(labels, dtype=torch.float32),
    )


def encode_batch(
    model: SentenceTransformer,
    texts: list[str],
    device: torch.device,
) -> torch.Tensor:
    """Generate differentiable sentence embeddings."""

    features = model.tokenize(texts)

    features = {
        key: value.to(device)
        if isinstance(value, torch.Tensor)
        else value
        for key, value in features.items()
    }

    output = model(features)

    return output["sentence_embedding"]


def contrastive_loss(
    original_embeddings: torch.Tensor,
    mutated_embeddings: torch.Tensor,
    labels: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    """Contrastive loss using cosine distance.

    Stable pairs: minimize cosine distance.
    Unstable pairs: enforce a minimum cosine distance.
    """

    similarities = torch.nn.functional.cosine_similarity(
        original_embeddings,
        mutated_embeddings,
        dim=1,
    )

    distances = 1.0 - similarities

    positive_loss = labels * distances.square()

    negative_loss = (
        (1.0 - labels)
        * torch.relu(margin - distances).square()
    )

    return (positive_loss + negative_loss).mean()


def train_experiment(
    frame: pd.DataFrame,
    experiment_id: str,
    output_root: Path,
    batch_size: int,
    margin: float,
    seed: int,
) -> None:
    """Run one predefined SBERT experiment."""

    set_seed(seed)

    config = EXPERIMENTS[experiment_id]

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    if device.type != "cuda":
        raise RuntimeError(
            "This training script requires a Colab GPU runtime."
        )

    model = SentenceTransformer(MODEL_ID, device=str(device))

    trainable, total = configure_trainable_layers(
        model,
        config["trainable_layers"],
    )

    print(f"Experiment: {experiment_id}")
    print(f"Trainable parameters: {trainable:,}")
    print(f"Total parameters: {total:,}")
    print(f"Trainable percentage: {100 * trainable / total:.2f}%")

    output_dir = output_root / experiment_id
    output_dir.mkdir(parents=True, exist_ok=True)

    history = []

    if config["epochs"] > 0:
        dataset = RequirementPairDataset(frame)

        generator = torch.Generator()
        generator.manual_seed(seed)

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_pairs,
            generator=generator,
        )

        optimizer = torch.optim.AdamW(
            [
                parameter
                for parameter in model.parameters()
                if parameter.requires_grad
            ],
            lr=config["learning_rate"],
            weight_decay=0.01,
        )

        model.train()

        for epoch in range(config["epochs"]):
            total_loss = 0.0

            for originals, mutated, labels in loader:
                labels = labels.to(device)

                optimizer.zero_grad()

                original_embeddings = encode_batch(
                    model,
                    originals,
                    device,
                )

                mutated_embeddings = encode_batch(
                    model,
                    mutated,
                    device,
                )

                loss = contrastive_loss(
                    original_embeddings,
                    mutated_embeddings,
                    labels,
                    margin,
                )

                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            average_loss = total_loss / len(loader)

            history.append(
                {
                    "epoch": epoch + 1,
                    "training_loss": average_loss,
                }
            )

            print(
                f"Epoch {epoch + 1}/{config['epochs']} "
                f"- Loss: {average_loss:.6f}"
            )

    model.save(str(output_dir / "encoder"))

    metadata = {
        "experiment_id": experiment_id,
        "base_model": MODEL_ID,
        "trainable_layers": config["trainable_layers"],
        "learning_rate": config["learning_rate"],
        "epochs": config["epochs"],
        "batch_size": batch_size,
        "margin": margin,
        "seed": seed,
        "training_pairs": len(frame),
        "trainable_parameters": trainable,
        "total_parameters": total,
        "training_history": history,
        "label_status": "AI_ASSISTED_PROVISIONAL",
        "evaluation_status": "NOT_EVALUATED",
    }

    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print(f"Saved experiment to: {output_dir}")


def main() -> None:
    """Parse arguments and run the selected experiment."""

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--experiment",
        choices=list(EXPERIMENTS),
        required=True,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--margin",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("Batch size must be positive.")

    if not 0.0 < args.margin <= 2.0:
        parser.error("Margin must be between 0 and 2.")

    frame = load_training_data(args.train)

    train_experiment(
        frame=frame,
        experiment_id=args.experiment,
        output_root=args.output,
        batch_size=args.batch_size,
        margin=args.margin,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
