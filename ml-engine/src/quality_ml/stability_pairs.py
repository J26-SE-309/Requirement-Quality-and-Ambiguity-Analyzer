
"""Generate controlled requirement mutation pairs for Task 3.

This module creates unlabeled original-mutated requirement pairs.
Mutation types must not be interpreted as stability labels.
"""

from pathlib import Path

import pandas as pd

from quality_ml.stability_mutations import (
    mutate_actor_role,
    mutate_condition_scope,
    mutate_numeric_capacity,
    mutate_temporal_constraint,
    mutate_vague_term,
)


REQUIRED_COLUMNS = {
    "requirement_id",
    "requirement_text",
}

MUTATION_FUNCTIONS = (
    mutate_vague_term,
    mutate_condition_scope,
    mutate_actor_role,
    mutate_numeric_capacity,
    mutate_temporal_constraint,
)

SPLIT_NAMES = ("train", "val", "test")

# Exclude identified cross-split near-duplicate requirements.
# These requirements remain in the original dataset splits
# but will not be used to generate mutation pairs.
EXCLUDED_REQUIREMENT_IDS = frozenset({
    "QURE-1525",
    "QURE-1526",
})

OUTPUT_COLUMNS = [
    "pair_id",
    "requirement_id",
    "split",
    "original_text",
    "mutated_text",
    "mutation_type",
    "original_span",
    "replacement_span",
]


def validate_requirements(
    df: pd.DataFrame,
    split_name: str,
) -> None:
    """Validate an original requirement split."""

    missing = REQUIRED_COLUMNS - set(df.columns)

    if missing:
        raise ValueError(
            f"{split_name}: missing columns {sorted(missing)}"
        )

    if df.empty:
        raise ValueError(
            f"{split_name}: requirement dataset is empty."
        )

    if df[list(REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError(
            f"{split_name}: null requirement IDs or texts."
        )

    if df["requirement_id"].duplicated().any():
        raise ValueError(
            f"{split_name}: duplicate requirement IDs."
        )

    if not df["requirement_id"].map(
        lambda value: isinstance(value, str)
        and bool(value.strip())
    ).all():
        raise ValueError(
            f"{split_name}: invalid requirement IDs."
        )

    if not df["requirement_text"].map(
        lambda value: isinstance(value, str)
        and bool(value.strip())
    ).all():
        raise ValueError(
            f"{split_name}: invalid requirement texts."
        )


def validate_mutation(
    mutation: dict,
    original_text: str,
) -> None:
    """Validate a generated mutation against its source."""

    required_keys = {
        "mutation_type",
        "original_text",
        "mutated_text",
        "original_span",
        "replacement_span",
    }

    missing = required_keys - set(mutation)

    if missing:
        raise ValueError(
            f"Mutation missing fields: {sorted(missing)}"
        )

    if mutation["original_text"] != original_text:
        raise ValueError(
            "Mutation original text does not match source."
        )

    if mutation["mutated_text"] == original_text:
        raise ValueError(
            "Mutation did not change the requirement."
        )

    original_span = mutation["original_span"]
    replacement_span = mutation["replacement_span"]

    if (
        not isinstance(original_span, str)
        or not original_span
        or not isinstance(replacement_span, str)
        or not replacement_span
    ):
        raise ValueError(
            "Mutation contains invalid replacement spans."
        )

    # Verify that the mutated requirement can be reconstructed
    # using one exact replacement in the original text.
    possible_reconstruction = False
    start = 0

    while True:
        position = original_text.find(original_span, start)

        if position == -1:
            break

        reconstructed = (
            original_text[:position]
            + replacement_span
            + original_text[position + len(original_span):]
        )

        if reconstructed == mutation["mutated_text"]:
            possible_reconstruction = True
            break

        start = position + 1

    if not possible_reconstruction:
        raise ValueError(
            "Mutation does not match a single span replacement."
        )


def generate_mutation_pairs(
    df: pd.DataFrame,
    split_name: str,
) -> pd.DataFrame:
    """Generate validated, unlabeled mutation pairs."""

    if split_name not in SPLIT_NAMES:
        raise ValueError(
            f"Unknown split name: {split_name}"
        )

    validate_requirements(df, split_name)

    rows = []
    seen_pairs = set()

    for requirement in df.itertuples(index=False):
        requirement_id = requirement.requirement_id
        original_text = requirement.requirement_text

        # Exclude identified cross-split near-duplicate requirements.
        # This prevents them from reappearing during regeneration.
        if requirement_id in EXCLUDED_REQUIREMENT_IDS:
            continue

        for mutation_function in MUTATION_FUNCTIONS:
            mutation = mutation_function(original_text)

            if mutation is None:
                continue

            validate_mutation(mutation, original_text)

            pair_key = (
                requirement_id,
                mutation["mutated_text"],
            )

            if pair_key in seen_pairs:
                raise ValueError(
                    f"Duplicate mutation pair: {requirement_id}"
                )

            seen_pairs.add(pair_key)

            rows.append(
                {
                    "pair_id": (
                        f"{split_name.upper()}-"
                        f"{len(rows) + 1:05d}"
                    ),
                    "requirement_id": requirement_id,
                    "split": split_name,
                    "original_text": original_text,
                    "mutated_text": mutation["mutated_text"],
                    "mutation_type": mutation["mutation_type"],
                    "original_span": mutation["original_span"],
                    "replacement_span": mutation["replacement_span"],
                }
            )

    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def verify_split_isolation(
    splits: dict[str, pd.DataFrame],
) -> None:
    """Check that requirement IDs do not cross splits."""

    id_sets = {
        name: set(df["requirement_id"])
        for name, df in splits.items()
    }

    for index, first in enumerate(SPLIT_NAMES):
        for second in SPLIT_NAMES[index + 1:]:
            overlap = id_sets[first] & id_sets[second]

            if overlap:
                raise ValueError(
                    "Requirement ID leakage between "
                    f"{first} and {second}: "
                    f"{len(overlap)} overlapping IDs."
                )


def build_all_mutation_pairs(
    input_dir: str | Path,
) -> dict[str, pd.DataFrame]:
    """Load existing splits and generate mutation pairs."""

    input_path = Path(input_dir)

    splits = {}

    for split_name in SPLIT_NAMES:
        file_path = (
            input_path
            / f"task3_{split_name}_requirements.csv"
        )

        df = pd.read_csv(file_path)

        validate_requirements(df, split_name)
        splits[split_name] = df

    verify_split_isolation(splits)

    return {
        name: generate_mutation_pairs(df, name)
        for name, df in splits.items()
    }


def save_mutation_pairs(
    pairs: dict[str, pd.DataFrame],
    output_dir: str | Path,
) -> None:
    """Save mutation pairs without overwriting existing files."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Check all output paths before writing any files.
    output_files = {
        name: output_path / f"task3_{name}_mutation_pairs.csv"
        for name in SPLIT_NAMES
    }

    existing = [
        str(path)
        for path in output_files.values()
        if path.exists()
    ]

    if existing:
        raise FileExistsError(
            "Mutation pair output already exists: "
            + ", ".join(existing)
        )

    for split_name in SPLIT_NAMES:
        df = pairs[split_name]
        file_path = output_files[split_name]

        df.to_csv(file_path, index=False)

        print(
            f"Saved {split_name}: "
            f"{len(df)} mutation pairs"
        )

    print(f"Output directory: {output_path}")


def main() -> None:
    """Generate Task 3 mutation pairs from existing splits."""

    dataset_dir = (
        Path.home()
        / "Desktop"
        / "Synapse"
        / "Datasets"
        / "requirement-quality"
        / "task3"
    )

    input_dir = dataset_dir / "processed"
    output_dir = dataset_dir / "mutation_pairs"

    pairs = build_all_mutation_pairs(input_dir)

    for split_name in SPLIT_NAMES:
        print(
            f"{split_name}: "
            f"{len(pairs[split_name])} mutation pairs"
        )

    total = sum(len(df) for df in pairs.values())

    print(f"Total mutation pairs: {total}")

    save_mutation_pairs(pairs, output_dir)


if __name__ == "__main__":
    main()
