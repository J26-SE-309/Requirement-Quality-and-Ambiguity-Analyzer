
"""Dataset preparation utilities for Task 3 stability analysis."""

import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


REQUIRED_COLUMNS = {
    "requirement_id",
    "source_dataset",
    "requirement_text",
}


def load_task3_dataset(path: str | Path) -> pd.DataFrame:
    """Load the combined Task 3 requirements dataset."""
    return pd.read_csv(path)


def normalize_requirement(text: str) -> str:
    """Normalize formatting for requirement duplicate detection."""

    normalized = str(text).strip().casefold()

    # Remove leading numbered-list markers.
    normalized = re.sub(
        r"^\s*\d+[.)]\s+",
        "",
        normalized,
    )

    # Remove insignificant punctuation differences.
    normalized = re.sub(
        r"[^\w\s]",
        " ",
        normalized,
    )

    # Normalize whitespace.
    normalized = re.sub(
        r"\s+",
        " ",
        normalized,
    ).strip()

    return normalized


def clean_task3_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean the combined requirements dataset."""

    missing_columns = REQUIRED_COLUMNS - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Task 3 dataset is missing columns: {sorted(missing_columns)}"
        )

    clean = df.copy()

    clean = clean.dropna(
        subset=["requirement_id", "requirement_text"]
    )

    clean["requirement_text"] = (
        clean["requirement_text"].astype(str).str.strip()
    )

    clean = clean[
        clean["requirement_text"] != ""
    ].copy()

    clean["normalized_text"] = (
        clean["requirement_text"].map(normalize_requirement)
    )

    clean = clean[
        clean["normalized_text"] != ""
    ].copy()

    clean = clean.drop_duplicates(
        subset="requirement_id",
        keep="first",
    )

    clean = clean.drop_duplicates(
        subset="normalized_text",
        keep="first",
    )

    clean = clean.drop(columns=["normalized_text"])

    return clean.reset_index(drop=True)


def build_near_duplicate_groups(
    df: pd.DataFrame,
    near_duplicate_pairs: pd.DataFrame,
) -> pd.Series:
    """Assign connected near-duplicate requirements to groups.

    Each pair must contain first_id and second_id.
    Requirements connected through multiple pairs share a group.
    """

    required_pair_columns = {"first_id", "second_id"}

    if not required_pair_columns.issubset(
        near_duplicate_pairs.columns
    ):
        raise ValueError(
            "Near-duplicate pairs must contain "
            "'first_id' and 'second_id'."
        )

    ids = df["requirement_id"].astype(str).tolist()

    if len(ids) != len(set(ids)):
        raise ValueError(
            "Duplicate requirement IDs detected."
        )

    parent = {requirement_id: requirement_id for requirement_id in ids}

    def find(requirement_id: str) -> str:
        """Find the representative of a requirement group."""

        while parent[requirement_id] != requirement_id:
            parent[requirement_id] = parent[
                parent[requirement_id]
            ]
            requirement_id = parent[requirement_id]

        return requirement_id

    def union(first_id: str, second_id: str) -> None:
        """Merge two connected requirement groups."""

        first_root = find(first_id)
        second_root = find(second_id)

        if first_root != second_root:
            parent[second_root] = first_root

    for pair in near_duplicate_pairs.itertuples(index=False):
        first_id = str(pair.first_id)
        second_id = str(pair.second_id)

        if first_id not in parent or second_id not in parent:
            raise ValueError(
                "Near-duplicate pair references an unknown "
                f"requirement: {first_id}, {second_id}"
            )

        union(first_id, second_id)

    return pd.Series(
        [find(requirement_id) for requirement_id in ids],
        index=df.index,
        name="duplicate_group",
    )


def split_task3_dataset(
    df: pd.DataFrame,
    test_size: float = 0.10,
    validation_size: float = 0.10,
    random_state: int = 42,
    near_duplicate_pairs: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split original requirements while keeping related rows together.

    All mutations must inherit their original requirement's split.
    """

    if not 0 < test_size < 1:
        raise ValueError(
            "test_size must be between 0 and 1."
        )

    if not 0 < validation_size < 1:
        raise ValueError(
            "validation_size must be between 0 and 1."
        )

    if test_size + validation_size >= 1:
        raise ValueError(
            "test_size + validation_size must be less than 1."
        )

    if df.empty:
        raise ValueError("Cannot split an empty dataset.")

    if df["requirement_id"].duplicated().any():
        raise ValueError(
            "Duplicate requirement IDs detected."
        )

    normalized_text = (
        df["requirement_text"]
        .astype(str)
        .map(normalize_requirement)
    )

    if normalized_text.duplicated().any():
        raise ValueError(
            "Duplicate normalized requirement texts detected."
        )

    if near_duplicate_pairs is None or near_duplicate_pairs.empty:
        train_validation, test = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            shuffle=True,
        )

        validation_fraction = (
            validation_size / (1 - test_size)
        )

        train, validation = train_test_split(
            train_validation,
            test_size=validation_fraction,
            random_state=random_state,
            shuffle=True,
        )

    else:
        grouped = df.copy()

        grouped["duplicate_group"] = build_near_duplicate_groups(
            grouped,
            near_duplicate_pairs,
        )

        # Shuffle groups reproducibly, then allocate entire groups
        # to splits, targeting approximately 80/10/10 rows.
        group_sizes = (
            grouped.groupby("duplicate_group", sort=False)
            .size()
            .sample(frac=1, random_state=random_state)
        )

        total_rows = len(grouped)
        target_test = round(total_rows * test_size)
        target_validation = round(total_rows * validation_size)

        assignments = {}
        test_count = 0
        validation_count = 0

        for group_id, group_size in group_sizes.items():
            if test_count < target_test:
                split_name = "test"
                test_count += group_size
            elif validation_count < target_validation:
                split_name = "validation"
                validation_count += group_size
            else:
                split_name = "train"

            assignments[group_id] = split_name

        grouped["_split"] = (
            grouped["duplicate_group"].map(assignments)
        )

        train = grouped[
            grouped["_split"] == "train"
        ].copy()

        validation = grouped[
            grouped["_split"] == "validation"
        ].copy()

        test = grouped[
            grouped["_split"] == "test"
        ].copy()

        temporary_columns = [
            "duplicate_group",
            "_split",
        ]

        train = train.drop(columns=temporary_columns)
        validation = validation.drop(columns=temporary_columns)
        test = test.drop(columns=temporary_columns)

    # Verify that no requirement IDs overlap.
    train_ids = set(train["requirement_id"])
    validation_ids = set(validation["requirement_id"])
    test_ids = set(test["requirement_id"])

    if (
        train_ids & validation_ids
        or train_ids & test_ids
        or validation_ids & test_ids
    ):
        raise ValueError(
            "Data leakage detected: requirement IDs overlap."
        )

    # Verify normalized text does not overlap.
    train_texts = set(
        train["requirement_text"].map(normalize_requirement)
    )

    validation_texts = set(
        validation["requirement_text"].map(normalize_requirement)
    )

    test_texts = set(
        test["requirement_text"].map(normalize_requirement)
    )

    if (
        train_texts & validation_texts
        or train_texts & test_texts
        or validation_texts & test_texts
    ):
        raise ValueError(
            "Data leakage detected: normalized texts overlap."
        )

    # Verify known near-duplicate pairs stay in the same split.
    if near_duplicate_pairs is not None:
        split_lookup = {
            **dict.fromkeys(train_ids, "train"),
            **dict.fromkeys(validation_ids, "validation"),
            **dict.fromkeys(test_ids, "test"),
        }

        for pair in near_duplicate_pairs.itertuples(index=False):
            first_id = str(pair.first_id)
            second_id = str(pair.second_id)

            if split_lookup.get(first_id) != split_lookup.get(
                second_id
            ):
                raise ValueError(
                    "Near-duplicate leakage detected: "
                    f"{first_id} and {second_id}"
                )

    return (
        train.reset_index(drop=True),
        validation.reset_index(drop=True),
        test.reset_index(drop=True),
    )


def save_task3_splits(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    output_dir: str | Path,
) -> None:
    """Save original requirement splits as separate CSV files."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    splits = {
        "train": train,
        "val": validation,
        "test": test,
    }

    for split_name, split_df in splits.items():
        file_path = (
            output_path
            / f"task3_{split_name}_requirements.csv"
        )

        split_df.to_csv(
            file_path,
            index=False,
        )

        print(
            f"Saved {split_name}: "
            f"{len(split_df)} requirements"
        )

    print(f"Output directory: {output_path}")
