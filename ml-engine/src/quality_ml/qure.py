"""Utilities for preparing the QuRE requirement-quality dataset."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def load_qure(path: str | Path) -> pd.DataFrame:
    """Load the raw QuRE CSV dataset."""
    return pd.read_csv(path)


def clean_qure(df: pd.DataFrame) -> pd.DataFrame:
    """Return one clean row per requirement with a consistent label.

    Requirements whose identical text appears with conflicting labels
    are excluded from the supervised dataset.
    """
    required_columns = {"id", "requirement", "defect", "weak_word"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"QuRE dataset is missing columns: {sorted(missing_columns)}"
        )

    # Find requirement texts that have exactly one label.
    label_counts = df.groupby("requirement")["defect"].nunique()
    consistent_requirements = label_counts[label_counts == 1].index

    clean = df[df["requirement"].isin(consistent_requirements)].copy()

    # Keep exactly one row for each unique requirement text.
    clean = clean.drop_duplicates(subset="requirement").reset_index(drop=True)

    return clean


def split_qure(
    df: pd.DataFrame,
    test_size: float = 0.10,
    validation_size: float = 0.10,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split clean QuRE data into train, validation and test sets.

    The split is stratified by the QuRE defect label so that the
    class distribution stays approximately the same in each split.
    """
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    if not 0 < validation_size < 1:
        raise ValueError("validation_size must be between 0 and 1.")

    if test_size + validation_size >= 1:
        raise ValueError("test_size + validation_size must be less than 1.")

    train_validation, test = train_test_split(
        df,
        test_size=test_size,
        stratify=df["defect"],
        random_state=random_state,
    )

    validation_fraction = validation_size / (1 - test_size)

    train, validation = train_test_split(
        train_validation,
        test_size=validation_fraction,
        stratify=train_validation["defect"],
        random_state=random_state,
    )

    return (
        train.reset_index(drop=True),
        validation.reset_index(drop=True),
        test.reset_index(drop=True),
    )
