"""Utilities for preparing and evaluating the DAMIR ambiguity dataset."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

REQUIRED_COLUMNS = {
    "Id",
    "Context",
    "Pronoun",
    "Position",
    "Candidate Antecedent",
    "ResolvedAs",
    "AckUnack",
}


def load_damir(path: str | Path) -> pd.DataFrame:
    """Load the DAMIR Excel dataset."""
    return pd.read_excel(path)


def validate_damir(df: pd.DataFrame) -> None:
    """Validate the structure and occurrence-level consistency of DAMIR."""
    missing_columns = REQUIRED_COLUMNS - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"DAMIR dataset is missing columns: {sorted(missing_columns)}"
        )

    if df[list(REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError("DAMIR dataset contains missing values.")

    occurrence_columns = [
        "Context",
        "Pronoun",
        "Position",
        "AckUnack",
    ]

    inconsistent = {
        column: int((df.groupby("Id")[column].nunique() > 1).sum())
        for column in occurrence_columns
    }

    inconsistent = {
        column: count
        for column, count in inconsistent.items()
        if count > 0
    }

    if inconsistent:
        raise ValueError(
            f"DAMIR occurrence fields are inconsistent: {inconsistent}"
        )


def clean_damir(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned candidate-level DAMIR dataset."""
    validate_damir(df)

    columns = [
        "Id",
        "Context",
        "Pronoun",
        "Position",
        "Candidate Antecedent",
        "ResolvedAs",
        "AckUnack",
    ]

    clean = df[columns].copy()

    clean["Id"] = clean["Id"].astype(str).str.strip()
    clean["Context"] = clean["Context"].astype(str).str.strip()
    clean["Pronoun"] = clean["Pronoun"].astype(str).str.strip()
    clean["Candidate Antecedent"] = (
        clean["Candidate Antecedent"]
        .astype(str)
        .str.strip()
    )
    clean["ResolvedAs"] = (
        clean["ResolvedAs"]
        .astype(str)
        .str.strip()
    )
    clean["AckUnack"] = (
        clean["AckUnack"]
        .astype(str)
        .str.strip()
    )

    return clean.reset_index(drop=True)


def split_damir(
    df: pd.DataFrame,
    test_size: float = 0.10,
    validation_size: float = 0.10,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split DAMIR by occurrence ID to prevent candidate-level leakage."""
    clean = clean_damir(df)

    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    if not 0 < validation_size < 1:
        raise ValueError("validation_size must be between 0 and 1.")

    if test_size + validation_size >= 1:
        raise ValueError(
            "test_size + validation_size must be less than 1."
        )

    occurrence_ids = clean["Id"].astype(str).unique().tolist()

    train_ids, temporary_ids = train_test_split(
        occurrence_ids,
        test_size=test_size + validation_size,
        random_state=random_state,
    )

    validation_fraction = (
        validation_size / (test_size + validation_size)
    )

    validation_ids, test_ids = train_test_split(
        temporary_ids,
        test_size=1 - validation_fraction,
        random_state=random_state,
    )

    train_ids = set(train_ids)
    validation_ids = set(validation_ids)
    test_ids = set(test_ids)

    if train_ids & validation_ids:
        raise ValueError(
            "Train and validation occurrences overlap."
        )

    if train_ids & test_ids:
        raise ValueError(
            "Train and test occurrences overlap."
        )

    if validation_ids & test_ids:
        raise ValueError(
            "Validation and test occurrences overlap."
        )

    all_ids = set(occurrence_ids)

    if train_ids | validation_ids | test_ids != all_ids:
        raise ValueError(
            "Some occurrences were lost during splitting."
        )

    train = clean[
        clean["Id"].isin(train_ids)
    ].reset_index(drop=True)

    validation = clean[
        clean["Id"].isin(validation_ids)
    ].reset_index(drop=True)

    test = clean[
        clean["Id"].isin(test_ids)
    ].reset_index(drop=True)

    return train, validation, test


def unique_damir_occurrences(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Return one row per unique pronoun occurrence."""

    return (
        df[
            [
                "Id",
                "Context",
                "Pronoun",
                "Position",
                "AckUnack",
            ]
        ]
        .drop_duplicates(
            subset=["Id", "Pronoun", "Position"]
        )
        .reset_index(drop=True)
    )


def candidate_annotations(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Return unique candidate-level DAMIR annotations."""

    return (
        df[
            [
                "Id",
                "Candidate Antecedent",
                "ResolvedAs",
            ]
        ]
        .drop_duplicates()
        .reset_index(drop=True)
    )


def calculate_candidate_coverage(
    annotated_candidates: list[str],
    extracted_candidates: list[str],
) -> dict:
    """Calculate exact candidate coverage."""

    annotated = {
        candidate.strip().lower()
        for candidate in annotated_candidates
        if isinstance(candidate, str)
    }

    extracted = {
        candidate.strip().lower()
        for candidate in extracted_candidates
        if isinstance(candidate, str)
    }

    matched = annotated & extracted

    coverage = (
        len(matched) / len(annotated)
        if annotated
        else 0.0
    )

    return {
        "annotated_count": len(annotated),
        "extracted_count": len(extracted),
        "matched_count": len(matched),
        "coverage": coverage,
    }
