"""Summarize DAMIR annotation relationships."""

from pathlib import Path

import pandas as pd
from quality_ml.damir import load_damir

DATA_PATH = (
    Path(__file__).resolve().parents[3]
    / "Datasets"
    / "requirement-quality"
    / "DAMIR.xlsx"
)


def main():
    """Print relationships between DAMIR occurrence and candidate labels."""

    df = load_damir(DATA_PATH)

    print("DAMIR label summary")
    print("===================")

    print()
    print("Occurrence-level labels")
    print("------------------------")

    occurrence_labels = (
        df[
            [
                "Id",
                "Context",
                "Pronoun",
                "Position",
                "AckUnack",
            ]
        ]
        .drop_duplicates()
    )

    print(
        occurrence_labels["AckUnack"]
        .value_counts()
        .to_string()
    )

    print()
    print("Candidate-level labels")
    print("----------------------")

    print(
        df["ResolvedAs"]
        .value_counts()
        .to_string()
    )

    print()
    print("ResolvedAs distribution by AckUnack")
    print("------------------------------------")

    cross_tab = pd.crosstab(
        df["AckUnack"],
        df["ResolvedAs"],
    )

    print(cross_tab.to_string())

    print()
    print("Occurrence-level correct-candidate availability")
    print("-----------------------------------------------")

    occurrence_correct = (
        df.assign(
            is_correct=df["ResolvedAs"].eq("correct")
        )
        .groupby("Id")["is_correct"]
        .any()
    )

    occurrence_labels_by_id = (
        df.groupby("Id")["AckUnack"]
        .first()
    )

    summary = pd.crosstab(
        occurrence_labels_by_id,
        occurrence_correct,
    )

    summary.columns = [
        "no_correct_candidate"
        if column is False
        else "has_correct_candidate"
        for column in summary.columns
    ]

    print(summary.to_string())

    print()
    print("Number of correct candidates per occurrence")
    print("-------------------------------------------")

    correct_counts = (
        df[df["ResolvedAs"] == "correct"]
        .groupby("Id")
        .size()
    )

    correct_counts = correct_counts.reindex(
        occurrence_labels_by_id.index,
        fill_value=0,
    )

    correct_summary = (
        pd.DataFrame(
            {
                "AckUnack": occurrence_labels_by_id,
                "correct_candidates": correct_counts,
            }
        )
        .groupby(
            ["AckUnack", "correct_candidates"]
        )
        .size()
        .rename("occurrences")
    )

    print(correct_summary.to_string())

    print()
    print("Unique candidate labels per occurrence")
    print("---------------------------------------")

    candidate_label_counts = (
        df.groupby("Id")["ResolvedAs"]
        .nunique()
    )

    print(
        candidate_label_counts
        .value_counts()
        .sort_index()
        .to_string()
    )

    print()
    print("Summary complete.")
    print("This script only summarizes the existing DAMIR annotations.")
    print("No model or production code was changed.")


if __name__ == "__main__":
    main()
