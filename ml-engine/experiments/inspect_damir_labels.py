"""Inspect representative DAMIR ambiguity annotations."""

from pathlib import Path

from quality_ml.damir import load_damir, split_damir

DATA_PATH = (
    Path(__file__).resolve().parents[3]
    / "Datasets"
    / "requirement-quality"
    / "DAMIR.xlsx"
)


def print_occurrence(group):
    """Print one complete DAMIR pronoun occurrence."""

    first = group.iloc[0]

    print()
    print("=" * 100)
    print(f"ID: {first['Id']}")
    print(f"Pronoun: {first['Pronoun']}")
    print(f"Position: {first['Position']}")
    print(f"AckUnack: {first['AckUnack']}")

    print()
    print("CONTEXT")
    print("-" * 100)
    print(first["Context"])

    print()
    print("CANDIDATE ANNOTATIONS")
    print("-" * 100)

    for _, row in group.iterrows():
        print(
            f"- {row['Candidate Antecedent']}"
            f" | ResolvedAs={row['ResolvedAs']}"
            f" | AckUnack={row['AckUnack']}"
        )


def main():
    """Inspect representative validation occurrences."""

    print("Loading DAMIR...")
    df = load_damir(DATA_PATH)

    print("Creating fixed train/validation/test split...")
    _, validation, _ = split_damir(df)

    groups = list(
        validation.groupby(
            ["Id", "Context", "Pronoun", "Position", "AckUnack"],
            sort=False,
        )
    )

    selected = []

    # Select up to 5 Unambiguous occurrences.
    for key, group in groups:
        if key[-1] == "Unambiguous":
            selected.append((key, group))

        if sum(
            1
            for selected_key, _
            in selected
            if selected_key[-1] == "Unambiguous"
        ) >= 5:
            break

    # Select up to 5 Unack occurrences.
    for key, group in groups:
        if key[-1] == "Unack":
            selected.append((key, group))

        if sum(
            1
            for selected_key, _
            in selected
            if selected_key[-1] == "Unack"
        ) >= 5:
            break

    # Select all Ack occurrences in validation.
    for key, group in groups:
        if key[-1] == "Ack":
            selected.append((key, group))

    print()
    print("Representative validation annotations")
    print("======================================")
    print(
        "Selected occurrences: "
        f"{len(selected)}"
    )

    for _, group in selected:
        print_occurrence(group)

    print()
    print("=" * 100)
    print("Annotation inspection complete.")
    print("Only the validation split was inspected.")
    print("The test split was not used.")


if __name__ == "__main__":
    main()
