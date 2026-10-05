"""Verify the reproducible DAMIR train/validation/test split."""

from quality_ml.damir import (
    load_damir,
    split_damir,
    unique_damir_occurrences,
)

DAMIR_PATH = (
    r"C:\Users\ASUS\Desktop\Synapse"
    r"\Datasets\requirement-quality\DAMIR.xlsx"
)


def print_split_summary(
    name: str,
    data,
) -> None:
    """Print occurrence and candidate-row statistics for a split."""

    occurrences = unique_damir_occurrences(data)

    print(f"{name}")
    print("=" * len(name))
    print(f"Candidate rows: {len(data)}")
    print(f"Occurrences: {len(occurrences)}")

    print("AckUnack:")
    print(
        occurrences["AckUnack"]
        .value_counts()
        .to_dict()
    )

    print("ResolvedAs:")
    print(
        data["ResolvedAs"]
        .value_counts()
        .to_dict()
    )

    print()


def main() -> None:
    """Create and verify the DAMIR split."""

    df = load_damir(DAMIR_PATH)

    train, validation, test = split_damir(
        df,
        test_size=0.10,
        validation_size=0.10,
        random_state=42,
    )

    train_ids = set(train["Id"])
    validation_ids = set(validation["Id"])
    test_ids = set(test["Id"])

    print_split_summary("Train", train)
    print_split_summary("Validation", validation)
    print_split_summary("Test", test)

    print("Split verification")
    print("==================")

    print(
        "Train ∩ Validation:",
        len(train_ids & validation_ids),
    )

    print(
        "Train ∩ Test:",
        len(train_ids & test_ids),
    )

    print(
        "Validation ∩ Test:",
        len(validation_ids & test_ids),
    )

    all_ids = set(df["Id"])

    print(
        "Total unique occurrences:",
        len(all_ids),
    )

    print(
        "Occurrences after split:",
        len(
            train_ids
            | validation_ids
            | test_ids
        ),
    )


if __name__ == "__main__":
    main()
