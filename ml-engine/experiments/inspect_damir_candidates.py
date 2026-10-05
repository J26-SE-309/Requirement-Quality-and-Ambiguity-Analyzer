"""Inspect extracted candidates against DAMIR validation annotations."""

from pathlib import Path

from quality_ml.ambiguity_checks import (
    find_candidate_antecedents,
    find_pronouns,
)
from quality_ml.damir import load_damir, split_damir

DATA_PATH = (
    Path(__file__).resolve().parents[3]
    / "Datasets"
    / "requirement-quality"
    / "DAMIR.xlsx"
)


def normalize(text: str) -> str:
    """Normalize text for comparison."""

    return " ".join(
        str(text).strip().lower().split()
    )


def main():
    """Inspect the first validation occurrences."""

    df = load_damir(DATA_PATH)

    _, validation, _ = split_damir(df)

    occurrence_groups = validation.groupby(
        ["Id", "Context", "Pronoun", "Position", "AckUnack"],
        sort=False,
    )

    printed = 0

    for (
        occurrence_id,
        context,
        pronoun,
        position,
        ack_unack,
    ), group in occurrence_groups:

        pronouns = find_pronouns(context)

        matching = [
            item
            for item in pronouns
            if (
                item["position"] == int(position)
                and item["text"].lower()
                == str(pronoun).lower()
            )
        ]

        if not matching:
            continue

        pronoun_info = matching[0]

        candidates = find_candidate_antecedents(
            context,
            pronoun_info["position"],
        )

        annotated = {
            normalize(row["Candidate Antecedent"]): str(
                row["ResolvedAs"]
            ).strip().lower()
            for _, row in group.iterrows()
        }

        print()
        print("=" * 90)
        print(
            f"ID: {occurrence_id} | "
            f"DAMIR: {ack_unack} | "
            f"Pronoun: {pronoun}"
        )
        print()
        print("Context:")
        print(context)
        print()
        print("Candidates:")
        print("-" * 90)

        for index, candidate in enumerate(
            candidates,
            start=1,
        ):
            key = normalize(candidate["text"])
            label = annotated.get(
                key,
                "NOT ANNOTATED",
            )

            print(
                f"{index:>2}. "
                f"{candidate['text']} "
                f"[dependency={candidate['dependency']}, "
                f"root_position={candidate['root_position']}] "
                f"-> {label}"
            )

        print()
        print("DAMIR annotated candidates:")
        print("-" * 90)

        for candidate, label in annotated.items():
            print(
                f"- {candidate} -> {label}"
            )

        printed += 1

        if printed >= 5:
            break

    print()
    print("=" * 90)
    print("Inspection complete.")
    print(f"Occurrences inspected: {printed}")


if __name__ == "__main__":
    main()
