"""Diagnose ambiguity signals on the DAMIR validation split."""

from pathlib import Path

from quality_ml.damir import load_damir, split_damir
from quality_ml.task2 import Task2AmbiguityAnalyzer

DATA_PATH = (
    Path(__file__).resolve().parents[3]
    / "Datasets"
    / "requirement-quality"
    / "DAMIR.xlsx"
)


def main():
    """Print validation ambiguity signals for diagnosis."""

    print("Loading DAMIR...")
    df = load_damir(DATA_PATH)

    print("Creating the fixed train/validation/test split...")
    _, validation, _ = split_damir(df)

    analyzer = Task2AmbiguityAnalyzer()

    rows = []

    occurrence_groups = validation.groupby(
        ["Id", "Context", "Pronoun", "Position", "AckUnack"],
        sort=False,
    )

    for (
        occurrence_id,
        context,
        pronoun,
        position,
        ack_unack,
    ), _ in occurrence_groups:

        result = analyzer.analyze(context)

        matching = [
            item
            for item in result["pronouns"]
            if (
                item["position"] == int(position)
                and item["pronoun"].lower()
                == str(pronoun).lower()
            )
        ]

        if not matching:
            continue

        analysis = matching[0]

        rows.append(
            {
                "id": occurrence_id,
                "label": ack_unack,
                "pronoun": pronoun,
                "candidates": analysis["candidate_count"],
                "gap": round(
                    analysis["similarity_gap"],
                    4,
                ),
                "lexical": analysis["signals"]["lexical"],
                "syntactic": analysis["signals"]["syntactic"],
                "semantic": analysis["signals"]["semantic"],
                "anaphoric": analysis["signals"]["anaphoric"],
                "ambiguous": analysis["is_ambiguous"],
            }
        )

    print()
    print("Validation ambiguity diagnostics")
    print("================================")
    print(f"Evaluated occurrences: {len(rows)}")

    print()
    print(
        "ID | DAMIR | Pronoun | Candidates | Gap | "
        "Lexical | Syntactic | Semantic | Anaphoric | Prediction"
    )
    print("-" * 115)

    for row in rows:
        print(
            f"{row['id']!s:<8} | "
            f"{row['label']!s:<11} | "
            f"{row['pronoun']!s:<7} | "
            f"{row['candidates']:<10} | "
            f"{row['gap']:<6} | "
            f"{row['lexical']!s:<7} | "
            f"{row['syntactic']!s:<9} | "
            f"{row['semantic']!s:<8} | "
            f"{row['anaphoric']!s:<9} | "
            f"{row['ambiguous']!s}"
        )

    print()
    print("Signal summary")
    print("==============")

    signal_names = [
        "lexical",
        "syntactic",
        "semantic",
        "anaphoric",
        "ambiguous",
    ]

    for signal in signal_names:
        true_count = sum(
            row[signal]
            for row in rows
        )

        print(
            f"{signal:<10}: "
            f"{true_count}/{len(rows)}"
        )

    print()
    print("DAMIR labels")
    print("============")

    for label in [
        "Unambiguous",
        "Ack",
        "Unack",
    ]:
        count = sum(
            str(row["label"]).lower()
            == label.lower()
            for row in rows
        )

        print(
            f"{label:<12}: {count}"
        )


if __name__ == "__main__":
    main()
