"""Experiment with simple pronoun-compatible candidate filtering."""

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


SINGULAR_NEUTER = {
    "it",
    "its",
    "itself",
}

PLURAL = {
    "they",
    "their",
    "them",
    "themselves",
}

MASCULINE = {
    "he",
    "his",
    "him",
}

FEMININE = {
    "she",
    "her",
}

FIRST_PERSON_POSSESSIVE = {
    "your",
    "our",
}

INDEFINITE = {
    "one",
}


def compatible_with_pronoun(candidate, pronoun):
    """Return whether a candidate is potentially compatible with a pronoun."""

    text = candidate["text"].lower()

    root = candidate["root"].lower()

    # Keep the candidate's syntactic information available for
    # future ranking improvements.
    _ = text

    if pronoun in SINGULAR_NEUTER:
        return candidate["root_position"] >= 0 and root not in {
            "they",
            "them",
            "themselves",
        }

    if pronoun in PLURAL:
        # Simple heuristic: coordinated nouns and explicit plural
        # noun phrases are preferred for plural pronouns.
        return (
            candidate["dependency"] == "conj"
            or root.endswith("s")
            or " and " in candidate["text"].lower()
        )

    if pronoun in MASCULINE:
        return True

    if pronoun in FEMININE:
        return True

    if pronoun in FIRST_PERSON_POSSESSIVE:
        return True

    if pronoun in INDEFINITE:
        return True

    return True


def main():
    """Compare raw and compatibility-filtered candidates."""

    df = load_damir(DATA_PATH)

    _, validation, _ = split_damir(df)

    occurrence_groups = validation.groupby(
        ["Id", "Context", "Pronoun", "Position", "AckUnack"],
        sort=False,
    )

    inspected = 0

    for (
        occurrence_id,
        context,
        pronoun_text,
        position,
        ack_unack,
    ), group in occurrence_groups:

        pronouns = find_pronouns(context)

        matching = [
            pronoun
            for pronoun in pronouns
            if (
                pronoun["position"] == int(position)
                and pronoun["text"].lower()
                == str(pronoun_text).lower()
            )
        ]

        if not matching:
            continue

        pronoun = matching[0]

        candidates = find_candidate_antecedents(
            context,
            pronoun["position"],
        )

        filtered = [
            candidate
            for candidate in candidates
            if compatible_with_pronoun(
                candidate,
                pronoun["lemma"],
            )
        ]

        annotated = {
            str(row["Candidate Antecedent"]).strip().lower()
            for _, row in group.iterrows()
        }

        print()
        print("=" * 90)
        print(
            f"ID: {occurrence_id} | "
            f"DAMIR: {ack_unack} | "
            f"Pronoun: {pronoun_text}"
        )

        print()
        print(f"Raw candidates: {len(candidates)}")
        print(f"Filtered candidates: {len(filtered)}")

        print()
        print("Filtered candidates:")
        print("-" * 90)

        for candidate in filtered:
            label = (
                "ANNOTATED"
                if candidate["text"].strip().lower()
                in annotated
                else "NOT ANNOTATED"
            )

            print(
                f"- {candidate['text']} "
                f"[dependency={candidate['dependency']}] "
                f"-> {label}"
            )

        inspected += 1

        if inspected >= 5:
            break

    print()
    print("=" * 90)
    print("Candidate filtering experiment complete.")
    print(f"Occurrences inspected: {inspected}")


if __name__ == "__main__":
    main()
