"""Generate candidate interpretations for ambiguous pronouns."""

from quality_ml.ambiguity_checks import find_candidate_antecedents


def build_interpretation(
    text: str,
    pronoun_start: int,
    pronoun_end: int,
    candidate_text: str,
) -> str:
    """Replace a pronoun with a candidate antecedent."""

    replacement = candidate_text

    if text[pronoun_start:pronoun_end].islower():
        replacement = candidate_text.lower()

    return (
        text[:pronoun_start]
        + replacement
        + text[pronoun_end:]
    )


def generate_interpretations(
    text: str,
    pronoun: dict,
    candidates: list[dict] | None = None,
) -> list[dict]:
    """Generate one interpretation for each candidate antecedent."""

    if not text or not text.strip():
        raise ValueError("Requirement text must not be empty.")

    if candidates is None:
        candidates = find_candidate_antecedents(
            text,
            pronoun["position"],
        )

    interpretations = []

    for candidate in candidates:
        interpreted_text = build_interpretation(
            text,
            pronoun["start"],
            pronoun["end"],
            candidate["text"],
        )

        interpretations.append(
            {
                "candidate": candidate["text"],
                "candidate_start": candidate["start"],
                "candidate_end": candidate["end"],
                "text": interpreted_text,
            }
        )

    return interpretations
