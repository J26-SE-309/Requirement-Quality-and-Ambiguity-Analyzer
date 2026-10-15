"""Generate candidate interpretations for linguistic references."""

import spacy

from quality_ml.ambiguity_checks import find_candidate_antecedents

NLP = spacy.load("en_core_web_sm")


def build_interpretation(
    text: str,
    reference_start: int,
    reference_end: int,
    candidate_text: str,
) -> str:
    """Replace a pronoun or demonstrative reference with a candidate."""

    replacement = candidate_text

    if text[reference_start:reference_end].islower():
        replacement = candidate_text.lower()

    return (
        text[:reference_start]
        + replacement
        + text[reference_end:]
    )


def build_relative_interpretation(
    text: str,
    reference_position: int,
    candidate_text: str,
) -> str:
    """Build a natural interpretation of a relative reference."""

    doc = NLP(text)

    reference = next(
        (
            token
            for token in doc
            if token.i == reference_position
        ),
        None,
    )

    if reference is None:
        raise ValueError(
            f"No token found at reference position {reference_position}."
        )

    head = reference.head

    # The relative reference is normally attached to the main verb
    # of the relative clause, for example:
    #
    #     which -> validates
    #     who   -> updated
    #
    # Use that verb's dependency subtree rather than its left/right
    # document boundaries. This prevents unrelated surrounding
    # clauses from entering the interpretation.
    clause_tokens = list(head.subtree)

    clause_tokens.sort(
        key=lambda token: token.i
    )

    interpreted_tokens = []

    for token in clause_tokens:
        if token.i == reference.i:
            interpreted_tokens.append(candidate_text)
        else:
            interpreted_tokens.append(token.text)

    return " ".join(interpreted_tokens)


def generate_interpretations(
    text: str,
    reference: dict,
    candidates: list[dict] | None = None,
) -> list[dict]:
    """Generate one interpretation for each candidate antecedent."""

    if not text or not text.strip():
        raise ValueError("Requirement text must not be empty.")

    if candidates is None:
        candidates = find_candidate_antecedents(
            text,
            reference["position"],
        )

    interpretations = []

    for candidate in candidates:
        if reference.get("kind") == "relative":
            interpreted_text = build_relative_interpretation(
                text,
                reference["position"],
                candidate["text"],
            )
        else:
            interpreted_text = build_interpretation(
                text,
                reference["start"],
                reference["end"],
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
