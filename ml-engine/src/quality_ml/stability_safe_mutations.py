
"""Generate candidate meaning-preserving requirement mutations.

These mutations are unlabeled candidates and require human review.
"""

import re


SAFE_REPLACEMENTS = (
    (
        re.compile(r"\bshall be able to\b", re.IGNORECASE),
        "shall have the capability to",
    ),
    (
        re.compile(r"\bshall provide the ability to\b", re.IGNORECASE),
        "shall provide the capability to",
    ),
)


def mutate_safe_paraphrase(text: str) -> dict | None:
    """Produce one conservative paraphrase candidate."""

    if not isinstance(text, str) or not text.strip():
        return None

    for pattern, replacement in SAFE_REPLACEMENTS:
        match = pattern.search(text)

        if match is None:
            continue

        original_span = match.group(0)

        mutated_text = (
            text[:match.start()]
            + replacement
            + text[match.end():]
        )

        if mutated_text == text:
            continue

        return {
            "mutation_type": "F_safe_paraphrase",
            "original_text": text,
            "mutated_text": mutated_text,
            "original_span": original_span,
            "replacement_span": replacement,
        }

    return None
