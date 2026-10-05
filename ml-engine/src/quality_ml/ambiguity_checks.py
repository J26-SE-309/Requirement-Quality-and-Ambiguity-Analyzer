"""spaCy-based candidate and ambiguity checks for requirement text."""

import spacy

NLP = spacy.load("en_core_web_sm")

PRONOUNS = {
    "it",
    "its",
    "itself",
    "they",
    "their",
    "them",
    "themselves",
    "he",
    "his",
    "him",
    "she",
    "her",
    "hers",
    "your",
    "our",
    "one",
}


def find_pronouns(text: str) -> list[dict]:
    """Find pronouns and their positions in the requirement."""

    if not text or not text.strip():
        raise ValueError("Requirement text must not be empty.")

    doc = NLP(text)

    return [
        {
            "text": token.text,
            "lemma": token.lemma_.lower(),
            "start": token.idx,
            "end": token.idx + len(token.text),
            "position": token.i,
        }
        for token in doc
        if token.pos_ == "PRON" and token.lemma_.lower() in PRONOUNS
    ]


def _is_before_pronoun(span, pronoun_position: int) -> bool:
    """Return whether a span occurs before the pronoun."""

    return span.root.i < pronoun_position


def _add_candidate(
    candidates: list[dict],
    text: str,
    start: int,
    end: int,
    root: str,
    root_position: int,
    dependency: str,
) -> None:
    """Add a candidate unless an identical span already exists."""

    candidate = {
        "text": text,
        "start": start,
        "end": end,
        "root": root,
        "root_position": root_position,
        "dependency": dependency,
    }

    if not any(
        existing["start"] == start and existing["end"] == end
        for existing in candidates
    ):
        candidates.append(candidate)


def _find_noun_chunk_candidates(
    doc,
    pronoun_position: int,
) -> list[dict]:
    """Extract noun chunks occurring before the pronoun."""

    candidates = []

    for chunk in doc.noun_chunks:
        if not _is_before_pronoun(chunk, pronoun_position):
            continue

        _add_candidate(
            candidates,
            chunk.text,
            chunk.start_char,
            chunk.end_char,
            chunk.root.text,
            chunk.root.i,
            chunk.root.dep_,
        )

    return candidates


def _find_conjunct_candidates(
    doc,
    pronoun_position: int,
) -> list[dict]:
    """Extract coordinated noun phrases occurring before the pronoun."""

    candidates = []

    for token in doc:
        if token.i >= pronoun_position:
            continue

        if token.pos_ != "NOUN" or token.dep_ != "conj":
            continue

        head = token.head

        if head.i >= pronoun_position:
            continue

        start = min(head.left_edge.i, token.left_edge.i)
        end = max(head.right_edge.i, token.right_edge.i) + 1

        span = doc[start:end]

        _add_candidate(
            candidates,
            span.text,
            span.start_char,
            span.end_char,
            token.text,
            token.i,
            token.dep_,
        )

    return candidates


def find_candidate_antecedents(
    text: str,
    pronoun_position: int,
) -> list[dict]:
    """Find noun-phrase and coordinated candidates for a pronoun."""

    if not text or not text.strip():
        raise ValueError("Requirement text must not be empty.")

    doc = NLP(text)

    candidates = _find_noun_chunk_candidates(
        doc,
        pronoun_position,
    )

    for candidate in _find_conjunct_candidates(
        doc,
        pronoun_position,
    ):
        _add_candidate(
            candidates,
            candidate["text"],
            candidate["start"],
            candidate["end"],
            candidate["root"],
            candidate["root_position"],
            candidate["dependency"],
        )

    candidates.sort(key=lambda candidate: candidate["start"])

    return candidates


def get_ambiguity_signals(
    text: str,
    pronoun_position: int,
) -> dict:
    """Collect lexical, syntactic and anaphoric ambiguity signals."""

    if not text or not text.strip():
        raise ValueError("Requirement text must not be empty.")

    doc = NLP(text)

    pronoun = next(
        (
            token
            for token in doc
            if token.i == pronoun_position
        ),
        None,
    )

    if pronoun is None:
        raise ValueError(
            f"No token found at pronoun position {pronoun_position}."
        )

    candidates = find_candidate_antecedents(
        text,
        pronoun_position,
    )

    lexical_ambiguity = len(candidates) > 1

    syntactic_ambiguity = len(
        {
            candidate["root_position"]
            for candidate in candidates
        }
    ) > 1

    anaphoric_ambiguity = (
        pronoun.dep_
        in {
            "nsubj",
            "nsubjpass",
            "dobj",
            "obj",
            "poss",
            "pobj",
        }
        and len(candidates) > 1
    )

    return {
        "pronoun": pronoun.text,
        "pronoun_lemma": pronoun.lemma_.lower(),
        "candidates": candidates,
        "signals": {
            "lexical": lexical_ambiguity,
            "syntactic": syntactic_ambiguity,
            "anaphoric": anaphoric_ambiguity,
        },
    }
