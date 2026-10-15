"""spaCy-based linguistic reference and ambiguity analysis."""

import spacy

NLP = spacy.load("en_core_web_sm")


REFERENCE_PRONOUNS = {
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

DEMONSTRATIVE_REFERENCES = {
    "this",
    "that",
    "these",
    "those",
}

RELATIVE_REFERENCES = {
    "which",
    "who",
    "whom",
    "whose",
}


def _load_doc(text: str):
    """Parse requirement text with spaCy."""

    if not text or not text.strip():
        raise ValueError("Requirement text must not be empty.")

    return NLP(text)


def _reference_kind(token) -> str | None:
    """Classify a token as a linguistic reference expression."""

    lemma = token.lemma_.lower()
    text = token.text.lower()

    if token.pos_ == "PRON" and lemma in REFERENCE_PRONOUNS:
        return "pronoun"

    if (
        text in DEMONSTRATIVE_REFERENCES
        and token.pos_ in {"PRON", "DET"}
    ):
        return "demonstrative"

    if (
        text in RELATIVE_REFERENCES
        and token.pos_ in {"PRON", "DET"}
    ):
        return "relative"

    return None


def find_reference_expressions(text: str) -> list[dict]:
    """Find pronoun, demonstrative, and relative reference expressions."""

    doc = _load_doc(text)
    references = []

    for token in doc:
        kind = _reference_kind(token)

        if kind is None:
            continue

        references.append(
            {
                "text": token.text,
                "lemma": token.lemma_.lower(),
                "start": token.idx,
                "end": token.idx + len(token.text),
                "position": token.i,
                "kind": kind,
                "part_of_speech": token.pos_,
                "dependency": token.dep_,
                "head": token.head.text,
                "head_position": token.head.i,
            }
        )

    return references


def find_pronouns(text: str) -> list[dict]:
    """Find traditional pronouns for backward compatibility."""

    return [
        reference
        for reference in find_reference_expressions(text)
        if reference["kind"] == "pronoun"
    ]


def _is_before_reference(span, reference_position: int) -> bool:
    """Return whether a span occurs before the reference."""

    return span.root.i < reference_position


def _is_valid_candidate_root(token) -> bool:
    """Return whether a token can act as an antecedent candidate root."""

    return _reference_kind(token) is None


def _add_candidate(
    candidates: list[dict],
    text: str,
    start: int,
    end: int,
    root: str,
    root_position: int,
    dependency: str,
    is_plural: bool | None = None,
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

    if is_plural is not None:
        candidate["is_plural"] = is_plural

    if not any(
        existing["start"] == start and existing["end"] == end
        for existing in candidates
    ):
        candidates.append(candidate)


def _is_plural_noun(token) -> bool:
    """Return whether spaCy identifies a noun as plural."""

    return token.tag_ in {"NNS", "NNPS"}


def _is_human_candidate(token) -> bool:
    """Return whether a candidate is likely to refer to a person."""

    if token.ent_type_ == "PERSON":
        return True

    human_nouns = {
        "administrator",
        "analyst",
        "client",
        "customer",
        "developer",
        "director",
        "engineer",
        "manager",
        "member",
        "operator",
        "person",
        "programmer",
        "reviewer",
        "staff",
        "student",
        "supervisor",
        "tester",
        "user",
        "worker",
    }

    return token.lemma_.lower() in human_nouns


def _candidate_features(
    token,
    *,
    is_plural: bool | None = None,
) -> dict:
    """Extract linguistic features used by candidate ranking."""

    if is_plural is None:
        is_plural = _is_plural_noun(token)

    return {
        "is_plural": is_plural,
        "is_human": _is_human_candidate(token),
        "pos": token.pos_,
        "tag": token.tag_,
        "dependency": token.dep_,
    }


def _find_noun_chunk_candidates(
    doc,
    reference_position: int,
) -> list[dict]:
    """Extract noun chunks occurring before a reference expression."""

    candidates = []

    for chunk in doc.noun_chunks:
        if not _is_before_reference(chunk, reference_position):
            continue

        if not _is_valid_candidate_root(chunk.root):
            continue

        features = _candidate_features(chunk.root)

        _add_candidate(
            candidates,
            chunk.text,
            chunk.start_char,
            chunk.end_char,
            chunk.root.text,
            chunk.root.i,
            chunk.root.dep_,
            is_plural=features["is_plural"],
        )

        candidates[-1]["is_human"] = features["is_human"]
        candidates[-1]["pos"] = features["pos"]
        candidates[-1]["tag"] = features["tag"]

    return candidates


def _find_conjunct_candidates(
    doc,
    reference_position: int,
) -> list[dict]:
    """Extract coordinated noun phrases before a reference expression."""

    candidates = []

    for token in doc:
        if token.i >= reference_position:
            continue

        if token.pos_ != "NOUN" or token.dep_ != "conj":
            continue

        if not _is_valid_candidate_root(token):
            continue

        head = token.head

        if head.i >= reference_position:
            continue

        if not _is_valid_candidate_root(head):
            continue

        start = min(
            head.left_edge.i,
            token.left_edge.i,
        )

        end = max(
            head.right_edge.i,
            token.right_edge.i,
        ) + 1

        span = doc[start:end]

        _add_candidate(
            candidates,
            span.text,
            span.start_char,
            span.end_char,
            token.text,
            token.i,
            "conj",
            is_plural=True,
        )

        candidates[-1]["is_human"] = (
            _is_human_candidate(head)
            or _is_human_candidate(token)
        )
        candidates[-1]["pos"] = "NOUN"
        candidates[-1]["tag"] = "NNS"

    return candidates


def find_candidate_antecedents(
    text: str,
    reference_position: int,
) -> list[dict]:
    """Find linguistic noun-phrase candidates for a reference expression."""

    doc = _load_doc(text)

    candidates = _find_noun_chunk_candidates(
        doc,
        reference_position,
    )

    for candidate in _find_conjunct_candidates(
        doc,
        reference_position,
    ):
        _add_candidate(
            candidates,
            candidate["text"],
            candidate["start"],
            candidate["end"],
            candidate["root"],
            candidate["root_position"],
            candidate["dependency"],
            is_plural=candidate.get("is_plural"),
        )

        existing = next(
            item
            for item in candidates
            if (
                item["start"] == candidate["start"]
                and item["end"] == candidate["end"]
            )
        )

        existing["is_human"] = candidate.get(
            "is_human",
            False,
        )
        existing["pos"] = candidate.get(
            "pos",
            "NOUN",
        )
        existing["tag"] = candidate.get(
            "tag",
            "NOUN",
        )

    candidates.sort(
        key=lambda candidate: candidate["start"]
    )

    return candidates


def _dependency_path_distance(
    reference_token,
    candidate_token,
) -> int:
    """Return the shortest dependency-tree path length between two tokens."""

    reference_ancestors = {
        reference_token.i: 0,
    }

    current = reference_token
    distance = 0

    while current.head.i != current.i:
        current = current.head
        distance += 1
        reference_ancestors[current.i] = distance

    current = candidate_token
    distance_from_candidate = 0

    if current.i in reference_ancestors:
        return reference_ancestors[current.i]

    while current.head.i != current.i:
        current = current.head
        distance_from_candidate += 1

        if current.i in reference_ancestors:
            return (
                distance_from_candidate
                + reference_ancestors[current.i]
            )

    return distance_from_candidate + reference_ancestors.get(
        current.i,
        0,
    )


def _dependency_proximity(
    reference_token,
    candidate_token,
) -> float:
    """Convert dependency-tree distance into a normalized proximity score."""

    distance = _dependency_path_distance(
        reference_token,
        candidate_token,
    )

    return 1.0 / (1.0 + distance)


def get_reference_compatibility(
    reference: dict,
    candidate: dict,
) -> dict:
    """Calculate general linguistic compatibility between reference and candidate."""

    reference_text = reference["text"].lower()

    candidate_is_human = candidate.get(
        "is_human",
        False,
    )

    candidate_is_plural = candidate.get(
        "is_plural",
        False,
    )

    singular_pronouns = {
        "he",
        "him",
        "his",
        "she",
        "her",
        "hers",
        "it",
        "its",
        "itself",
    }

    plural_pronouns = {
        "they",
        "them",
        "their",
        "theirs",
        "themselves",
        "these",
        "those",
    }

    human_pronouns = {
        "he",
        "him",
        "his",
        "she",
        "her",
        "hers",
        "they",
        "them",
        "their",
        "theirs",
        "themselves",
    }

    inanimate_pronouns = {
        "it",
        "its",
        "itself",
    }

    relative_human = {
        "who",
        "whom",
        "whose",
    }

    number_score = 0.5

    if reference_text in plural_pronouns:
        number_score = 1.0 if candidate_is_plural else 0.0
    elif reference_text in singular_pronouns:
        number_score = 0.0 if candidate_is_plural else 1.0

    human_score = 0.5

    if reference_text in human_pronouns:
        human_score = 1.0 if candidate_is_human else 0.0
    elif reference_text in inanimate_pronouns:
        human_score = 0.0 if candidate_is_human else 1.0
    elif reference_text in relative_human:
        human_score = 1.0 if candidate_is_human else 0.5

    return {
        "number": number_score,
        "human": human_score,
        "overall": (
            0.6 * number_score
            + 0.4 * human_score
        ),
    }


def get_ambiguity_signals(
    text: str,
    reference_position: int,
) -> dict:
    """Collect linguistic ambiguity signals for a reference expression."""

    doc = _load_doc(text)

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

    reference_kind = _reference_kind(reference)

    if reference_kind is None:
        raise ValueError(
            f"Token at position {reference_position} "
            "is not a supported reference expression."
        )

    candidates = find_candidate_antecedents(
        text,
        reference_position,
    )

    lexical_ambiguity = len(candidates) > 1

    syntactic_ambiguity = len(
        {
            candidate["root_position"]
            for candidate in candidates
        }
    ) > 1

    anaphoric_ambiguity = (
        reference.dep_
        in {
            "nsubj",
            "nsubjpass",
            "dobj",
            "obj",
            "poss",
            "pobj",
            "attr",
            "relcl",
        }
        and len(candidates) > 1
    )

    reference_data = {
        "text": reference.text,
        "lemma": reference.lemma_.lower(),
        "kind": reference_kind,
        "dependency": reference.dep_,
        "head": reference.head.text,
    }

    for candidate in candidates:
        candidate["compatibility"] = get_reference_compatibility(
            reference_data,
            candidate,
        )

        candidate_token = next(
            (
                token
                for token in doc
                if token.i == candidate["root_position"]
            ),
            None,
        )

        if candidate_token is not None:
            candidate["dependency_proximity"] = (
                _dependency_proximity(
                    reference,
                    candidate_token,
                )
            )
        else:
            candidate["dependency_proximity"] = 0.0

    return {
        "reference": reference.text,
        "reference_lemma": reference.lemma_.lower(),
        "reference_kind": reference_kind,
        "reference_position": reference.i,
        "reference_dependency": reference.dep_,
        "reference_head": reference.head.text,

        # Backward compatibility with the original pronoun-based API.
        "pronoun": (
            reference.text
            if reference_kind == "pronoun"
            else None
        ),

        "candidates": candidates,

        "signals": {
            "lexical": lexical_ambiguity,
            "syntactic": syntactic_ambiguity,
            "anaphoric": anaphoric_ambiguity,
        },
    }
