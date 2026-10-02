"""spaCy-based structural and vague-term checks for Task 1."""

import spacy

NLP = spacy.load("en_core_web_sm")

VAGUE_TERMS = {
    "adequate",
    "appropriate",
    "bad",
    "certain",
    "close",
    "common",
    "comparable",
    "detail",
    "early",
    "fast",
    "few",
    "large",
    "long",
    "much",
    "near",
    "particular",
    "several",
    "short",
    "similar",
    "small",
    "soon",
    "special",
    "various",
    "good",
}


def check_structural_completeness(text: str) -> dict:
    """Check whether a requirement contains basic structural elements."""

    if not text or not text.strip():
        raise ValueError("Requirement text must not be empty.")

    doc = NLP(text)

    has_subject = any(
        token.dep_ in {"nsubj", "nsubjpass", "csubj"}
        for token in doc
    )

    requirement_modals = {"shall", "must", "should", "will", "required"}

    has_requirement_modal = any(
        token.lemma_.lower() in requirement_modals
        for token in doc
    )

    has_action_verb = any(
        token.pos_ in {"VERB", "AUX"}
        and token.lemma_.lower() not in requirement_modals
        for token in doc
    )

    has_object_or_complement = any(
        token.dep_ in {
            "dobj",
            "obj",
            "attr",
            "oprd",
            "acomp",
            "xcomp",
            "ccomp",
        }
        for token in doc
    )

    checks = {
        "subject": has_subject,
        "requirement_modal": has_requirement_modal,
        "action": has_action_verb,
        "object_or_complement": has_object_or_complement,
    }

    missing_elements = [
        name.replace("_", " ")
        for name, present in checks.items()
        if not present
    ]

    completeness_score = sum(checks.values()) / len(checks)

    return {
        "complete": completeness_score == 1.0,
        "score": completeness_score,
        "checks": checks,
        "missing_elements": missing_elements,
    }


def find_vague_terms(text: str) -> list[str]:
    """Find vague terms from the requirement text."""

    if not text or not text.strip():
        raise ValueError("Requirement text must not be empty.")

    doc = NLP(text)

    return [
        token.text
        for token in doc
        if token.is_alpha and token.lemma_.lower() in VAGUE_TERMS
    ]
