"""spaCy-based structural and vague-term checks for Task 1."""

import spacy

NLP = spacy.load("en_core_web_sm")

# Single-word terms that may be vague in software requirements.
VAGUE_TERMS = {
    "adequate",
    "appropriate",
    "bad",
    "certain",
    "close",
    "common",
    "comparable",
    "convenient",
    "detail",
    "early",
    "easy",
    "effective",
    "efficient",
    "fast",
    "few",
    "flexible",
    "good",
    "large",
    "long",
    "much",
    "near",
    "normal",
    "normally",
    "particular",
    "powerful",
    "quick",
    "quickly",
    "reasonable",
    "reliable",
    "robust",
    "satisfactory",
    "secure",
    "several",
    "short",
    "similar",
    "simple",
    "small",
    "soon",
    "special",
    "sufficient",
    "various",
    "better",
    "best",
    "modern",
    "minimal",
    "maximum",
    "immediately",
    "usually",
}

# Multi-word expressions that may be vague in software requirements.
VAGUE_PHRASES = {
    "as needed",
    "as soon as possible",
    "easy to use",
    "high performance",
    "user friendly",
    "user-friendly",
    "where appropriate",
    "and so on",
    "etc",
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
    """Find potentially vague words and phrases in a requirement."""

    if not text or not text.strip():
        raise ValueError("Requirement text must not be empty.")

    doc = NLP(text)

    detected_terms: list[str] = []

    # Detect single-word vague terms.
    for token in doc:
        if token.is_alpha and token.lemma_.lower() in VAGUE_TERMS:
            term = token.text

            if term.lower() not in {
                detected.lower() for detected in detected_terms
            }:
                detected_terms.append(term)

    # Detect multi-word vague phrases directly from the original text.
    # This handles hyphenated phrases such as "user-friendly".
    normalized_text = " ".join(text.lower().split())

    for phrase in sorted(VAGUE_PHRASES, key=len, reverse=True):
        if phrase in normalized_text:
            phrase_start = normalized_text.find(phrase)
            phrase_end = phrase_start + len(phrase)

            # Avoid matching a phrase inside a larger word.
            before_ok = (
                phrase_start == 0
                or not normalized_text[phrase_start - 1].isalnum()
            )
            after_ok = (
                phrase_end == len(normalized_text)
                or not normalized_text[phrase_end].isalnum()
            )

            if before_ok and after_ok:
                display_phrase = text[
                    phrase_start:phrase_end
                ]

                if display_phrase.lower() not in {
                    detected.lower() for detected in detected_terms
                }:
                    detected_terms.append(display_phrase)

    return detected_terms
