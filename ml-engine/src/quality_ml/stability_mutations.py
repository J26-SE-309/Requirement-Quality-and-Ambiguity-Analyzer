"""Controlled requirement mutations for Task 3 stability analysis."""
import re
from decimal import Decimal, InvalidOperation
import spacy
# --------------------------------------------------
# Shared helpers
# --------------------------------------------------
NLP = spacy.load("en_core_web_sm")


def _valid_text(text: str) -> bool:
    """Check whether input contains usable requirement text."""
    return isinstance(text, str) and bool(text.strip())


def _preserve_case(original: str, replacement: str) -> str:
    """Preserve the capitalization of a replaced expression."""
    if original.isupper():
        return replacement.upper()
    if original[0].isupper():
        return replacement.capitalize()
    return replacement


def _create_mutation(
    text: str,
    start: int,
    end: int,
    replacement: str,
    mutation_type: str,
) -> dict | None:
    """Create a mutation using exact character offsets."""
    original_span = text[start:end]
    if not original_span or original_span == replacement:
        return None
    mutated_text = text[:start] + replacement + text[end:]
    if mutated_text == text:
        return None
    return {
        "mutation_type": mutation_type,
        "original_text": text,
        "mutated_text": mutated_text,
        "original_span": original_span,
        "replacement_span": replacement,
    }


# --------------------------------------------------
# Mutation A — Vague Term Introduction
# --------------------------------------------------
TIME_TO_VAGUE_PATTERN = re.compile(
    r"\bwithin\s+"
    r"(?:\d+(?:\.\d+)?)\s*"
    r"(?:milliseconds?|seconds?|minutes?|hours?)\b",
    re.IGNORECASE,
)


def mutate_vague_term(text: str) -> dict | None:
    """Replace a measurable time limit with vague wording.
    Skip replacements that would create known grammatical errors.
    """
    if not _valid_text(text):
        return None
    match = TIME_TO_VAGUE_PATTERN.search(text)
    if match is None:
        return None
    prefix = text[:match.start()].rstrip()
    # Avoid "shall be quickly".
    if re.search(
        r"\b(?:shall|should|must|will|can|may)\s+be$",
        prefix,
        re.IGNORECASE,
    ):
        return None
    # Avoid "quickly of alarm detection".
    suffix = text[match.end():]
    if re.match(r"\s+of\b", suffix, re.IGNORECASE):
        return None
    replacement = _preserve_case(
        match.group(),
        "quickly",
    )
    return _create_mutation(
        text,
        match.start(),
        match.end(),
        replacement,
        "A_vague_term",
    )


# --------------------------------------------------
# Mutation B — Condition/Scope Modification
# --------------------------------------------------
CONDITION_REPLACEMENTS = {
    "succeeds": "fails",
    "fails": "succeeds",
    "successful": "unsuccessful",
    "unsuccessful": "successful",
    "approved": "rejected",
    "rejected": "approved",
    "valid": "invalid",
    "invalid": "valid",
    "enabled": "disabled",
    "disabled": "enabled",
    "available": "unavailable",
    "unavailable": "available",
    "connected": "disconnected",
    "disconnected": "connected",
    "authenticated": "unauthenticated",
    "unauthenticated": "authenticated",
    "open": "closed",
    "closed": "open",
    "active": "inactive",
    "inactive": "active",
}
CONDITION_MARKERS = {
    "if",
    "when",
    "unless",
    "provided",
}
NEGATION_WORDS = {
    "not",
    "never",
    "no",
    "n't",
}
TECHNICAL_ANNOTATION_PATTERN = re.compile(
    r"\b[A-Za-z]:[A-Za-z_][A-Za-z0-9_]*"
    r"(?:\s+[A-Za-z0-9_]+){0,3}",
    re.IGNORECASE,
)
MENU_OPTION_PATTERN = re.compile(
    r"\b(?:New|Open|Close|Save|Edit|Delete|View|Print|Exit)\s+"
    r"(?:or\s+(?:New|Open|Close|Save|Edit|Delete|View|Print|Exit)\s+)?"
    r"(?:menu\s+)?(?:option|command|item)\b",
    re.IGNORECASE,
)


def _is_negated_condition(token, clause_tokens) -> bool:
    """Check whether a condition contains explicit negation."""
    if any(
        child.dep_ == "neg"
        for child in token.children
    ):
        return True
    if any(
        child.dep_ == "neg"
        for child in token.head.children
    ):
        return True
    return any(
        candidate.lower_ in NEGATION_WORDS
        for candidate in clause_tokens
    )


def _is_unsafe_condition_candidate(candidate, doc) -> bool:
    """Reject some unsafe state-word replacements."""
    # Protect likely technical names.
    if (
        candidate.text[0].isupper()
        and candidate.i + 1 < len(doc)
        and doc[candidate.i + 1].text[0].isupper()
    ):
        return True
    # Protect slash-separated state enumerations.
    if (
        candidate.i + 1 < len(doc)
        and doc[candidate.i + 1].text == "/"
    ):
        return True
    if (
        candidate.i > 0
        and doc[candidate.i - 1].text == "/"
    ):
        return True
    # Avoid "a invalid CAN-signal".
    if (
        candidate.i > 0
        and doc[candidate.i - 1].lower_ in {"a", "an"}
    ):
        return True
    # Protect "open circuit".
    if (
        candidate.i + 1 < len(doc)
        and candidate.lower_ == "open"
        and doc[candidate.i + 1].lower_ == "circuit"
    ):
        return True
    # Avoid changing active/inactive noun modifiers.
    if candidate.lower_ in {"active", "inactive"}:
        if candidate.dep_ in {"amod", "compound"}:
            return True
    # Avoid connected/disconnected followed by "to".
    if candidate.lower_ in {"connected", "disconnected"}:
        if (
            candidate.i + 1 < len(doc)
            and doc[candidate.i + 1].lower_ == "to"
        ):
            return True
    # Protect named user-interface menu commands.
    if candidate.lower_ in {"open", "closed"}:
        for match in MENU_OPTION_PATTERN.finditer(doc.text):
            if (
                match.start() <= candidate.idx
                and candidate.idx + len(candidate.text) <= match.end()
            ):
                return True
    return False


def mutate_condition_scope(text: str) -> dict | None:
    """Modify an explicit state inside a conditional clause.
    Use spaCy dependency subtrees to distinguish the
    condition from the resulting requirement action.
    Reject candidates that match known unsafe patterns.
    """
    if not _valid_text(text):
        return None
    if TECHNICAL_ANNOTATION_PATTERN.search(text):
        return None
    doc = NLP(text)
    for marker in doc:
        if marker.lower_ not in CONDITION_MARKERS:
            continue
        if marker.dep_ != "mark":
            continue
        if marker.lower_ == "provided":
            if (
                marker.i + 1 >= len(doc)
                or doc[marker.i + 1].lower_ != "that"
            ):
                continue
        predicate = marker.head
        if predicate.dep_ != "advcl":
            continue
        clause_tokens = sorted(
            predicate.subtree,
            key=lambda token: token.i,
        )
        if _is_negated_condition(predicate, clause_tokens):
            continue
        clause_text = doc[
            clause_tokens[0].i:clause_tokens[-1].i + 1
        ].text
        if TECHNICAL_ANNOTATION_PATTERN.search(clause_text):
            continue
        for candidate in clause_tokens:
            replacement = CONDITION_REPLACEMENTS.get(
                candidate.lower_
            )
            if replacement is None:
                continue
            if _is_unsafe_condition_candidate(candidate, doc):
                continue
            if candidate.i == marker.i:
                continue

            # Avoid reversing a state repeated elsewhere in the requirement.
            repeated_state = sum(
                token.lower_ == candidate.lower_
                for token in doc
            )
            if repeated_state > 1:
                continue

            # Avoid conflicts with a separately described unsuccessful outcome.
            if (
                candidate.lower_ == "successful"
                and any(token.lower_ == "unsuccessful" for token in doc)
            ):
                continue
            replacement = _preserve_case(
                candidate.text,
                replacement,
            )
            return _create_mutation(
                text,
                candidate.idx,
                candidate.idx + len(candidate.text),
                replacement,
                "B_condition_scope",
            )
    return None


# --------------------------------------------------
# Mutation C — Actor/Role Modification
# --------------------------------------------------
ACTOR_REPLACEMENTS = {
    "administrator": "customer",
    "customer": "administrator",
    "manager": "employee",
    "employee": "manager",
    "teacher": "student",
    "student": "teacher",
    "doctor": "patient",
    "patient": "doctor",
    "buyer": "seller",
    "seller": "buyer",
}
SUBJECT_DEPENDENCIES = {
    "nsubj",
    "nsubjpass",
    "csubj",
    "csubjpass",
}


def mutate_actor_role(text: str) -> dict | None:
    """Replace an explicitly identified grammatical actor.
    Reject replacements that introduce known grammatical
    or referential inconsistencies.
    Protect compound and modified role names.
    """
    if not _valid_text(text):
        return None
    doc = NLP(text)
    for token in doc:
        replacement = ACTOR_REPLACEMENTS.get(
            token.lower_
        )
        if replacement is None:
            continue
        if token.dep_ not in SUBJECT_DEPENDENCIES:
            continue
        if token.pos_ not in {"NOUN", "PROPN"}:
            continue
        # Protect compound and modified role names.
        # Example: "system administrator" -> "system customer".
        if any(
            child.dep_ in {"compound", "amod"}
            for child in token.children
        ):
            continue
        determiners = [
            child
            for child in token.children
            if child.dep_ == "det"
            and child.lower_ in {"the", "a", "an"}
        ]
        if not determiners:
            continue
        # Avoid incorrect articles.
        # Example: "A customer" -> "A administrator".
        if any(
            det.lower_ in {"a", "an"}
            and (
                (
                    det.lower_ == "a"
                    and replacement[0].lower() in "aeiou"
                )
                or (
                    det.lower_ == "an"
                    and replacement[0].lower() not in "aeiou"
                )
            )
            for det in determiners
        ):
            continue
        # Avoid leaving repeated references to the original role.
        original_actor = token.text.lower()
        other_occurrences = [
            other
            for other in doc
            if other.i != token.i
            and other.lower_ == original_actor
        ]
        if other_occurrences:
            continue
        replacement = _preserve_case(
            token.text,
            replacement,
        )
        return _create_mutation(
            text,
            token.idx,
            token.idx + len(token.text),
            replacement,
            "C_actor_role",
        )
    return None


# --------------------------------------------------
# Mutation D — Numeric/Capacity Modification
# --------------------------------------------------
CAPACITY_PATTERN = re.compile(
    r"\b(?:maximum(?:\s+of)?|minimum(?:\s+of)?|"
    r"at\s+least|at\s+most|up\s+to)\s+"
    r"(\d+(?:\.\d+)?)\b",
    re.IGNORECASE,
)


def mutate_numeric_capacity(text: str) -> dict | None:
    """Increase a numeric capacity limit with conservative safeguards.
    Reject percentage limits exceeding 100%, grammatical number
    mismatches, and obvious contradictions with other numeric
    statements in the same requirement.
    """
    if not _valid_text(text):
        return None
    match = CAPACITY_PATTERN.search(text)
    if match is None:
        return None
    original_number = match.group(1)
    try:
        value = Decimal(original_number)
    except InvalidOperation:
        return None
    if not value.is_finite() or value <= 0:
        return None
    new_value = value * Decimal("2")
    # Check the unit immediately after the matched number.
    suffix = text[match.end(1):]
    # FIX: "%" is a non-word character, so it must not
    # rely on the word-boundary rule used for text units.
    unit_match = re.match(
        r"\s*(?:([%])|((?:percent(?:age)?|cars?|users?|digits?)\b))",
        suffix,
        re.IGNORECASE,
    )
    unit = (
        (unit_match.group(1) or unit_match.group(2)).lower()
        if unit_match
        else ""
    )
    # Percentages cannot exceed 100%.
    if unit in {"%", "percent", "percentage"} and new_value > 100:
        return None
    # Avoid singular/plural errors such as "2 car".
    if unit in {"car", "user", "digit"} and new_value != 1:
        return None
    # Avoid changing one numeric value when the same number
    # appears elsewhere in the requirement.
    number_pattern = re.compile(
        rf"(?<![\w.]){re.escape(original_number)}(?![\w.])"
    )
    if len(number_pattern.findall(text)) > 1:
        return None
    # Protect explicitly stated digit-format explanations.
    # Example: "6 numeric digits (XXX.XX 4 digits plus 2 digits)".
    if re.search(
        r"\b(?:digits?|characters?)\b",
        suffix[:50],
        re.IGNORECASE,
    ) and re.search(
        r"\([^)]*\b\d+\s+digits?\b[^)]*\)",
        text,
        re.IGNORECASE,
    ):
        return None
    if "." in original_number:
        decimal_places = len(original_number.split(".")[1])
        replacement_number = f"{new_value:.{decimal_places}f}"
    else:
        replacement_number = str(int(new_value))
    start, end = match.span(1)
    return _create_mutation(
        text,
        start,
        end,
        replacement_number,
        "D_numeric_capacity",
    )


# --------------------------------------------------
# Mutation E — Temporal Constraint Modification
# --------------------------------------------------
TEMPORAL_REPLACEMENTS = {
    "before": "after",
    "after": "before",
}
TEMPORAL_DEPENDENCIES = {
    "prep",
    "mark",
    "advmod",
}


def mutate_temporal_constraint(text: str) -> dict | None:
    """Reverse a temporal ordering constraint conservatively.
    Reject known unsafe technical expressions, malformed
    constructions, and contradictory timing phrases.
    """
    if not _valid_text(text):
        return None
    doc = NLP(text)
    for token in doc:
        replacement = TEMPORAL_REPLACEMENTS.get(
            token.lower_
        )
        if replacement is None:
            continue
        if token.dep_ not in TEMPORAL_DEPENDENCIES:
            continue
        if token.i + 1 >= len(doc):
            continue
        following = doc[token.i + 1]
        if following.pos_ not in {
            "DET",
            "NOUN",
            "PROPN",
            "PRON",
            "VERB",
            "AUX",
            "NUM",
            "ADJ",
        }:
            continue
        if (
            token.dep_ == "amod"
            or (
                token.head.pos_ == "NOUN"
                and token.dep_ not in {"prep", "mark"}
            )
        ):
            continue
        if any(
            child.dep_ == "pobj"
            and child.lower_ in {"example", "comparison"}
            for child in token.children
        ):
            continue
        # Protect technical expressions such as
        # "low after start voltage".
        if (
            token.i > 0
            and doc[token.i - 1].lower_ == "low"
            and following.lower_ == "start"
        ):
            continue
        # Reject malformed constructions such as:
        # "after message lost"
        # "after actuate"
        # "to before provide"
        if (
            following.lower_ == "actuate"
            or (
                following.lower_ == "message"
                and token.i + 2 < len(doc)
                and doc[token.i + 2].lower_ == "lost"
            )
            or (
                token.i > 0
                and doc[token.i - 1].lower_ == "to"
                and following.pos_ == "VERB"
            )
        ):
            continue
        # Avoid conflicting timing expressions such as:
        # "at the earliest before 30 minutes".
        prefix = text[:token.idx].lower()
        if replacement == "before" and re.search(
            r"\bat\s+the\s+earliest\s*$",
            prefix,
        ):
            continue
        # Avoid creating "after passing" when the same
        # requirement also says "no later than passing".
        if (
            replacement == "after"
            and re.search(
                r"\bno\s+later\s+than\b",
                text[token.idx:],
                re.IGNORECASE,
            )
            and following.lemma_ == "pass"
        ):
            continue
        # Avoid collapsing a before/after comparison into
        # repeated identical conditions, e.g.
        # "before posting and after posting"
        # becoming "after posting and after posting".
        suffix = text[token.idx + len(token.text):]
        if replacement == "after" and re.search(
            r"\band\s+(?:only\s+)?after\s+posting\b",
            suffix,
            re.IGNORECASE,
        ):
            continue
        replacement = _preserve_case(
            token.text,
            replacement,
        )
        return _create_mutation(
            text,
            token.idx,
            token.idx + len(token.text),
            replacement,
            "E_temporal_constraint",
        )
    return None
