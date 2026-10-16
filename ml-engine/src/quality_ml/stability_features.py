"""Pairwise features for Task 3 requirement interpretation stability.

This module does not load an embedding model. Embeddings must be produced
in the approved Google Colab training environment or loaded as artifacts.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence

import numpy as np

TOKEN_RE = re.compile(r"\b[a-zA-Z]+(?:[-'][a-zA-Z]+)*\b|\b\d+(?:\.\d+)?\b")
NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?\b")
MODALS = frozenset({"shall", "must", "should", "may",
                   "might", "can", "could", "will", "would"})
NEGATIONS = frozenset(
    {"not", "never", "no", "without", "cannot", "neither", "nor"})
CONDITIONS = frozenset(
    {"if", "unless", "when", "whenever", "provided", "assuming"})
TEMPORAL = frozenset({"before", "after", "until", "while", "once", "during"})
COMPARATORS = frozenset({"at", "least", "most", "more",
                        "less", "than", "within", "minimum", "maximum", "exactly"})

FEATURE_NAMES = (
    "semantic_cosine",
    "token_jaccard",
    "token_length_ratio",
    "token_count_difference",
    "number_changed",
    "number_count_difference",
    "modal_changed",
    "negation_changed",
    "condition_changed",
    "temporal_changed",
    "comparator_changed",
)


def _tokens(text: str) -> list[str]:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Both requirement texts must be nonempty strings.")
    return [match.group(0).lower() for match in TOKEN_RE.finditer(text)]


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if a.ndim != 1 or b.ndim != 1 or a.shape != b.shape or not a.size:
        raise ValueError(
            "Embeddings must be nonempty vectors of equal dimension.")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Embeddings must contain finite values.")
    norm = float(np.linalg.norm(a) * np.linalg.norm(b))
    if norm == 0:
        raise ValueError("Embeddings must have nonzero norms.")
    return float(np.clip(np.dot(a, b) / norm, -1.0, 1.0))


def extract_stability_features(
    original_text: str,
    reworded_text: str,
    original_embedding: Sequence[float],
    reworded_embedding: Sequence[float],
) -> dict[str, float]:
    """Extract deterministic comparison features from texts and supplied embeddings.

    No model loading, training, or embedding generation occurs here.
    """
    original = _tokens(original_text)
    reworded = _tokens(reworded_text)
    a, b = set(original), set(reworded)

    def changed(vocabulary: frozenset[str]) -> float:
        return float((Counter(t for t in original if t in vocabulary)) !=
                     (Counter(t for t in reworded if t in vocabulary)))

    original_numbers = NUMBER_RE.findall(original_text)
    reworded_numbers = NUMBER_RE.findall(reworded_text)
    result = {
        "semantic_cosine": _cosine(original_embedding, reworded_embedding),
        "token_jaccard": len(a & b) / len(a | b) if a | b else 1.0,
        "token_length_ratio": min(len(original), len(reworded)) / max(len(original), len(reworded)),
        "token_count_difference": float(abs(len(original) - len(reworded))),
        "number_changed": float(Counter(original_numbers) != Counter(reworded_numbers)),
        "number_count_difference": float(abs(len(original_numbers) - len(reworded_numbers))),
        "modal_changed": changed(MODALS),
        "negation_changed": changed(NEGATIONS),
        "condition_changed": changed(CONDITIONS),
        "temporal_changed": changed(TEMPORAL),
        "comparator_changed": changed(COMPARATORS),
    }
    return {name: float(result[name]) for name in FEATURE_NAMES}
