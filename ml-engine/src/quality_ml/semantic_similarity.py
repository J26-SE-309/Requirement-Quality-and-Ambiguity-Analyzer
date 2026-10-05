"""SBERT-based semantic similarity utilities for ambiguity analysis."""

from functools import lru_cache

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def load_embedding_model(
    model_name: str = DEFAULT_MODEL_NAME,
) -> SentenceTransformer:
    """Load and cache the sentence-transformer model."""

    return SentenceTransformer(model_name)


def calculate_similarity(
    text_a: str,
    text_b: str,
    model_name: str = DEFAULT_MODEL_NAME,
) -> float:
    """Calculate cosine similarity between two texts."""

    if not text_a or not text_a.strip():
        raise ValueError("First text must not be empty.")

    if not text_b or not text_b.strip():
        raise ValueError("Second text must not be empty.")

    model = load_embedding_model(model_name)

    embeddings = model.encode(
        [text_a, text_b],
        normalize_embeddings=True,
    )

    return float(
        cosine_similarity(
            [embeddings[0]],
            [embeddings[1]],
        )[0][0]
    )


def rank_candidates(
    original_text: str,
    interpretations: list[dict],
    model_name: str = DEFAULT_MODEL_NAME,
) -> list[dict]:
    """Score and rank candidate interpretations by semantic similarity."""

    if not original_text or not original_text.strip():
        raise ValueError("Original text must not be empty.")

    if not interpretations:
        return []

    model = load_embedding_model(model_name)

    texts = [
        interpretation["text"]
        for interpretation in interpretations
    ]

    embeddings = model.encode(
        [original_text, *texts],
        normalize_embeddings=True,
    )

    original_embedding = embeddings[0]
    interpretation_embeddings = embeddings[1:]

    scores = cosine_similarity(
        [original_embedding],
        interpretation_embeddings,
    )[0]

    ranked = []

    for interpretation, score in zip(
        interpretations,
        scores,
    ):
        ranked.append(
            {
                **interpretation,
                "similarity": float(score),
            }
        )

    ranked.sort(
        key=lambda candidate: candidate["similarity"],
        reverse=True,
    )

    for rank, candidate in enumerate(ranked, start=1):
        candidate["rank"] = rank

    return ranked


def similarity_gap(ranked_candidates: list[dict]) -> float:
    """Return the similarity gap between the top two candidates."""

    if len(ranked_candidates) < 2:
        return 1.0

    return (
        ranked_candidates[0]["similarity"]
        - ranked_candidates[1]["similarity"]
    )
