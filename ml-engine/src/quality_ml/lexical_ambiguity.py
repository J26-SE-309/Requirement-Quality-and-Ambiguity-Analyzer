
"""WordNet sense competition: no new model training or lexical trigger lists."""

from nltk.corpus import wordnet as wn

from .semantic_similarity import load_embedding_model
from .competition import candidate_competition


POS_MAP = {
    "NOUN": wn.NOUN,
    "VERB": wn.VERB,
    "ADJ": wn.ADJ,
    "ADV": wn.ADV,
}

# Experimental threshold for lexical sense competition.
# This is not a validated probability or final ambiguity decision.
LEXICAL_MARGIN_THRESHOLD = 0.05


def analyze_word_senses(
    text: str,
    nlp,
    max_senses: int = 12,
) -> list[dict]:
    """Rank WordNet senses using SBERT and screen for close competition."""

    doc = nlp(text)
    model = load_embedding_model()
    terms = []

    for token in doc:
        # Consider content words only.
        if (
            token.pos_ not in POS_MAP
            or token.is_stop
            or not token.is_alpha
        ):
            continue

        # Retrieve WordNet senses for the word's lemma and part of speech.
        senses = wn.synsets(
            token.lemma_.lower(),
            pos=POS_MAP[token.pos_],
        )[:max_senses]

        if len(senses) < 2:
            continue

        # Build natural-language descriptions for semantic comparison.
        descriptions = [
            sense.definition()
            + (
                " Examples: " + " ".join(sense.examples())
                if sense.examples()
                else ""
            )
            for sense in senses
        ]

        # Compare the requirement with each WordNet sense description.
        # These similarities are heuristic scores, not WSD probabilities.
        embeddings = model.encode(
            [text, *descriptions],
            normalize_embeddings=True,
        )

        similarities = embeddings[1:] @ embeddings[0]

        # Rank senses by contextual semantic similarity.
        ranked = sorted(
            (
                {
                    "sense": sense.name(),
                    "definition": sense.definition(),
                    "similarity": float(similarity),
                    "combined_score": float(similarity),
                }
                for sense, similarity in zip(senses, similarities)
            ),
            key=lambda item: item["similarity"],
            reverse=True,
        )

        # EXP-04 candidate competition metrics.
        competition = candidate_competition(ranked)

        margin = competition["top2_margin"]

        # Experimental screening flag:
        # two closely ranked senses may indicate lexical ambiguity.
        # A close margin alone does not establish that either sense
        # is genuinely plausible in this requirement.
        is_ambiguous = (
            competition["candidate_count"] >= 2
            and margin is not None
            and margin <= LEXICAL_MARGIN_THRESHOLD
        )

        terms.append(
            {
                "term": token.text,
                "span": {
                    "start": token.idx,
                    "end": token.idx + len(token.text),
                },
                "candidate_senses": ranked,
                **competition,
                "is_ambiguous": is_ambiguous,
            }
        )

    return terms
