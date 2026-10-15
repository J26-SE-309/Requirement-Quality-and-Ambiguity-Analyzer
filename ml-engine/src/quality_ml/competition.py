"""Uncertainty statistics for scored candidate interpretations."""
import math

def candidate_competition(candidates: list[dict], score_key: str = 'combined_score',
                          temperature: float = 0.1) -> dict:
    if temperature <= 0:
        raise ValueError('temperature must be positive')
    if not candidates:
        return {'top2_margin': None, 'normalized_entropy': None, 'candidate_count': 0}
    scores = sorted((float(c[score_key]) for c in candidates), reverse=True)
    gap = scores[0] - scores[1] if len(scores) > 1 else None
    if len(scores) == 1:
        entropy = 0.0
    else:
        # Softmax gives a distribution; similarity scores are not probabilities.
        exps = [math.exp((s - scores[0]) / temperature) for s in scores]
        probs = [x / sum(exps) for x in exps]
        entropy = -sum(p * math.log(p) for p in probs) / math.log(len(probs))
    return {'top2_margin': gap, 'normalized_entropy': entropy,
            'candidate_count': len(scores)}
