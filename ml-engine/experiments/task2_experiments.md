# Task 2 Experiments — DAMIR Antecedent Resolution

## Dataset

Dataset: DAMIR.xlsx

Official project split:
- Train: 469 occurrences, 4442 candidate rows
- Validation: 59 occurrences, 503 candidate rows
- Test: 59 occurrences, 560 candidate rows
- Split by occurrence ID to prevent data leakage

## Baseline

Model:
- Pre-trained all-MiniLM-L6-v2
- No fine-tuning

Validation:
- Evaluated occurrences: 35
- Top-1 Accuracy: 20.00%
- MRR: 0.4462

## Fine-Tuning Experiments

| Experiment | Configuration | Learning Rate | Epochs | Margin | Batch | Validation Top-1 | Validation MRR |
|---|---|---:|---:|---:|---:|---:|---:|
| Baseline | Pre-trained SBERT | — | — | — | — | 20.00% | 0.4462 |
| EXP-01 | Full model | 2e-5 | 1 | 0.5 | 16 | 71.43% | 0.8333 |
| EXP-02 | Full model | 1e-5 | 1 | 0.5 | 16 | 48.57% | 0.6720 |
| EXP-03 | Full model | 2e-5 | 2 | 0.5 | 16 | 68.57% | 0.8238 |
| EXP-04 | Full model | 2e-5 | 1 | 1.0 | 16 | 71.43% | 0.8429 |
| EXP-05 | Full model | 2e-5 | 1 | 1.0 | 32 | 60.00% | 0.7714 |
| EXP-06 | Full model | 2e-5 | 1 | 0.75 | 16 | 65.71% | 0.8000 |
| EXP-08 | Last 3 layers | 2e-5 | 1 | 1.0 | 16 | 45.71% | 0.6617 |

EXP-07 was not run because all-MiniLM-L6-v2 has exactly 6 transformer layers, so "last 6 layers" is equivalent to full-model fine-tuning.

## Selected Model

EXP-04 was selected because it achieved the strongest validation MRR among the tested configurations.

Configuration:
- Model: all-MiniLM-L6-v2
- Full-model fine-tuning
- Learning rate: 2e-5
- Epochs: 1
- Margin: 1.0
- Batch size: 16
- Loss: ContrastiveLoss

## Final Ranking

The final candidate ranking combines:

- Semantic similarity: 0.9
- Proximity: 0.1
- Syntactic score: 0.0

Final ranking formula:

0.9 × semantic score + 0.1 × proximity score

## Final Test Result

- Test occurrences: 59
- Evaluated occurrences: 30
- Top-1 Accuracy: 73.33%
- MRR: 0.8167

Important: only test occurrences containing at least one `ResolvedAs=correct` candidate are evaluable under the current antecedent-resolution target.

## Ambiguity Detection Limitation

DAMIR's AckUnack annotation does not provide a clean binary ambiguity ground truth.

The dataset is therefore used primarily for candidate antecedent resolution/ranking through the `ResolvedAs` annotations.

The final system should use:
- best candidate antecedent
- ranking score
- competition/confidence signal

rather than claiming a validated binary ambiguity classifier.

## Model Artifact

The final EXP-04 model is stored in Google Drive and is intentionally not committed to GitHub because the model artifact is approximately 87 MB.

Drive location:

MyDrive/Synapse/Task2/EXP-04/

GitHub should contain the implementation, experiment documentation, configuration and evaluation results, while the trained model artifact remains in Drive.
