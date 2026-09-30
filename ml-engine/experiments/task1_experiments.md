# Task 1 — Experiment Record

## EXP-05 — Current Baseline

### Objective

Fine-tune DeBERTa-v3-base for binary requirement-quality classification
using the cleaned QuRE dataset.

### Dataset

- Dataset: QuRE
- Raw rows: 2,187
- Clean unique requirements: 2,089
- Train: 1,671
- Validation: 209
- Test: 209
- Labels:
  - 0 = defect
  - 1 = ok

### Model

- Base model: microsoft/deberta-v3-base
- Task: Binary sequence classification
- Total parameters: 184,423,682
- Trainable parameters: 184,423,682
- All encoder layers trainable: Yes

### Training configuration

- Learning rate: 1e-5
- Epochs: 3
- Training batch size: 4
- Evaluation batch size: 8
- Gradient accumulation steps: 4
- Effective batch size: 16
- Maximum sequence length: 128
- Weight decay: 0.01
- Warmup steps: 100
- Random seed: 42
- Loss: Inverse-frequency class-weighted cross-entropy
- Mixed precision (FP16): No
- Model precision: FP32

### Class weights

Calculated from the training split:

- defect: 1.708589
- ok: 0.706853

### Test results

| Metric | Result |
|---|---:|
| Accuracy | 78.47% |
| Defect Precision | 60.53% |
| Defect Recall | 75.41% |
| Defect F1 | 67.15% |
| OK Precision | 88.72% |
| OK Recall | 79.73% |
| OK F1 | 83.99% |
| Macro F1 | 75.57% |
| Weighted F1 | 79.07% |

### Confusion matrix

| Actual / Predicted | Defect | OK |
|---|---:|---:|
| Defect | 46 | 15 |
| OK | 30 | 118 |

### Model artifact

The trained model was saved in Colab at:

`/content/task1_exp05_fp32_weighted/final_model`

### Result interpretation

EXP-05 successfully fine-tuned DeBERTa-v3-base without the prediction-collapse observed
in the earlier experiments. The model identifies 46 of 61 defect requirements and
118 of 148 OK requirements on the held-out test set.

The primary overall F1 metric recorded for comparison is Macro F1 = 75.57%.


---

## EXP-06 to EXP-09 — Layer-wise Fine-tuning Experiments

### Objective

Following the panel recommendation, controlled experiments were conducted to
investigate how the number of trainable DeBERTa-v3-base encoder layers affects
requirement-quality classification performance.

The independent variable was the set of trainable encoder layers.

All other training conditions were kept constant.

### Controlled variables

- Dataset: cleaned QuRE
- Raw rows: 2,187
- Clean unique requirements: 2,089
- Train: 1,671
- Validation: 209
- Test: 209
- Base model: microsoft/deberta-v3-base
- Learning rate: 1e-5
- Epochs: 3
- Training batch size: 4
- Evaluation batch size: 8
- Gradient accumulation steps: 4
- Effective batch size: 16
- Maximum sequence length: 128
- Weight decay: 0.01
- Warmup steps: 100
- Random seed: 42
- Loss: Inverse-frequency class-weighted cross-entropy
- Mixed precision (FP16): No
- Model precision: FP32

Each experiment was initialized from the original
`microsoft/deberta-v3-base` checkpoint rather than from another experiment.

### Experimental configurations

| Experiment | Trainable encoder layers | Trainable parameters |
|---|---|---:|
| EXP-06 | 0–11 (all layers) | 184,423,682 |
| EXP-07 | 6–11 (last 6 layers) | 43,119,362 |
| EXP-08 | 9–11 (last 3 layers) | 21,855,746 |
| EXP-09 | None (classifier only) | 592,130 |

### Test results

| Experiment | Accuracy | Macro Precision | Macro Recall | Macro F1 | Defect F1 | OK F1 |
|---|---:|---:|---:|---:|---:|---:|
| EXP-06 | 81.34% | 77.46% | 79.60% | 78.32% | 70.23% | 86.41% |
| EXP-07 | 81.82% | 78.68% | 75.60% | 76.83% | 66.07% | 87.58% |
| EXP-08 | 78.47% | 73.95% | 73.23% | 73.57% | 62.18% | 84.95% |
| EXP-09 | 70.81% | 35.41% | 50.00% | 41.46% | 0.00% | 82.91% |

### Confusion matrices

#### EXP-06 — All layers

| Actual / Predicted | Defect | OK |
|---|---:|---:|
| Defect | 46 | 15 |
| OK | 24 | 124 |

#### EXP-07 — Last 6 layers

| Actual / Predicted | Defect | OK |
|---|---:|---:|
| Defect | 37 | 24 |
| OK | 14 | 134 |

#### EXP-08 — Last 3 layers

| Actual / Predicted | Defect | OK |
|---|---:|---:|
| Defect | 37 | 24 |
| OK | 21 | 127 |

#### EXP-09 — Classifier only

| Actual / Predicted | Defect | OK |
|---|---:|---:|
| Defect | 0 | 61 |
| OK | 0 | 148 |

### Observations

1. EXP-06 fine-tuned all 12 encoder layers and achieved 78.32% Macro F1
   and 70.23% Defect F1.

2. EXP-07 fine-tuned only the last 6 encoder layers. It achieved 81.82%
   accuracy while using 43,119,362 trainable parameters.

3. EXP-08 reduced fine-tuning further to the last 3 encoder layers.
   Macro F1 decreased to 73.57% and Defect F1 decreased to 62.18%.

4. EXP-09 trained only the classifier and pooler while keeping all encoder
   layers frozen. The model predicted every test requirement as OK, resulting
   in 0.00% Defect F1.

5. The experiments demonstrate that the amount of encoder fine-tuning has a
   measurable effect on requirement-quality classification performance.

6. Accuracy alone is insufficient for evaluating these experiments because
   the dataset is class-imbalanced and EXP-09 achieved 70.81% accuracy despite
   completely failing to identify the defect class. Macro F1 and Defect F1
   are therefore also reported.

### Experimental conclusion

The layer-wise experiments provide controlled evidence for the panel's request
to investigate different trainable-layer configurations. Reducing the number
of trainable layers substantially reduces the number of updated parameters,
while aggressive freezing of the encoder eventually causes a substantial
decrease in defect-class performance.

The results will be used to justify the fine-tuning configuration selected for
the subsequent Task 1 development and evaluation.
