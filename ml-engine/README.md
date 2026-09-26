# ML engine: Requirement Quality and Ambiguity Analyzer

Research code for this component: preparing datasets, training and evaluating models.
The API in [`../backend`](../backend) loads the trained models to serve results to the platform.

## Planned work (from the proposal)

- **Task 1 – Requirement quality assessment:** fine-tune DeBERTa-v3 on the QURE dataset to classify quality and identify quality issues.
- **Task 2 – Ambiguity and vagueness detection:** SBERT embeddings, spaCy linguistic analysis and semantic similarity (DAMIR dataset plus annotated requirements).
- **Task 3 – Interpretation stability:** generate controlled lexical and syntactic mutations (WordNet, spaCy), extract SBERT and structural features, classify stability with XGBoost.

## Layout

```
ml-engine/
├── src/quality_ml/   # reusable code: data loading, features, models, evaluation
├── notebooks/        # exploration only; move anything reusable into src/
└── tests/
```

## Data and models

Never commit datasets or trained models. Keep them in `AgilePlatform/Datasets/requirement-quality/`, next to the
repositories; `quality_ml.config.DATA_DIR` points there (override it with the `QUALITY_DATA_DIR` environment variable).

## Adding libraries

Add what you need (for example `transformers`, `sentence-transformers`, `torch`) to `dependencies` in
`ml-engine/pyproject.toml`, then reinstall from the repository root:

```powershell
.venv\Scripts\python -m pip install -e "ml-engine[dev]"
```
