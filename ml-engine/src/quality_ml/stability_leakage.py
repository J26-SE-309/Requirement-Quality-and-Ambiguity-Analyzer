
"""Detect near-duplicate requirements in Task 3 datasets."""

from itertools import combinations

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


CROSS_SPLIT_COLUMNS = [
    "first_split",
    "first_id",
    "first_text",
    "second_split",
    "second_id",
    "second_text",
    "similarity",
]

ALL_PAIRS_COLUMNS = [
    "first_id",
    "first_text",
    "second_id",
    "second_text",
    "similarity",
]


def _validate_parameters(
    threshold: float,
    batch_size: int,
) -> None:
    """Validate similarity detection parameters."""

    if not 0 < threshold <= 1:
        raise ValueError(
            "Similarity threshold must be between 0 and 1."
        )

    if batch_size < 1:
        raise ValueError(
            "Batch size must be positive."
        )


def _validate_dataset(
    df: pd.DataFrame,
    dataset_name: str,
) -> None:
    """Check that a dataset contains required columns."""

    required_columns = {
        "requirement_id",
        "requirement_text",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"{dataset_name} is missing columns: "
            f"{sorted(missing_columns)}"
        )

    if df["requirement_id"].isna().any():
        raise ValueError(
            f"{dataset_name} contains missing requirement IDs."
        )

    if df["requirement_text"].isna().any():
        raise ValueError(
            f"{dataset_name} contains missing requirement texts."
        )


def _build_tfidf_vectors(
    texts: pd.Series,
):
    """Convert requirement texts into TF-IDF vectors."""

    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True,
    )

    return vectorizer.fit_transform(
        texts.astype(str)
    )


def find_all_near_duplicates(
    df: pd.DataFrame,
    threshold: float = 0.90,
    batch_size: int = 128,
) -> pd.DataFrame:
    """Find near-duplicate pairs across the entire dataset.

    This function should run before splitting the dataset.

    It uses TF-IDF cosine similarity to identify highly
    similar requirements.

    Similarity calculations are performed in batches,
    avoiding a full dense all-pairs similarity matrix.

    Only unique requirement pairs are returned.
    """

    _validate_parameters(
        threshold,
        batch_size,
    )

    _validate_dataset(
        df,
        "Task 3 dataset",
    )

    dataset = df.reset_index(drop=True).copy()

    if dataset["requirement_id"].duplicated().any():
        raise ValueError(
            "Duplicate requirement IDs detected."
        )

    if len(dataset) < 2:
        return pd.DataFrame(
            columns=ALL_PAIRS_COLUMNS
        )

    vectors = _build_tfidf_vectors(
        dataset["requirement_text"]
    )

    total_requirements = len(dataset)

    matches = []

    for start in range(
        0,
        total_requirements,
        batch_size,
    ):
        end = min(
            start + batch_size,
            total_requirements,
        )

        similarities = cosine_similarity(
            vectors[start:end],
            vectors,
        )

        # Compare each requirement only with later rows.
        # This avoids self-matches and duplicate pairs.
        for local_row in range(end - start):
            first_index = start + local_row

            candidate_indices = (
                similarities[local_row] >= threshold
            ).nonzero()[0]

            for second_index in candidate_indices:
                if second_index <= first_index:
                    continue

                first_record = dataset.iloc[first_index]
                second_record = dataset.iloc[second_index]

                matches.append(
                    {
                        "first_id": first_record[
                            "requirement_id"
                        ],
                        "first_text": first_record[
                            "requirement_text"
                        ],
                        "second_id": second_record[
                            "requirement_id"
                        ],
                        "second_text": second_record[
                            "requirement_text"
                        ],
                        "similarity": round(
                            float(
                                similarities[
                                    local_row,
                                    second_index,
                                ]
                            ),
                            4,
                        ),
                    }
                )

    result = pd.DataFrame(
        matches,
        columns=ALL_PAIRS_COLUMNS,
    )

    if result.empty:
        return result

    return result.sort_values(
        "similarity",
        ascending=False,
    ).reset_index(drop=True)


def find_cross_split_near_duplicates(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    threshold: float = 0.90,
    batch_size: int = 128,
) -> pd.DataFrame:
    """Find highly similar requirements across different splits.

    Uses TF-IDF cosine similarity without creating a full
    dense all-pairs similarity matrix.
    """

    _validate_parameters(
        threshold,
        batch_size,
    )

    splits = {
        "train": train.reset_index(drop=True),
        "validation": validation.reset_index(drop=True),
        "test": test.reset_index(drop=True),
    }

    for name, df in splits.items():
        _validate_dataset(
            df,
            name,
        )

    combined = pd.concat(
        [
            df.assign(split=name)
            for name, df in splits.items()
        ],
        ignore_index=True,
    )

    if combined.empty:
        return pd.DataFrame(
            columns=CROSS_SPLIT_COLUMNS
        )

    if combined["requirement_id"].duplicated().any():
        raise ValueError(
            "Requirement IDs overlap across dataset splits."
        )

    vectors = _build_tfidf_vectors(
        combined["requirement_text"]
    )

    split_indices = {
        name: combined.index[
            combined["split"] == name
        ].to_numpy()
        for name in splits
    }

    matches = []

    for first_name, second_name in combinations(
        splits.keys(),
        2,
    ):
        first_indices = split_indices[first_name]
        second_indices = split_indices[second_name]

        if (
            len(first_indices) == 0
            or len(second_indices) == 0
        ):
            continue

        for start in range(
            0,
            len(first_indices),
            batch_size,
        ):
            batch_indices = first_indices[
                start:start + batch_size
            ]

            similarities = cosine_similarity(
                vectors[batch_indices],
                vectors[second_indices],
            )

            rows, cols = (
                similarities >= threshold
            ).nonzero()

            for row, col in zip(rows, cols):
                first_index = batch_indices[row]
                second_index = second_indices[col]

                first_record = combined.iloc[first_index]
                second_record = combined.iloc[second_index]

                matches.append(
                    {
                        "first_split": first_name,
                        "first_id": first_record[
                            "requirement_id"
                        ],
                        "first_text": first_record[
                            "requirement_text"
                        ],
                        "second_split": second_name,
                        "second_id": second_record[
                            "requirement_id"
                        ],
                        "second_text": second_record[
                            "requirement_text"
                        ],
                        "similarity": round(
                            float(
                                similarities[row, col]
                            ),
                            4,
                        ),
                    }
                )

    result = pd.DataFrame(
        matches,
        columns=CROSS_SPLIT_COLUMNS,
    )

    if result.empty:
        return result

    return result.sort_values(
        "similarity",
        ascending=False,
    ).reset_index(drop=True)
