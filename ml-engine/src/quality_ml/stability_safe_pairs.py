"""Generate separate meaning-preserving mutation candidates for Task 3."""

from pathlib import Path

import pandas as pd

from quality_ml.stability_pairs import (
    EXCLUDED_REQUIREMENT_IDS,
    OUTPUT_COLUMNS,
    SPLIT_NAMES,
    validate_mutation,
    validate_requirements,
    verify_split_isolation,
)
from quality_ml.stability_safe_mutations import mutate_safe_paraphrase


def build_safe_pairs(input_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Build candidate paraphrase pairs using existing dataset splits."""

    input_path = Path(input_dir)
    splits = {}

    for split_name in SPLIT_NAMES:
        path = input_path / f"task3_{split_name}_requirements.csv"
        df = pd.read_csv(path)
        validate_requirements(df, split_name)
        splits[split_name] = df

    verify_split_isolation(splits)

    result = {}

    for split_name, df in splits.items():
        rows = []

        for requirement in df.itertuples(index=False):
            requirement_id = requirement.requirement_id

            if requirement_id in EXCLUDED_REQUIREMENT_IDS:
                continue

            original_text = requirement.requirement_text
            mutation = mutate_safe_paraphrase(original_text)

            if mutation is None:
                continue

            validate_mutation(mutation, original_text)

            rows.append(
                {
                    "pair_id": f"SAFE-{split_name.upper()}-{len(rows) + 1:05d}",
                    "requirement_id": requirement_id,
                    "split": split_name,
                    "original_text": original_text,
                    "mutated_text": mutation["mutated_text"],
                    "mutation_type": mutation["mutation_type"],
                    "original_span": mutation["original_span"],
                    "replacement_span": mutation["replacement_span"],
                }
            )

        result[split_name] = pd.DataFrame(
            rows,
            columns=OUTPUT_COLUMNS,
        )

    return result


def main() -> None:
    """Save safe mutation candidates without overwriting existing files."""

    root = (
        Path.home()
        / "Desktop"
        / "Synapse"
        / "Datasets"
        / "requirement-quality"
        / "task3"
    )

    output_dir = root / "safe_mutation_pairs"
    pairs = build_safe_pairs(root / "processed")

    expected_counts = {
        "train": 190,
        "val": 23,
        "test": 26,
    }

    counts = {
        name: len(df)
        for name, df in pairs.items()
    }

    if counts != expected_counts:
        raise ValueError(
            f"Unexpected candidate counts: {counts}"
        )

    paths = {
        name: output_dir / f"task3_{name}_safe_pairs.csv"
        for name in SPLIT_NAMES
    }

    existing = [
        str(path)
        for path in paths.values()
        if path.exists()
    ]

    if existing:
        raise FileExistsError(
            "Safe mutation files already exist: "
            + ", ".join(existing)
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    for name in SPLIT_NAMES:
        pairs[name].to_csv(paths[name], index=False)
        print(f"{name}: {len(pairs[name])} safe candidates")

    print(f"TOTAL: {sum(counts.values())}")
    print(f"OUTPUT: {output_dir}")


if __name__ == "__main__":
    main()
