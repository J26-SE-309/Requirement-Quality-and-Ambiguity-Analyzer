from pathlib import Path

import pytest
from quality_ml.damir import clean_damir, load_damir, split_damir, validate_damir

DATA_PATH = Path(r"C:\\Users\\ASUS\\Desktop\\Synapse\\Datasets\\requirement-quality\\DAMIR.xlsx")


def test_load_damir_has_expected_structure():
    df = load_damir(DATA_PATH)

    assert df.shape == (5505, 8)
    assert {
        "Id",
        "Context",
        "Pronoun",
        "Position",
        "Candidate Antecedent",
        "ResolvedAs",
        "AckUnack",
    }.issubset(df.columns)


def test_clean_damir_keeps_candidate_level_rows():
    df = load_damir(DATA_PATH)
    clean = clean_damir(df)

    assert clean.shape == (5505, 7)
    assert "Unnamed: 0" not in clean.columns
    assert clean["Id"].nunique() == 587


def test_validate_damir_rejects_missing_columns():
    df = load_damir(DATA_PATH).drop(columns=["Pronoun"])

    with pytest.raises(ValueError, match="missing columns"):
        validate_damir(df)


def test_validate_damir_rejects_inconsistent_occurrence_fields():
    df = load_damir(DATA_PATH).copy()
    first_id = df["Id"].iloc[0]
    matching_rows = df["Id"] == first_id
    first_matching_index = df.index[matching_rows][0]
    df.loc[first_matching_index, "Pronoun"] = "different"

    with pytest.raises(ValueError, match="occurrence fields are inconsistent"):
        validate_damir(df)


def test_split_damir_keeps_occurrences_together():
    df = load_damir(DATA_PATH)
    train, validation, test = split_damir(df)

    train_ids = set(train["Id"])
    validation_ids = set(validation["Id"])
    test_ids = set(test["Id"])

    assert len(train_ids) == 469
    assert len(validation_ids) == 59
    assert len(test_ids) == 59

    assert train_ids.isdisjoint(validation_ids)
    assert train_ids.isdisjoint(test_ids)
    assert validation_ids.isdisjoint(test_ids)

    assert len(train) + len(validation) + len(test) == 5505
    assert len(train_ids | validation_ids | test_ids) == 587


def test_split_damir_is_deterministic():
    df = load_damir(DATA_PATH)

    first = split_damir(df)
    second = split_damir(df)

    for first_part, second_part in zip(first, second):
        assert first_part["Id"].tolist() == second_part["Id"].tolist()


