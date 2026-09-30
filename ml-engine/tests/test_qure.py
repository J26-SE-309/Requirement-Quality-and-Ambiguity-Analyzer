import pandas as pd

from quality_ml.qure import clean_qure, split_qure


def test_clean_qure_removes_conflicting_requirements():
    df = pd.DataFrame(
        {
            "id": ["1", "2", "3"],
            "requirement": [
                "The system shall display the status.",
                "The system shall display the status.",
                "The system shall save the result.",
            ],
            "defect": ["ok", "defect", "ok"],
            "weak_word": ["status", "status", "result"],
        }
    )

    clean = clean_qure(df)

    assert len(clean) == 1
    assert clean.iloc[0]["requirement"] == "The system shall save the result."


def test_clean_qure_keeps_consistent_duplicates_once():
    df = pd.DataFrame(
        {
            "id": ["1", "2", "3"],
            "requirement": [
                "The system shall display the status.",
                "The system shall display the status.",
                "The system shall save the result.",
            ],
            "defect": ["ok", "ok", "defect"],
            "weak_word": ["status", "status", "result"],
        }
    )

    clean = clean_qure(df)

    assert len(clean) == 2
    assert clean["requirement"].is_unique


def test_split_qure_has_expected_sizes():
    df = pd.DataFrame(
        {
            "requirement": [f"Requirement {i}" for i in range(100)],
            "defect": ["ok"] * 70 + ["defect"] * 30,
        }
    )

    train, validation, test = split_qure(df)

    assert len(train) == 80
    assert len(validation) == 10
    assert len(test) == 10


def test_split_qure_has_no_requirement_overlap():
    df = pd.DataFrame(
        {
            "requirement": [f"Requirement {i}" for i in range(100)],
            "defect": ["ok"] * 70 + ["defect"] * 30,
        }
    )

    train, validation, test = split_qure(df)

    train_requirements = set(train["requirement"])
    validation_requirements = set(validation["requirement"])
    test_requirements = set(test["requirement"])

    assert not train_requirements.intersection(validation_requirements)
    assert not train_requirements.intersection(test_requirements)
    assert not validation_requirements.intersection(test_requirements)


def test_split_qure_is_reproducible():
    df = pd.DataFrame(
        {
            "requirement": [f"Requirement {i}" for i in range(100)],
            "defect": ["ok"] * 70 + ["defect"] * 30,
        }
    )

    train1, validation1, test1 = split_qure(df, random_state=42)
    train2, validation2, test2 = split_qure(df, random_state=42)

    assert train1["requirement"].tolist() == train2["requirement"].tolist()
    assert validation1["requirement"].tolist(
    ) == validation2["requirement"].tolist()
    assert test1["requirement"].tolist() == test2["requirement"].tolist()
