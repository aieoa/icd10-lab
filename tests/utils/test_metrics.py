# tests/utils/test_metrics.py

import numpy as np
import pandas as pd
import pytest
from src.utils.metrics import macro_f1_at_k, micro_f1_at_k


@pytest.fixture
def sample_data():
    # Create dummy prediction and ground truth DataFrames
    pred = pd.DataFrame(
        {
            "report_id": [1, 1, 2, 2, 2],
            "sentence_id": [1, 2, 1, 2, 3],
            "icd_code": ["A", "B", "A", "C", "C"],
            "distance": [0.1, 0.2, 0.3, 0.4, 0.5],
            "threshold_passed": [True, True, True, True, True],
        }
    )
    true = pd.DataFrame(
        {
            "report_id": [1, 1, 2, 2, 2],
            "sentence_id": [1, 2, 1, 2, 3],
            "icd_code": ["A", "C", "A", "B", "C"],
        }
    )
    return pred, true


def sample_data_k_2():
    # Create dummy prediction and ground truth DataFrames
    pred = pd.DataFrame(
        {
            "report_id": [1, 1, 1],
            "sentence_id": [1, 1, 1],
            "icd_code": ["A", "B", "C"],
            "distance": [0.1, 0.2, 0.3],
        }
    )
    true = pd.DataFrame({"report_id": [1], "sentence_id": [1], "icd_code": ["B"]})
    return pred, true


def test_macro_f1_at_1(sample_data):
    pred, true = sample_data
    result = macro_f1_at_k(pred, true, top_k=1)
    assert "macro_f1_at_k" in result
    assert "per_code" in result
    assert result["k"] == 1

    assert result["per_code"]["A"]["f1"] == 1.0
    assert result["per_code"]["A"]["precision"] == 1.0
    assert result["per_code"]["A"]["recall"] == 1.0

    assert result["per_code"]["B"]["f1"] == 0
    assert result["per_code"]["B"]["precision"] == 0
    assert result["per_code"]["B"]["recall"] == 0

    assert result["per_code"]["C"]["f1"] == 0.5
    assert result["per_code"]["C"]["precision"] == 0.5
    assert result["per_code"]["C"]["recall"] == 0.5

    assert np.isclose(result["macro_f1_at_k"], 0.5)


def test_micro_f1_at_1(sample_data):
    pred, true = sample_data
    result = micro_f1_at_k(pred, true, top_k=1)
    assert np.isclose(result["precision_at_k"], 3 / 5)
    assert np.isclose(result["recall_at_k"], 3 / 5)
    assert np.isclose(result["micro_f1_at_k"], 3 / 5)


def test_macro_f1_at_k_no_overlap():
    pred = pd.DataFrame(
        {
            "report_id": [1, 2],
            "sentence_id": [1, 1],
            "icd_code": ["A", "B"],
            "distance": [0.1, 0.2],
            "threshold_passed": [True, True],
        }
    )
    true = pd.DataFrame(
        {"report_id": [1, 2], "sentence_id": [1, 1], "icd_code": ["C", "D"]}
    )
    result = macro_f1_at_k(pred, true, top_k=1)
    assert result["macro_f1_at_k"] == 0.0


def test_macro_f1_at_2_no_threshold():
    pred = pd.DataFrame(
        {
            "report_id": [1, 1, 1],
            "sentence_id": [1, 1, 1],
            "icd_code": ["A", "B", "C"],
            "distance": [0.1, 0.2, 0.3],
            "threshold_passed": [True, True, True],
        }
    )
    true = pd.DataFrame({"report_id": [1], "sentence_id": [1], "icd_code": ["B"]})
    result = macro_f1_at_k(pred.copy(), true.copy(), top_k=2)
    print(result)
    assert result["per_code"]["A"]["f1"] == 0.0
    assert result["per_code"]["A"]["precision"] == 0.0
    assert result["per_code"]["A"]["recall"] == 0.0
    assert result["per_code"]["B"]["f1"] == 1.0
    assert result["per_code"]["B"]["precision"] == 1.0
    assert result["per_code"]["B"]["recall"] == 1.0
    assert result["per_code"]["C"]["f1"] == 0.0
    assert result["per_code"]["C"]["precision"] == 0.0
    assert result["per_code"]["C"]["recall"] == 0.0
    assert np.isclose(result["macro_f1_at_k"], 1 / 3)


def test_macro_f1_at_2_threshold():
    pred = pd.DataFrame(
        {
            "report_id": [1, 1, 1],
            "sentence_id": [1, 1, 1],
            "icd_code": ["A", "B", "C"],
            "distance": [0.1, 0.2, 0.3],
            "threshold_passed": [True, True, False],
        }
    )
    true = pd.DataFrame({"report_id": [1], "sentence_id": [1], "icd_code": ["B"]})
    result = macro_f1_at_k(pred.copy(), true.copy(), top_k=2)
    print(result)
    assert result["per_code"]["A"]["f1"] == 0.0
    assert result["per_code"]["A"]["precision"] == 0.0
    assert result["per_code"]["A"]["recall"] == 0.0
    assert result["per_code"]["B"]["f1"] == 1.0
    assert result["per_code"]["B"]["precision"] == 1.0
    assert result["per_code"]["B"]["recall"] == 1.0
    assert np.isclose(result["macro_f1_at_k"], 1 / 2)


def test_macro_f1_at_2_beyond():
    pred = pd.DataFrame(
        {
            "report_id": [1, 1, 1],
            "sentence_id": [1, 1, 1],
            "icd_code": ["A", "B", "C"],
            "distance": [0.1, 0.2, 0.3],
            "threshold_passed": [True, True, True],
        }
    )
    true = pd.DataFrame({"report_id": [1], "sentence_id": [1], "icd_code": ["C"]})
    result = macro_f1_at_k(pred.copy(), true.copy(), top_k=2)
    print(result)
    assert result["per_code"]["A"]["f1"] == 0.0
    assert result["per_code"]["A"]["precision"] == 0.0
    assert result["per_code"]["A"]["recall"] == 0.0
    assert result["per_code"]["B"]["f1"] == 0.0
    assert result["per_code"]["B"]["precision"] == 0.0
    assert result["per_code"]["B"]["recall"] == 0.0
    assert result["per_code"]["C"]["f1"] == 0.0
    assert result["per_code"]["C"]["precision"] == 0.0
    assert result["per_code"]["C"]["recall"] == 0.0
    assert result["macro_f1_at_k"] == 0.0
