import pytest
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from utils.mimic_helper import (
    load_diagnoses,
    load_and_merge_notes,
    filter_notes,
    filter_admission_text,
)


@pytest.fixture
def temp_mimic_dir(tmp_path):
    """Create a temporary mimic directory structure with mock CSV files."""
    mimic_dir = tmp_path / "mimic"
    mimic_dir.mkdir()

    diagnoses_df = pd.DataFrame(
        {
            "hadm_id": [1001, 1001, 1002, 1002, 1003, 1004],
            "subject_id": [101, 101, 102, 102, 102, 103],
            "icd_code": ["E1100", "E1189", "I1000", "I1010", "I1011", "Z8922"],
            "icd_version": [10, 10, 10, 10, 10, 10],
        }
    )
    diagnoses_df.to_csv(
        mimic_dir / "diagnoses_icd.csv.gz", index=False, compression="gzip"
    )

    notes_df = pd.DataFrame(
        {
            "hadm_id": [1001, 1002, 1003, 1004],
            "subject_id": [101, 102, 102, 103],
            "note_id": ["note001", "note002", "note003", "note004"],
            "text": [
                "Notes for stay 1001/101",
                "Notes for stay 1002/102",
                "Notes for stay 1003/102",
                "Notes for stay 1004/103",
            ],
        }
    )
    notes_df.to_csv(mimic_dir / "discharge.csv.gz", index=False, compression="gzip")

    return tmp_path


def test_icd10_code_formatting(temp_mimic_dir):
    """Test that load_diagnoses applies ICD-10 formatting (e.g., E11.00 instead of E1100)."""
    result = load_diagnoses(str(temp_mimic_dir))

    # Test first row (hadm_id 1001)
    expected_codes_row1 = {"E11.00", "E11.89"}
    first_codes = result.iloc[0]["icd_code"]
    assert (
        first_codes == expected_codes_row1
    ), f"Row 1: Expected {expected_codes_row1}, got {first_codes}"

    # Test second row (hadm_id 1002)
    expected_codes_row2 = {"I10.00", "I10.10"}
    second_codes = result.iloc[1]["icd_code"]
    assert (
        second_codes == expected_codes_row2
    ), f"Row 2: Expected {expected_codes_row2}, got {second_codes}"


def test_table_merge(temp_mimic_dir):
    dia = load_diagnoses(str(temp_mimic_dir))
    dia_notes = load_and_merge_notes(str(temp_mimic_dir), dia)

    expected_frame = pd.DataFrame(
        {
            "hadm_id": [1001, 1002, 1003, 1004],
            "subject_id": [101, 102, 102, 103],
            "icd_code": [
                {"E11.00", "E11.89"},
                {"I10.00", "I10.10"},
                {"I10.11"},
                {"Z89.22"},
            ],
            "icd_version": [{10}, {10}, {10}, {10}],
            "note_id": ["note001", "note002", "note003", "note004"],
            "text": [
                "Notes for stay 1001/101",
                "Notes for stay 1002/102",
                "Notes for stay 1003/102",
                "Notes for stay 1004/103",
            ],
        }
    )

    pd.testing.assert_frame_equal(
        dia_notes.reset_index(drop=True),
        expected_frame.reset_index(drop=True),
        check_dtype=False,
    )
