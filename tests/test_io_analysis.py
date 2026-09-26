import pandas as pd
import pytest

from sensorimotor.analysis import fit_participants
from sensorimotor.cli import synthetic_frame
from sensorimotor.io import load_trial_csv


def test_fit_participants_returns_one_row_per_subject_condition():
    frame = synthetic_frame()
    result = fit_participants(frame)

    assert len(result) == 2
    assert {
        "retention",
        "error_sensitivity",
        "trial_update_gain",
    }.issubset(result.columns)


def test_load_trial_csv_validates_schema(tmp_path):
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"subject_id": ["S01"]}).to_csv(bad, index=False)

    with pytest.raises(ValueError):
        load_trial_csv(bad)
