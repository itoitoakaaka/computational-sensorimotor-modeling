from __future__ import annotations

from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {"subject_id", "trial", "target", "observed"}


def load_trial_csv(path):
    """Load de-identified trial-level data in the public input schema."""
    path = Path(path)
    frame = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    frame = frame.copy()
    frame["trial"] = pd.to_numeric(frame["trial"], errors="raise")
    frame["target"] = pd.to_numeric(frame["target"], errors="raise")
    frame["observed"] = pd.to_numeric(frame["observed"], errors="raise")
    return frame.sort_values(["subject_id", "trial"]).reset_index(drop=True)
