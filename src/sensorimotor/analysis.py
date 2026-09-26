from __future__ import annotations

import pandas as pd

from .state_space import fit_state_space
from .trial_by_trial import fit_trial_by_trial_update


def fit_participants(frame):
    """Fit state-space and trial-by-trial models separately for each participant."""
    rows = []
    grouping = ["subject_id"]
    for optional in ("condition", "group"):
        if optional in frame.columns:
            grouping.append(optional)

    for keys, part in frame.groupby(grouping, sort=True, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        metadata = dict(zip(grouping, keys))
        target = part["target"].to_numpy(float)
        observed = part["observed"].to_numpy(float)

        state_space = fit_state_space(target, observed)
        trial_update = fit_trial_by_trial_update(target, observed)

        rows.append(
            {
                **metadata,
                "retention": state_space["retention"],
                "error_sensitivity": state_space["error_sensitivity"],
                "state_space_mse": state_space["mse"],
                "trial_update_gain": trial_update["gain"],
                "trial_update_r2": trial_update["r2"],
                "n_trials": len(part),
            }
        )

    return pd.DataFrame(rows)
