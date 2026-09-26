from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .analysis import fit_participants
from .io import load_trial_csv
from .state_space import simulate_state_space


def synthetic_frame():
    rows = []
    targets = np.concatenate([np.zeros(10), np.ones(60), np.zeros(30)])
    configs = {
        "Land": (0.93, 0.18, 42),
        "Water": (0.88, 0.28, 43),
    }

    for subject_id, (condition, (a, b, seed)) in enumerate(
        configs.items(),
        start=1,
    ):
        simulation = simulate_state_space(
            targets,
            retention=a,
            error_sensitivity=b,
            process_noise_sd=0.01,
            observation_noise_sd=0.03,
            random_state=seed,
        )

        for trial, (target, observed) in enumerate(
            zip(targets, simulation["observed"])
        ):
            rows.append(
                {
                    "subject_id": f"S{subject_id:02d}",
                    "condition": condition,
                    "trial": trial,
                    "target": target,
                    "observed": observed,
                }
            )

    return pd.DataFrame(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Fit interpretable sensorimotor models to trial-level behavior."
    )
    parser.add_argument(
        "--input",
        help="CSV with subject_id, trial, target, observed",
    )
    parser.add_argument(
        "--output",
        default="output/participant_parameters.csv",
    )
    args = parser.parse_args(argv)

    frame = load_trial_csv(args.input) if args.input else synthetic_frame()
    result = fit_participants(frame)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)

    print(result.to_string(index=False))
    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()
