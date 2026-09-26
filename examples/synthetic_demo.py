from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from sensorimotor.state_space import fit_state_space, simulate_state_space
from sensorimotor.trial_by_trial import compare_conditions


def main():
    targets = np.concatenate([np.zeros(10), np.ones(60), np.zeros(30)])
    configs = {
        "Land": (0.93, 0.18, 42),
        "Water": (0.88, 0.28, 43),
    }
    conditions = {}

    for label, (a, b, seed) in configs.items():
        simulation = simulate_state_space(
            targets,
            retention=a,
            error_sensitivity=b,
            process_noise_sd=0.01,
            observation_noise_sd=0.03,
            random_state=seed,
        )
        conditions[label] = {"targets": targets, **simulation}
        fitted = fit_state_space(targets, simulation["observed"])
        print(
            f"{label}: A={fitted['retention']:.3f}, "
            f"B={fitted['error_sensitivity']:.3f}, "
            f"MSE={fitted['mse']:.6f}"
        )

    print(
        compare_conditions(
            {
                label: {
                    "targets": data["targets"],
                    "observed": data["observed"],
                }
                for label, data in conditions.items()
            }
        )
    )

    output = Path("output")
    output.mkdir(exist_ok=True)

    plt.figure(figsize=(9, 5))
    for label, data in conditions.items():
        plt.plot(data["observed"], label=label)
    plt.plot(targets, linestyle="--", label="Target")
    plt.xlabel("Trial")
    plt.ylabel("Behavioral output")
    plt.title("Synthetic sensorimotor adaptation")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output / "synthetic_adaptation.png", dpi=200)

    print(f"Saved: {output / 'synthetic_adaptation.png'}")


if __name__ == "__main__":
    main()
