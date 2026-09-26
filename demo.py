from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from state_space import fit_state_space, simulate_state_space
from trial_by_trial import compare_conditions


def build_synthetic_conditions(random_state=42):
    targets = np.concatenate([np.zeros(10), np.ones(60), np.zeros(30)])

    land = simulate_state_space(
        targets,
        retention=0.93,
        error_sensitivity=0.18,
        process_noise_sd=0.01,
        observation_noise_sd=0.03,
        random_state=random_state,
    )
    water = simulate_state_space(
        targets,
        retention=0.88,
        error_sensitivity=0.28,
        process_noise_sd=0.01,
        observation_noise_sd=0.03,
        random_state=random_state + 1,
    )

    return {
        "Land": {"targets": targets, **land},
        "Water": {"targets": targets, **water},
    }


def main():
    conditions = build_synthetic_conditions()

    print("=== State-space fits ===")
    for label, data in conditions.items():
        fitted = fit_state_space(data["targets"], data["observed"])
        print(
            f"{label}: A={fitted['retention']:.3f}, "
            f"B={fitted['error_sensitivity']:.3f}, "
            f"MSE={fitted['mse']:.6f}"
        )

    tbt = compare_conditions(
        {
            label: {
                "targets": data["targets"],
                "observed": data["observed"],
            }
            for label, data in conditions.items()
        }
    )

    print("\n=== Trial-by-trial fits ===")
    for label, result in tbt.items():
        print(
            f"{label}: gain={result['gain']:.3f}, "
            f"R^2={result['r2']:.3f}, "
            f"pairs={result['n_pairs']}"
        )

    output = Path("output")
    output.mkdir(exist_ok=True)

    plt.figure(figsize=(9, 5))
    for label, data in conditions.items():
        plt.plot(data["observed"], label=label, alpha=0.85)
    plt.plot(conditions["Land"]["targets"], linestyle="--", label="Target")
    plt.xlabel("Trial")
    plt.ylabel("Behavioral output")
    plt.title("Synthetic sensorimotor adaptation")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output / "synthetic_adaptation.png", dpi=200)


if __name__ == "__main__":
    main()
