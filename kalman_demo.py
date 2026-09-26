import numpy as np

from kalman_filter import kalman_filter_1d


def main():
    rng = np.random.default_rng(42)

    n_trials = 120
    true_state = np.zeros(n_trials)
    for t in range(1, n_trials):
        true_state[t] = 0.97 * true_state[t - 1] + rng.normal(0.0, 0.04)

    observations = true_state + rng.normal(0.0, 0.15, size=n_trials)

    result = kalman_filter_1d(
        observations,
        a=0.97,
        process_var=0.04 ** 2,
        observation_var=0.15 ** 2,
    )

    raw_rmse = np.sqrt(np.mean((observations - true_state) ** 2))
    filtered_rmse = np.sqrt(np.mean((result["state"] - true_state) ** 2))

    print(f"Raw observation RMSE: {raw_rmse:.4f}")
    print(f"Kalman estimate RMSE: {filtered_rmse:.4f}")
    print(f"Mean Kalman gain: {result['gain'].mean():.4f}")


if __name__ == "__main__":
    main()
