# Math Notes: From Trial-by-Trial Learning to Bayesian State Estimation

This note is a compact map of the mathematical language behind the code in this repository.

## 1. Observable behavior vs latent state

In a motor-learning experiment, the quantity we measure on each trial is an observation:

    y[t]

Examples include produced velocity, directional error, or endpoint position.

The internal state that generated that behavior is not observed directly:

    x[t]

A state-space model therefore separates the hidden state from the noisy measurement.

## 2. State transition model

A simple error-based adaptation model is

    x[t+1] = A * x[t] + B * e[t] + w[t]

where

- A: retention of the previous internal state
- B: sensitivity to trial error
- e[t]: prediction or performance error
- w[t]: process noise

Interpretation:

- A close to 1 means the learned state is retained strongly.
- Larger B means behavior is updated more strongly from error.

These parameters are model-dependent summaries, not direct measurements of a biological mechanism.

## 3. Observation model

The behavioral measurement is represented as

    y[t] = C * x[t] + v[t]

where

- C maps latent state to the observable variable
- v[t] is observation noise

For the simplest model, C = 1.

This distinction matters because noisy behavior does not imply that the latent state itself changed by the same amount.

## 4. Likelihood

Given parameters theta = {A, B, sigma}, the model predicts behavior.

If observation error is Gaussian,

    y[t] ~ Normal(y_hat[t], sigma^2)

the likelihood is

    p(y | theta)

Maximum-likelihood or least-squares fitting asks:

    Which parameters make the observed data most probable?

## 5. Bayesian parameter estimation

Bayes' rule is

    p(theta | y) ∝ p(y | theta) * p(theta)

where

- p(theta | y): posterior
- p(y | theta): likelihood
- p(theta): prior

Instead of returning only one best-fitting A and B, Bayesian estimation represents uncertainty over plausible parameter values.

The `bayesian_fit.py` demo uses a simple grid posterior so the calculation remains transparent.

## 6. Kalman filtering

When the state itself must be estimated sequentially, a Kalman filter alternates between:

### Prediction

    x_pred = A * x_prev + B * u[t]

    P_pred = A^2 * P_prev + Q

### Measurement update

    K = P_pred * C / (C^2 * P_pred + R)

    x_updated = x_pred + K * (y[t] - C * x_pred)

    P_updated = (1 - K * C) * P_pred

where

- P: uncertainty about the state
- Q: process-noise variance
- R: observation-noise variance
- K: Kalman gain

The Kalman gain determines how strongly the estimate should move toward the new observation.

## 7. Connection to sensorimotor research

A useful conceptual chain is:

    environment
        -> sensory evidence
        -> latent state estimate
        -> motor command
        -> observed behavior
        -> error
        -> state update

This is why state-space and Bayesian models are useful for studying adaptation to changing environments.

## 8. What not to overclaim

A fitted parameter is not automatically a neural mechanism.

For example:

- lower A does not directly prove poorer motor memory
- larger B does not directly prove greater sensory weighting
- a Kalman gain is not a measured brain signal

Those interpretations require converging experimental evidence.

The value of the model is that it makes the assumed update rule explicit and testable.
