# Computational Sensorimotor Modeling

Compact, reproducible models for trial-by-trial human sensorimotor adaptation.

## Why this repository exists

My experimental work focuses on how human sensorimotor control changes across environments. This repository adds a computational layer to that work by expressing repeated behavioral changes with interpretable model parameters.

The public examples use synthetic data only.

## Models

### 1. One-state adaptation model

    x[t+1] = A * x[t] + B * e[t] + w[t]

where:

- A = retention
- B = error sensitivity / learning rate
- e[t] = target - observed output

The model can be simulated and fitted to trial-by-trial behavior.

### 2. Trial-by-trial correction model

    correction[t+1] = beta * error[t] + intercept

This estimates how strongly the error on one trial predicts the behavioral correction on the next trial.

## Repository structure

- `state_space.py`: simulation and parameter fitting
- `trial_by_trial.py`: error-to-next-trial correction model
- `demo.py`: synthetic adaptation and parameter recovery
- `tests/test_models.py`: unit tests

## Setup

    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

## Run

    python demo.py
    python -m unittest discover -s tests -v

## Current scope

This is intentionally a simple starting point. Natural extensions include:

- participant-level parameter estimation
- Land vs Water comparisons
- experienced vs non-experienced group comparisons
- hierarchical Bayesian models
- Kalman filtering / latent-state estimation
- multi-rate adaptation models
- linking latent behavioral states to EEG/SEP measures

The goal is not to make a simple model look more sophisticated than it is. The goal is to make assumptions explicit and build toward computational sensorimotor neuroscience.
