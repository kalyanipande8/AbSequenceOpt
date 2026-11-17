"""Simulate 1,220 antibody CDR sequence variants, each with 14 real
sequence-derived scoring features (hydrophobicity, charge, length,
etc. style descriptors) and a hidden true binding affinity -- the
real search space a Bayesian optimization loop explores without
seeing the affinity function directly."""
from __future__ import annotations

import numpy as np
import pandas as pd

N_VARIANTS = 1220
N_FEATURES = 14
SEED = 1171


def simulate(n: int = N_VARIANTS, n_features: int = N_FEATURES, seed: int = SEED):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n, n_features))

    true_weights = rng.normal(0, 1, n_features)
    true_weights[:4] *= 2  # 4 of the 14 features are the real strong affinity drivers
    nonlinear_term = -0.3 * (X[:, 0] - 1.0) ** 2  # a real optimum, not monotonic

    true_affinity = X @ true_weights + nonlinear_term + rng.normal(0, 1.0, n)

    columns = [f"feature_{i:02d}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=columns)
    df.insert(0, "variant_id", [f"CDR-{i:04d}" for i in range(n)])
    df["true_affinity"] = true_affinity
    return df


if __name__ == "__main__":
    df = simulate()
    print(df.shape, df["true_affinity"].describe())
