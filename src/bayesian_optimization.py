"""Real Bayesian optimization loop: a Gaussian Process surrogate
model over the 14 scoring features, refined each iteration with an
Upper Confidence Bound acquisition function selecting the next
CDR variant to 'evaluate' (reveal its true affinity) -- the real
iterative refinement loop, not a single-shot ranking."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern


def run_bayesian_optimization(df: pd.DataFrame, feature_cols: list, n_initial: int = 20,
                               n_iterations: int = 60, seed: int = 1171) -> dict:
    rng = np.random.default_rng(seed)
    X_all = df[feature_cols].values
    y_all = df["true_affinity"].values
    n = len(df)

    evaluated_idx = list(rng.choice(n, n_initial, replace=False))
    unevaluated_idx = [i for i in range(n) if i not in evaluated_idx]

    best_so_far = [max(y_all[evaluated_idx])]

    for iteration in range(n_iterations):
        if not unevaluated_idx:
            break
        X_train = X_all[evaluated_idx]
        y_train = y_all[evaluated_idx]

        gp = GaussianProcessRegressor(kernel=Matern(nu=2.5), normalize_y=True, random_state=seed)
        gp.fit(X_train, y_train)

        X_candidates = X_all[unevaluated_idx]
        mean, std = gp.predict(X_candidates, return_std=True)
        ucb = mean + 1.5 * std  # Upper Confidence Bound acquisition

        chosen_local_idx = int(np.argmax(ucb))
        chosen_global_idx = unevaluated_idx.pop(chosen_local_idx)
        evaluated_idx.append(chosen_global_idx)

        best_so_far.append(max(best_so_far[-1], y_all[chosen_global_idx]))

    initial_best = best_so_far[0]
    final_best = best_so_far[-1]
    improvement_pct = round(100 * (final_best - initial_best) / abs(initial_best), 2)

    return {"n_variants_total": n, "n_initial_random": n_initial, "n_bo_iterations": len(best_so_far) - 1,
            "initial_best_affinity": round(float(initial_best), 4),
            "final_best_affinity": round(float(final_best), 4),
            "improvement_pct": improvement_pct, "best_so_far_curve": [round(float(v), 4) for v in best_so_far]}
