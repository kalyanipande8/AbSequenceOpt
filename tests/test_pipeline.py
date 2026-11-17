import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.simulate_cdr_variants import simulate, N_FEATURES
from src.bayesian_optimization import run_bayesian_optimization


def test_at_least_1200_variants():
    df = simulate()
    assert len(df) >= 1200


def test_14_features_defined():
    assert N_FEATURES == 14


def test_bo_improves_over_initial_random_sample():
    df = simulate()
    feature_cols = [f"feature_{i:02d}" for i in range(N_FEATURES)]
    result = run_bayesian_optimization(df, feature_cols, n_initial=20, n_iterations=40)
    assert result["final_best_affinity"] >= result["initial_best_affinity"]
    assert result["improvement_pct"] > 0


def test_best_so_far_curve_is_monotonically_nondecreasing():
    df = simulate()
    feature_cols = [f"feature_{i:02d}" for i in range(N_FEATURES)]
    result = run_bayesian_optimization(df, feature_cols, n_initial=20, n_iterations=40)
    curve = result["best_so_far_curve"]
    assert all(curve[i] <= curve[i + 1] for i in range(len(curve) - 1))


def test_bo_evaluates_expected_number_of_variants():
    df = simulate()
    feature_cols = [f"feature_{i:02d}" for i in range(N_FEATURES)]
    result = run_bayesian_optimization(df, feature_cols, n_initial=20, n_iterations=40)
    assert result["n_bo_iterations"] == 40
