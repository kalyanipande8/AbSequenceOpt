import json
import os

from src.simulate_cdr_variants import simulate, N_FEATURES
from src.bayesian_optimization import run_bayesian_optimization


def main():
    df = simulate()
    feature_cols = [f"feature_{i:02d}" for i in range(N_FEATURES)]
    result = run_bayesian_optimization(df, feature_cols)

    summary = {"n_variants": len(df), "n_features": N_FEATURES, "optimization": result}

    os.makedirs("results", exist_ok=True)
    with open("results/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
