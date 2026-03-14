import numpy as np
import pandas as pd
from scipy.optimize import minimize
from src.utils.paths import PROCESSED_DIR
from src.data.load_data import ASSET_COLUMNS
from src.data.build_regime_features import GROWTH_ASSETS, DEFENSIVE_ASSETS


def load_regime_inputs(
    summary_filename: str = "hmm_regime_asset_summary.csv"
) -> tuple[pd.DataFrame, dict[int, pd.DataFrame]]:
    summary = pd.read_csv(PROCESSED_DIR / summary_filename)

    cov_dict = {}
    for regime in [0, 1, 2]:
        cov = pd.read_csv(PROCESSED_DIR / f"hmm_regime_cov_regime{regime}.csv", index_col=0)
        cov = cov.loc[ASSET_COLUMNS, ASSET_COLUMNS]
        cov_dict[regime] = cov

    return summary, cov_dict


def build_regime_expected_return_vectors(
        summary_df: pd.DataFrame,
        return_col: str = "mean_monthly_return"
) -> dict[int, pd.Series]:
    mu_dict = {}

    for regime in sorted(summary_df["regime"].unique()):
        sub = summary_df[summary_df["regime"] == regime].copy()
        mu = sub.set_index("asset_code")[return_col].reindex(ASSET_COLUMNS)
        mu_dict[regime] = mu

    return mu_dict


def portfolio_stats(w: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> tuple[float, float]:
    port_return = float(np.dot(w, mu))
    port_var = float(w.T @ sigma @ w)
    port_vol = float(np.sqrt(max(port_var, 0.0)))
    return port_return, port_vol


def objective_negative_quadratic_utility(
    w: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    risk_aversion: float
) -> float:
    """
    Minimise negative quadratic utility:
    -(w^T mu - 0.5 * risk_aversion * w^T sigma w)
    """
    ret = float(np.dot(w, mu))
    var = float(w.T @ sigma @ w)
    utility = ret - 0.5 * risk_aversion * var
    return -utility


def make_constraints(
    min_growth: float | None = None,
    max_growth: float | None = None,
):
    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    ]

    if min_growth is not None:
        growth_idx = [ASSET_COLUMNS.index(a) for a in GROWTH_ASSETS]
        constraints.append({
            "type": "ineq",
            "fun": lambda w, idx=growth_idx, mg=min_growth: np.sum(w[idx]) - mg
        })

    if max_growth is not None:
        growth_idx = [ASSET_COLUMNS.index(a) for a in GROWTH_ASSETS]
        constraints.append({
            "type": "ineq",
            "fun": lambda w, idx=growth_idx, Mg=max_growth: Mg - np.sum(w[idx])
        })

    return constraints


def optimise_regime_portfolio(
    mu: pd.Series,
    sigma: pd.DataFrame,
    risk_aversion: float = 3.0,
    bounds_map: dict[str, tuple[float, float]] | None = None,
    min_growth: float | None = None,
    max_growth: float | None = None,
) -> dict:
    mu_vec = mu.values.astype(float)
    sigma_mat = sigma.values.astype(float)

    if bounds_map is None:
        bounds_map = {asset: (0.0, 1.0) for asset in ASSET_COLUMNS}

    bounds = [bounds_map[a] for a in ASSET_COLUMNS]
    x0 = np.array([1.0 / len(ASSET_COLUMNS)] * len(ASSET_COLUMNS))

    constraints = make_constraints(min_growth=min_growth, max_growth=max_growth)

    result = minimize(
        objective_negative_quadratic_utility,
        x0=x0,
        args=(mu_vec, sigma_mat, risk_aversion),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 200, "disp": False},
    )

    if not result.success:
        raise RuntimeError("Optimisation failed:", result.message)

    w_opt = result.x
    port_return, port_vol = portfolio_stats(w_opt, mu_vec, sigma_mat)
    utility = port_return - 0.5 * risk_aversion * (port_vol**2)

    output = {
        "weights": pd.Series(w_opt, index=ASSET_COLUMNS),
        "expected_return_monthly": port_return,
        "expected_vol_monthly": port_vol,
        "quadratic_utility": utility,
        "growth_weight": float(np.sum([w_opt[ASSET_COLUMNS.index(a)] for a in GROWTH_ASSETS])),
        "defensive_weight": float(np.sum([w_opt[ASSET_COLUMNS.index(a)] for a in DEFENSIVE_ASSETS])),
        "success": result.success,
        "message": result.message,
    }

    return output


def build_default_bounds() -> dict[str, tuple[float, float]]:
    """
    First-pass bounds. Adjust later if you want more realistic SAA bands.
    """
    return {
        "AEQ": (0.00, 0.35),
        "ILE_H": (0.00, 0.30),
        "ILE_UH": (0.00, 0.30),
        "ALP": (0.00, 0.20),
        "ILP_H": (0.00, 0.20),
        "ILI_H": (0.00, 0.20),
        "AFI": (0.00, 0.40),
        "IFI_H": (0.00, 0.30),
        "CASH": (0.00, 0.25),
    }


def run_regime_optimisation(
    summary_df: pd.DataFrame,
    cov_dict: dict[int, pd.DataFrame],
    risk_aversion: float = 3.0,
    min_growth: float | None = 0.50,
    max_growth: float | None = 0.90,
) -> tuple[pd.DataFrame, dict[int, dict]]:
    mu_dict = build_regime_expected_return_vectors(summary_df)
    bounds_map = build_default_bounds()

    results = {}
    rows = []

    regime_name_map = (
        summary_df[["regime", "regime_name"]]
        .drop_duplicates()
        .set_index("regime")["regime_name"]
        .to_dict()
    )

    for regime in sorted(mu_dict.keys()):
        mu = mu_dict[regime]
        sigma = cov_dict[regime]

        out = optimise_regime_portfolio(
            mu=mu,
            sigma=sigma,
            risk_aversion=risk_aversion,
            bounds_map=bounds_map,
            min_growth=min_growth,
            max_growth=max_growth,
        )

        results[regime] = out

        row = {
            "regime": regime,
            "regime_name": regime_name_map.get(regime, f"Regime {regime}"),
            "expected_return_monthly": out["expected_return_monthly"],
            "expected_vol_monthly": out["expected_vol_monthly"],
            "quadratic_utility": out["quadratic_utility"],
            "growth_weight": out["growth_weight"],
            "defensive_weight": out["defensive_weight"],
        }

        for asset in ASSET_COLUMNS:
            row[f"w_{asset}"] = out["weights"][asset]

        rows.append(row)

        out["weights"].to_csv(
            PROCESSED_DIR / f"hmm_regime_optimal_weights_regime{regime}.csv",
            header=["weight"]
        )
        print(f"Saved weights for regime {regime} to "
              f"{PROCESSED_DIR / f'hmm_regime_optimal_weights_regime{regime}.csv'}")

    results_df = pd.DataFrame(rows)
    results_df.to_csv(PROCESSED_DIR / "hmm_regime_optimal_weights_summary.csv", index=False)
    print(f"Saved regime optimisation summary to "
          f"{PROCESSED_DIR / 'hmm_regime_optimal_weights_summary.csv'}")

    return results_df, results


def main():
    summary_df, cov_dict = load_regime_inputs()
    results_df, results = run_regime_optimisation(
        summary_df=summary_df,
        cov_dict=cov_dict,
        risk_aversion=3.0,
        min_growth=0.50,
        max_growth=0.90,
    )

    print("\n=== Regime-Aware Optimal Portfolio Summary ===")
    print(results_df)

    for regime, out in results.items():
        print(f"\n=== Optimal Weights: Regime {regime} ===")
        print(out["weights"])
        print(f"Expected monthly return: {out['expected_return_monthly']:.4f}")
        print(f"Expected monthly vol: {out['expected_vol_monthly']:.4f}")
        print(f"Quadratic utility: {out['quadratic_utility']:.6f}")
        print(f"Growth weight: {out['growth_weight']:.4f}")
        print(f"Defensive weight: {out['defensive_weight']:.4f}")


if __name__ == "__main__":
    main()