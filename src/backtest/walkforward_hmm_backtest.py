import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.data.load_data import ASSET_COLUMNS
from src.regime.regime_models import HMMRegimeModel
from src.regime.preprocess import DEFAULT_HMM_FEATURES
from src.portfolio.regime_optimisation import optimise_regime_portfolio, build_default_bounds
from src.utils.paths import PROCESSED_DIR


def build_static_balanced_weights() -> pd.Series:
    """
    Simple 70/30 balanced benchmark.
    """
    weights = pd.Series({
        "AEQ": 0.20,
        "ILE_H": 0.10,
        "ILE_UH": 0.15,
        "ALP": 0.10,
        "ILP_H": 0.05,
        "ILI_H": 0.10,
        "AFI": 0.15,
        "IFI_H": 0.10,
        "CASH": 0.05,
    })
    return weights.reindex(ASSET_COLUMNS)


def load_walkforward_inputs() -> pd.DataFrame:
    returns = pd.read_csv(PROCESSED_DIR / "monthly_returns_wide.csv")
    features = pd.read_csv(PROCESSED_DIR / "regime_features.csv")

    returns["date"] = pd.to_datetime(returns["date"])
    features["date"] = pd.to_datetime(features["date"])

    merged = returns.merge(features, on="date", how="inner")
    merged = merged.sort_values("date").reset_index(drop=True)
    return merged


def relabel_states_by_risk(training_features_df: pd.DataFrame, raw_labels: np.ndarray):
    """
    Relabel HMM states for semantic consistency across refits.

    First-pass rule:
    - sort states by average growth_vol_12m ascending
    - low vol -> lower regime index
    - high vol -> higher regime index

    Returns
    -------
    relabeled_states : np.ndarray
    mapping : dict
        raw_state -> relabeled_state
    """
    temp = training_features_df.copy().reset_index(drop=True)
    temp["state_raw"] = raw_labels

    state_risk = (
        temp.groupby("state_raw")["growth_vol_12m"]
        .mean()
        .sort_values()
    )

    mapping = {old_state: new_state for new_state, old_state in enumerate(state_risk.index.tolist())}
    relabeled_states = np.array([mapping[x] for x in raw_labels])

    return relabeled_states, mapping


def estimate_regime_moments(
    training_returns_df: pd.DataFrame,
    relabeled_states: np.ndarray
) -> tuple[dict, dict, dict]:
    """
    Estimate regime-specific expected returns and covariance matrices
    from the training sample only.

    Returns
    -------
    mu_dict : dict[int, pd.Series]
    sigma_dict : dict[int, pd.DataFrame]
    count_dict : dict[int, int]
    """
    temp = training_returns_df.copy().reset_index(drop=True)
    temp["regime"] = relabeled_states

    mu_dict = {}
    sigma_dict = {}
    count_dict = {}

    for regime in sorted(temp["regime"].unique()):
        sub = temp.loc[temp["regime"] == regime, ASSET_COLUMNS].copy()

        mu = sub.mean()

        sigma = sub.cov()
        sigma = sigma.fillna(0.0)
        sigma = sigma + np.eye(len(ASSET_COLUMNS)) * 1e-8

        mu_dict[regime] = mu
        sigma_dict[regime] = sigma
        count_dict[regime] = len(sub)

    return mu_dict, sigma_dict, count_dict


def get_current_hard_regime(hmm_model: HMMRegimeModel, X_train_scaled: np.ndarray, mapping: dict) -> int:
    raw_labels = hmm_model.predict(X_train_scaled)
    raw_current = int(raw_labels[-1])
    current_regime = mapping[raw_current]
    return current_regime


def get_current_soft_probs(
    hmm_model: HMMRegimeModel,
    X_train_scaled: np.ndarray,
    mapping: dict,
    n_states: int = 3,
) -> np.ndarray:
    raw_probs = hmm_model.predict_proba(X_train_scaled)[-1]

    relabeled_probs = np.zeros(n_states)
    for raw_state, new_state in mapping.items():
        relabeled_probs[new_state] = raw_probs[raw_state]

    return relabeled_probs


def compute_turnover(current_w: pd.Series, previous_w: pd.Series | None) -> float:
    if previous_w is None:
        return 0.0
    return float(np.abs(current_w - previous_w).sum())


def compute_portfolio_return(asset_returns: pd.Series, weights: pd.Series) -> float:
    return float((asset_returns * weights).sum())


def run_static_balanced_on_oos_dates(
    df: pd.DataFrame,
    min_train_months: int = 84,
    transaction_cost_bps: float = 0.0,
) -> pd.DataFrame:
    """
    Run a fixed balanced benchmark on the same out-of-sample dates
    used by the walk-forward strategy.
    """
    weights = build_static_balanced_weights()

    rows = []
    prev_weights = None

    for t in range(min_train_months, len(df) - 1):
        next_row = df.iloc[t + 1].copy()
        next_returns = next_row[ASSET_COLUMNS].astype(float)

        turnover = compute_turnover(weights, prev_weights)
        tc = turnover * (transaction_cost_bps / 10000.0)

        gross_ret = compute_portfolio_return(next_returns, weights)
        net_ret = gross_ret - tc

        rows.append({
            "date": next_row["date"],
            "strategy": "WF_Static_Balanced",
            "gross_return": gross_ret,
            "transaction_cost": tc,
            "net_return": net_ret,
            "turnover": turnover,
            **{f"w_{a}": weights[a] for a in ASSET_COLUMNS},
        })

        prev_weights = weights.copy()

    out = pd.DataFrame(rows)
    if not out.empty:
        out["wealth"] = (1 + out["net_return"]).cumprod() * 100

    return out


def annualised_return(returns: pd.Series) -> float:
    returns = returns.dropna()
    if len(returns) == 0:
        return np.nan
    return float((1 + returns).prod() ** (12 / len(returns)) - 1)


def annualised_volatility(returns: pd.Series) -> float:
    returns = returns.dropna()
    if len(returns) == 0:
        return np.nan
    return float(returns.std() * np.sqrt(12))


def sharpe_ratio(returns: pd.Series) -> float:
    vol = annualised_volatility(returns)
    if pd.isna(vol) or vol == 0:
        return np.nan
    return annualised_return(returns) / vol


def max_drawdown(wealth: pd.Series) -> float:
    running_max = wealth.cummax()
    dd = wealth / running_max - 1
    return float(dd.min())


def build_performance_summary(backtest_df: pd.DataFrame, filename: str) -> pd.DataFrame:
    rows = []

    for strategy, sub in backtest_df.groupby("strategy"):
        rows.append({
            "strategy": strategy,
            "n_months": len(sub),
            "ann_return": annualised_return(sub["net_return"]),
            "ann_vol": annualised_volatility(sub["net_return"]),
            "sharpe": sharpe_ratio(sub["net_return"]),
            "max_drawdown": max_drawdown(sub["wealth"]),
            "avg_turnover": float(sub["turnover"].mean()),
            "final_wealth": float(sub["wealth"].iloc[-1]),
        })

    summary = pd.DataFrame(rows).sort_values("strategy").reset_index(drop=True)
    summary.to_csv(PROCESSED_DIR / filename, index=False)
    return summary


def optimise_with_fallback(
    mu: pd.Series,
    sigma: pd.DataFrame,
    bounds_map: dict[str, tuple[float, float]],
    risk_aversion: float,
    min_growth: float | None,
    max_growth: float | None,
    fallback_weights: pd.Series,
) -> pd.Series:
    """
    Try optimisation under the intended constraints first.
    If it fails, relax growth constraints.
    If that still fails, fall back to equal-weight.
    """
    try:
        out = optimise_regime_portfolio(
            mu=mu,
            sigma=sigma,
            risk_aversion=risk_aversion,
            bounds_map=bounds_map,
            min_growth=min_growth,
            max_growth=max_growth,
        )
        return out["weights"]

    except RuntimeError:
        try:
            out = optimise_regime_portfolio(
                mu=mu,
                sigma=sigma,
                risk_aversion=risk_aversion,
                bounds_map=bounds_map,
                min_growth=None,
                max_growth=None,
            )
            return out["weights"]

        except RuntimeError:
            return fallback_weights.copy()


def run_walkforward_backtest(
    min_train_months: int = 84,
    min_regime_obs: int = 24,
    n_states: int = 3,
    risk_aversion: float = 3.0,
    transaction_cost_bps: float = 10.0,
    min_growth: float = 0.50,
    max_growth: float = 0.90,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Expanding-window walk-forward HMM backtest.

    At time t:
    - use data up to t only
    - fit HMM on training features
    - estimate regime-specific moments using training sample
    - optimise regime-specific portfolios
    - apply weights to returns in t+1

    Robustness additions:
    - minimum regime observations
    - optimisation fallback logic
    """
    df = load_walkforward_inputs()
    bounds_map = build_default_bounds()

    hard_rows = []
    soft_rows = []

    prev_hard_w = None
    prev_soft_w = None

    fallback_weights = pd.Series(
        np.array([1.0 / len(ASSET_COLUMNS)] * len(ASSET_COLUMNS)),
        index=ASSET_COLUMNS,
        dtype=float
    )

    for t in range(min_train_months, len(df) - 1):
        train_full = df.iloc[:t + 1].copy()
        next_row = df.iloc[t + 1].copy()

        # Use only dates with complete selected features
        train_features = train_full[["date"] + DEFAULT_HMM_FEATURES].dropna().reset_index(drop=True)

        # Need enough feature observations
        if len(train_features) < min_train_months:
            continue

        X_train = train_features[DEFAULT_HMM_FEATURES].values
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        # Fit HMM
        hmm = HMMRegimeModel(
            n_states=n_states,
            covariance_type="full",
            random_state=42
        )
        hmm.fit(X_train_scaled)

        # Predict states on training sample
        raw_labels = hmm.predict(X_train_scaled)
        relabeled_states, mapping = relabel_states_by_risk(train_features, raw_labels)

        # Align training asset returns to the same dates used in feature sample
        train_returns = train_full.merge(train_features[["date"]], on="date", how="inner")
        train_returns = train_returns.sort_values("date").reset_index(drop=True)

        # Estimate regime-specific moments from training sample only
        mu_dict, sigma_dict, count_dict = estimate_regime_moments(train_returns, relabeled_states)

        # Full-sample fallback moments from training set
        full_sample_mu = train_returns[ASSET_COLUMNS].mean()
        full_sample_sigma = train_returns[ASSET_COLUMNS].cov().fillna(0.0)
        full_sample_sigma = full_sample_sigma + np.eye(len(ASSET_COLUMNS)) * 1e-8

        # Optimise regime-specific portfolios with fallback
        regime_weight_map = {}

        for regime in sorted(mu_dict.keys()):
            mu_use = mu_dict[regime]
            sigma_use = sigma_dict[regime]

            # If regime sample too small, fall back to full training sample moments
            if count_dict[regime] < min_regime_obs:
                mu_use = full_sample_mu
                sigma_use = full_sample_sigma

            weights = optimise_with_fallback(
                mu=mu_use,
                sigma=sigma_use,
                bounds_map=bounds_map,
                risk_aversion=risk_aversion,
                min_growth=min_growth,
                max_growth=max_growth,
                fallback_weights=fallback_weights,
            )

            regime_weight_map[regime] = weights

        # HARD SWITCH: choose current most likely regime at time t
        current_regime = get_current_hard_regime(hmm, X_train_scaled, mapping)
        hard_w = regime_weight_map[current_regime].copy()

        # SOFT SWITCH: posterior-weighted combination at time t
        current_probs = get_current_soft_probs(hmm, X_train_scaled, mapping, n_states=n_states)

        soft_w = pd.Series(
            np.zeros(len(ASSET_COLUMNS)),
            index=ASSET_COLUMNS,
            dtype=float
        )
        for regime in sorted(regime_weight_map.keys()):
            soft_w += current_probs[regime] * regime_weight_map[regime]

        # Realised next-month returns
        next_returns = next_row[ASSET_COLUMNS].astype(float)

        # Hard strategy
        hard_turnover = compute_turnover(hard_w, prev_hard_w)
        hard_tc = hard_turnover * (transaction_cost_bps / 10000.0)
        hard_gross = compute_portfolio_return(next_returns, hard_w)
        hard_net = hard_gross - hard_tc

        hard_rows.append({
            "date": next_row["date"],
            "strategy": "WF_HMM_Hard",
            "predicted_regime": current_regime,
            "gross_return": hard_gross,
            "transaction_cost": hard_tc,
            "net_return": hard_net,
            "turnover": hard_turnover,
            **{f"w_{a}": hard_w[a] for a in ASSET_COLUMNS},
        })

        # Soft strategy
        soft_turnover = compute_turnover(soft_w, prev_soft_w)
        soft_tc = soft_turnover * (transaction_cost_bps / 10000.0)
        soft_gross = compute_portfolio_return(next_returns, soft_w)
        soft_net = soft_gross - soft_tc

        soft_rows.append({
            "date": next_row["date"],
            "strategy": "WF_HMM_Soft",
            "gross_return": soft_gross,
            "transaction_cost": soft_tc,
            "net_return": soft_net,
            "turnover": soft_turnover,
            "prob_regime_0": current_probs[0],
            "prob_regime_1": current_probs[1],
            "prob_regime_2": current_probs[2],
            **{f"w_{a}": soft_w[a] for a in ASSET_COLUMNS},
        })

        prev_hard_w = hard_w.copy()
        prev_soft_w = soft_w.copy()

    hard_df = pd.DataFrame(hard_rows)
    soft_df = pd.DataFrame(soft_rows)

    if not hard_df.empty:
        hard_df["wealth"] = (1 + hard_df["net_return"]).cumprod() * 100
    if not soft_df.empty:
        soft_df["wealth"] = (1 + soft_df["net_return"]).cumprod() * 100

    return hard_df, soft_df


def main():
    min_train_months = 84
    transaction_cost_bps = 10.0

    # Run walk-forward HMM strategies
    hard_df, soft_df = run_walkforward_backtest(
        min_train_months=min_train_months,
        min_regime_obs=24,
        n_states=3,
        risk_aversion=3.0,
        transaction_cost_bps=transaction_cost_bps,
        min_growth=0.50,
        max_growth=0.90,
    )

    # Run fixed benchmark on the same OOS horizon
    df = load_walkforward_inputs()
    static_balanced_df = run_static_balanced_on_oos_dates(
        df=df,
        min_train_months=min_train_months,
        transaction_cost_bps=0.0,
    )

    all_bt = pd.concat([hard_df, soft_df, static_balanced_df], ignore_index=True)

    hard_df.to_csv(PROCESSED_DIR / "walkforward_hmm_hard_paths.csv", index=False)
    soft_df.to_csv(PROCESSED_DIR / "walkforward_hmm_soft_paths.csv", index=False)
    static_balanced_df.to_csv(PROCESSED_DIR / "walkforward_static_balanced_paths.csv", index=False)
    all_bt.to_csv(PROCESSED_DIR / "walkforward_hmm_paths.csv", index=False)

    print(f"Saved hard-switch paths to {PROCESSED_DIR / 'walkforward_hmm_hard_paths.csv'}")
    print(f"Saved soft-switch paths to {PROCESSED_DIR / 'walkforward_hmm_soft_paths.csv'}")
    print(f"Saved static balanced paths to {PROCESSED_DIR / 'walkforward_static_balanced_paths.csv'}")
    print(f"Saved combined paths to {PROCESSED_DIR / 'walkforward_hmm_paths.csv'}")

    summary = build_performance_summary(
        all_bt,
        filename="walkforward_hmm_summary.csv"
    )

    print("\n=== Walk-Forward Performance Summary ===")
    print(summary)

    print("\n=== Final Wealth by Strategy ===")
    print(all_bt.groupby("strategy")["wealth"].last())


if __name__ == "__main__":
    main()