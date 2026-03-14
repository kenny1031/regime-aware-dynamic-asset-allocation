import numpy as np
import pandas as pd
from src.data.load_data import ASSET_COLUMNS
from src.utils.paths import PROCESSED_DIR

def load_backtest_inputs():
    returns = pd.read_csv(PROCESSED_DIR / "monthly_returns_wide.csv")
    returns["date"] = pd.to_datetime(returns["date"])

    regime_labels = pd.read_csv(PROCESSED_DIR / "hmm_regime_labels.csv")
    regime_labels["date"] = pd.to_datetime(regime_labels["date"])

    regime_opt = pd.read_csv(PROCESSED_DIR / "hmm_regime_optimal_weights_summary.csv")

    return returns, regime_labels, regime_opt


def build_static_balanced_weights() -> pd.Series:
    """
    Simple hand-crafted balanced benchmark:
    70% growth / 30% defensive
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


def load_regime_weight_map(regime_opt_df: pd.DataFrame) -> dict[int, pd.Series]:
    weight_map = {}

    for regime in sorted(regime_opt_df["regime"].unique()):
        row = regime_opt_df[regime_opt_df["regime"] == regime].iloc[0]
        weights = pd.Series(
            {asset: row[f"w_{asset}"] for asset in ASSET_COLUMNS}
        )
        weight_map[regime] = weights

    return weight_map


def build_static_mv_weights(regime_opt_df: pd.DataFrame) -> pd.Series:
    """
    First-pass proxy for a static MV benchmark:
    use Neutral regime weights as a simple long-run fixed optimiser baseline.
    """
    row = regime_opt_df[regime_opt_df["regime_name"] == "Neutral"].iloc[0]
    weights = pd.Series({asset: row[f"w_{asset}"] for asset in ASSET_COLUMNS})
    return weights


def align_returns_and_regimes(
    returns_df: pd.DataFrame,
    regime_df: pd.DataFrame,
) -> pd.DataFrame:
    merged = returns_df.merge(regime_df, on="date", how="inner")
    merged = merged.sort_values("date").reset_index(drop=True)
    return merged


def compute_turnover(current_w: pd.Series, previous_w: pd.Series | None) -> float:
    if previous_w is None:
        return 0.0
    return float(np.abs(current_w - previous_w).sum())


def compute_portfolio_return(
    asset_returns: pd.Series,
    weights: pd.Series,
) -> float:
    return float((asset_returns * weights).sum())


def run_static_strategy(
    merged_df: pd.DataFrame,
    weights: pd.Series,
    strategy_name: str,
    transaction_cost_bps: float = 0.0,
) -> pd.DataFrame:
    rows = []
    prev_weights = None

    for _, row in merged_df.iterrows():
        asset_ret = row[ASSET_COLUMNS].astype(float)

        turnover = compute_turnover(weights, prev_weights)
        tc = turnover * (transaction_cost_bps / 10000.0)

        gross_ret = compute_portfolio_return(asset_ret, weights)
        net_ret = gross_ret - tc

        rows.append({
            "date": row["date"],
            "strategy": strategy_name,
            "gross_return": gross_ret,
            "transaction_cost": tc,
            "net_return": net_ret,
            "turnover": turnover,
        })

        prev_weights = weights.copy()

    out = pd.DataFrame(rows)
    out["wealth"] = (1 + out["net_return"]).cumprod() * 100
    return out


def run_hard_switch_strategy(
    merged_df: pd.DataFrame,
    regime_weight_map: dict[int, pd.Series],
    strategy_name: str = "HMM_Hard_Switch",
    transaction_cost_bps: float = 10.0,
) -> pd.DataFrame:
    rows = []
    prev_weights = None

    for _, row in merged_df.iterrows():
        regime = int(row["regime"])
        weights = regime_weight_map[regime].copy()
        asset_ret = row[ASSET_COLUMNS].astype(float)

        turnover = compute_turnover(weights, prev_weights)
        tc = turnover * (transaction_cost_bps / 10000.0)

        gross_ret = compute_portfolio_return(asset_ret, weights)
        net_ret = gross_ret - tc

        rows.append({
            "date": row["date"],
            "strategy": strategy_name,
            "regime": regime,
            "gross_return": gross_ret,
            "transaction_cost": tc,
            "net_return": net_ret,
            "turnover": turnover,
        })

        prev_weights = weights.copy()

    out = pd.DataFrame(rows)
    out["wealth"] = (1 + out["net_return"]).cumprod() * 100
    return out


def run_soft_switch_strategy(
    merged_df: pd.DataFrame,
    regime_weight_map: dict[int, pd.Series],
    strategy_name: str = "HMM_Soft_Switch",
    transaction_cost_bps: float = 10.0,
) -> pd.DataFrame:
    rows = []
    prev_weights = None

    prob_cols = [f"prob_regime_{k}" for k in sorted(regime_weight_map.keys())]

    for _, row in merged_df.iterrows():
        probs = np.array([row[col] for col in prob_cols], dtype=float)

        weights = sum(
            probs[k] * regime_weight_map[k]
            for k in sorted(regime_weight_map.keys())
        )
        weights = pd.Series(weights, index=ASSET_COLUMNS)

        asset_ret = row[ASSET_COLUMNS].astype(float)

        turnover = compute_turnover(weights, prev_weights)
        tc = turnover * (transaction_cost_bps / 10000.0)

        gross_ret = compute_portfolio_return(asset_ret, weights)
        net_ret = gross_ret - tc

        rows.append({
            "date": row["date"],
            "strategy": strategy_name,
            "gross_return": gross_ret,
            "transaction_cost": tc,
            "net_return": net_ret,
            "turnover": turnover,
        })

        prev_weights = weights.copy()

    out = pd.DataFrame(rows)
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


def build_performance_summary(backtest_df: pd.DataFrame) -> pd.DataFrame:
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
    summary.to_csv(PROCESSED_DIR / "regime_backtest_summary.csv", index=False)
    print(f"Performance summary saved to {PROCESSED_DIR / 'regime_backtest_summary.csv'}")
    return summary


def main():
    returns_df, regime_df, regime_opt_df = load_backtest_inputs()

    merged_df = align_returns_and_regimes(returns_df, regime_df)
    regime_weight_map = load_regime_weight_map(regime_opt_df)

    static_balanced_w = build_static_balanced_weights()
    static_mv_w = build_static_mv_weights(regime_opt_df)

    bt_static_balanced = run_static_strategy(
        merged_df,
        static_balanced_w,
        strategy_name="Static_Balanced",
        transaction_cost_bps=0.0,
    )

    bt_static_mv = run_static_strategy(
        merged_df,
        static_mv_w,
        strategy_name="Static_MV_Neutral",
        transaction_cost_bps=0.0,
    )

    bt_hard = run_hard_switch_strategy(
        merged_df,
        regime_weight_map,
        strategy_name="HMM_Hard_Switch",
        transaction_cost_bps=10.0,
    )

    bt_soft = run_soft_switch_strategy(
        merged_df,
        regime_weight_map,
        strategy_name="HMM_Soft_Switch",
        transaction_cost_bps=10.0,
    )

    all_bt = pd.concat(
        [bt_static_balanced, bt_static_mv, bt_hard, bt_soft],
        ignore_index=True
    )

    all_bt.to_csv(PROCESSED_DIR / "regime_backtest_paths.csv", index=False)
    print(f"Backtest paths saved to {PROCESSED_DIR / 'regime_backtest_paths.csv'}")

    summary = build_performance_summary(all_bt)

    print("\n=== Backtest Performance Summary ===")
    print(summary)

    print("\n=== Final Wealth by Strategy ===")
    print(all_bt.groupby("strategy")["wealth"].last())


if __name__ == "__main__":
    main()