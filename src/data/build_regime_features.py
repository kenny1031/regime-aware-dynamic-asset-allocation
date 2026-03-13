import numpy as np
import pandas as pd
from src.data.load_data import ASSET_COLUMNS
from src.utils.paths import PROCESSED_DIR, INTERIM_DIR


EQUITY_ASSETS = [ASSET_COLUMNS[i] for i in range(3)]
GROWTH_ASSETS = [ASSET_COLUMNS[i] for i in range(len(ASSET_COLUMNS) - 3)]
DEFENSIVE_ASSETS = [ASSET_COLUMNS[i] for i in range(len(ASSET_COLUMNS) - 3, len(ASSET_COLUMNS))]


# Helper: proxy returns
def build_proxy_returns(returns_wide: pd.DataFrame) -> pd.DataFrame:
    df = returns_wide.copy()
    df["equity_proxy_ret"] = df[EQUITY_ASSETS].mean(axis=1, skipna=True)
    df["growth_proxy_ret"] = df[GROWTH_ASSETS].mean(axis=1, skipna=True)
    df["defensive_proxy_ret"] = df[DEFENSIVE_ASSETS].mean(axis=1, skipna=True)
    df["gd_spread"] = df["growth_proxy_ret"] - df["defensive_proxy_ret"]

    return df


# ==================
# Rolling volatility
# ==================
def add_rolling_vol_features(
    df: pd.DataFrame,
    window: int=12
) -> pd.DataFrame:
    out = df.copy()
    out[f"AEQ_vol_{window}m"] = out["AEQ"].rolling(window).std()
    out[f"ILE_UH_vol_{window}m"] = out["ILE_UH"].rolling(window).std()
    out[f"growth_vol_{window}m"] = out["growth_proxy_ret"].rolling(window).std()

    out["cross_sec_vol"] = out[["AEQ", "ILE_H", "ILE_UH", "ALP", "ILP_H", "ILI_H",
                                "AFI", "IFI_H", "CASH"]].std(axis=1, skipna=True)
    return out


# ========
# Momentum
# ========
def rolling_cum_return(series: pd.Series, window: int) -> pd.Series:
    return (1 + series).rolling(window).apply(np.prod, raw=True) - 1

def add_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["AEQ_mom_3m"] = rolling_cum_return(out["AEQ"], 3)
    out["AEQ_mom_12m"] = rolling_cum_return(out["AEQ"], 12)
    out["growth_mom_3m"] = rolling_cum_return(out["growth_proxy_ret"], 3)
    out["growth_mom_12m"] = rolling_cum_return(out["growth_proxy_ret"], 12)

    return out


# ===================
# Rolling correlation
# ===================
def rolling_corr(
    series1: pd.Series,
    series2: pd.Series,
    window: int
) -> pd.Series:
    return series1.rolling(window).corr(series2)

def average_pairwise_corr(
    df: pd.DataFrame,
    columns: list[str],
    window: int=12
) -> pd.Series:
    corr_values = []
    for t in range(len(df)):
        if t < window - 1:
            corr_values.append(np.nan)
            continue

        window_df = df.iloc[t-window+1 : t+1][columns]
        corr_matrix = window_df.corr()

        vals = []
        for i in range(len(columns)):
            for j in range(i+1, len(columns)):
                vals.append(corr_matrix.iloc[i, j])

        corr_values.append(np.nanmean(vals) if len(vals) > 0 else np.nan)

    return pd.Series(corr_values, index=df.index)

def add_correlation_features(df: pd.DataFrame, window: int=12) -> pd.DataFrame:
    out = df.copy()

    out[f"corr_AEQ_AFI_{window}m"] = rolling_corr(out["AEQ"], out["AFI"], window)
    out[f"corr_growth_def_{window}m"] = rolling_corr(
        out["growth_proxy_ret"],
        out["defensive_proxy_ret"], window
    )
    out[f"avg_corr_all_{window}m"] = average_pairwise_corr(out, ASSET_COLUMNS, window)
    out[f"avg_corr_growth_{window}m"] = average_pairwise_corr(out, GROWTH_ASSETS, window)

    return out


# =================
# Drawdown / stress
# =================
def compute_drawdown(series: pd.Series) -> pd.Series:
    wealth = (1 + series.fillna(0)).cumprod()
    running_max = wealth.cummax()
    drawdown = (wealth - running_max) / running_max
    return drawdown

def add_stress_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["AEQ_drawdown"] = compute_drawdown(out["AEQ"])
    out["growth_drawdown"] = compute_drawdown(out["growth_proxy_ret"])

    out["worst_asset_ret"] = out[ASSET_COLUMNS].min(axis=1, skipna=True)
    valid_counts = out[ASSET_COLUMNS].notna().sum(axis=1)
    neg_counts = (out[ASSET_COLUMNS] < 0).sum(axis=1)
    out["num_negative_assets"] = np.where(valid_counts > 0, neg_counts, np.nan)

    return out


# ==============
# Build features
# ==============
def build_regime_features(returns_wide: pd.DataFrame) -> pd.DataFrame:
    df = returns_wide.copy()
    df = build_proxy_returns(df)
    df = add_rolling_vol_features(df, window=12)
    df = add_momentum_features(df)
    df = add_correlation_features(df, window=12)
    df = add_stress_features(df)

    feature_cols = [
        "date",
        "equity_proxy_ret",
        "growth_proxy_ret",
        "defensive_proxy_ret",
        "gd_spread",
        "AEQ_vol_12m",
        "ILE_UH_vol_12m",
        "growth_vol_12m",
        "cross_sec_vol",
        "AEQ_mom_3m",
        "AEQ_mom_12m",
        "growth_mom_3m",
        "growth_mom_12m",
        "corr_AEQ_AFI_12m",
        "corr_growth_def_12m",
        "avg_corr_all_12m",
        "avg_corr_growth_12m",
        "AEQ_drawdown",
        "growth_drawdown",
        "worst_asset_ret",
        "num_negative_assets",
    ]

    features = df[feature_cols].copy()
    features.to_csv(PROCESSED_DIR / "regime_features.csv", index=False)

    return features

def build_regime_feature_metadata() -> pd.DataFrame:
    metadata = pd.DataFrame([
        ["equity_proxy_ret", "Average monthly return across AEQ, ILE_H, ILE_UH", "return"],
        ["growth_proxy_ret", "Average monthly return across growth assets", "return"],
        ["defensive_proxy_ret", "Average monthly return across defensive assets", "return"],
        ["gd_spread", "Growth minus defensive monthly return spread", "spread"],
        ["AEQ_vol_12m", "12-month rolling volatility of Australian equities", "volatility"],
        ["ILE_UH_vol_12m", "12-month rolling volatility of unhedged international equities", "volatility"],
        ["growth_vol_12m", "12-month rolling volatility of growth proxy", "volatility"],
        ["cross_sec_vol", "Cross-sectional standard deviation of asset returns in a month", "dispersion"],
        ["AEQ_mom_3m", "3-month cumulative return of Australian equities", "momentum"],
        ["AEQ_mom_12m", "12-month cumulative return of Australian equities", "momentum"],
        ["growth_mom_3m", "3-month cumulative return of growth proxy", "momentum"],
        ["growth_mom_12m", "12-month cumulative return of growth proxy", "momentum"],
        ["corr_AEQ_AFI_12m", "12-month rolling correlation between AEQ and AFI", "correlation"],
        ["corr_growth_def_12m", "12-month rolling correlation between growth and defensive proxies", "correlation"],
        ["avg_corr_all_12m", "12-month average pairwise correlation across all assets", "correlation"],
        ["avg_corr_growth_12m", "12-month average pairwise correlation across growth assets", "correlation"],
        ["AEQ_drawdown", "Running drawdown of Australian equities", "drawdown"],
        ["growth_drawdown", "Running drawdown of growth proxy", "drawdown"],
        ["worst_asset_ret", "Worst single-asset monthly return", "stress"],
        ["num_negative_assets", "Number of assets with negative monthly return", "breadth"],
    ], columns=["feature_name", "description", "feature_group"])

    metadata.to_csv(PROCESSED_DIR / "regime_feature_metadata.csv", index=False)
    return metadata


if __name__ == "__main__":
    returns_wide = pd.read_csv(INTERIM_DIR / "returns_wide.csv")
    regime_features = build_regime_features(returns_wide)
    regime_features_metadata = build_regime_feature_metadata()

    print(regime_features.head(15))
    print(regime_features_metadata)