import pandas as pd
import numpy as np
from src.data.load_data import ASSET_COLUMNS
from src.utils.paths import PROCESSED_DIR


def load_returns_and_regimes(
    returns_filename: str = "monthly_returns_wide.csv",
    regime_filename: str = "hmm_regime_labels.csv"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    returns = pd.read_csv(PROCESSED_DIR / returns_filename)
    regimes = pd.read_csv(PROCESSED_DIR / regime_filename)

    returns["date"] = pd.to_datetime(returns["date"])
    regimes["date"] = pd.to_datetime(regimes["date"])

    return returns, regimes


def merge_returns_with_regimes(
    returns_df: pd.DataFrame,
    regime_df: pd.DataFrame
) -> pd.DataFrame:
    cols_to_keep = ["date", "regime", "regime_name"]
    merged = returns_df.merge(
        regime_df[cols_to_keep],
        on="date",
        how="inner"
    )
    return merged


def build_regime_asset_summary(
    merged_df: pd.DataFrame,
    annualise: bool = True,
) -> pd.DataFrame:
    rows = []

    for regime in sorted(merged_df["regime"].dropna().unique()):
        sub = merged_df[merged_df["regime"] == regime].copy()
        regime_name = sub["regime_name"].iloc[0]

        for asset in ASSET_COLUMNS:
            s = sub[asset].dropna()

            mean_monthly = s.mean()
            vol_monthly = s.std()
            ann_return = (1 + mean_monthly) ** 12 - 1 if annualise else np.nan
            ann_vol = vol_monthly * np.sqrt(12) if annualise else np.nan

            rows.append({
                "regime": regime,
                "regime_name": regime_name,
                "asset_code": asset,
                "n_obs": s.shape[0],
                "mean_monthly_return": mean_monthly,
                "vol_monthly": vol_monthly,
                "ann_return": ann_return,
                "ann_vol": ann_vol,
                "min_return": s.min(),
                "max_return": s.max(),
                "skewness": s.skew(),
                "kurtosis": s.kurt(),
            })

    summary = pd.DataFrame(rows)
    summary.to_csv(PROCESSED_DIR / "hmm_regime_asset_summary.csv", index=False)
    print(f"Regime asset summary saved to {PROCESSED_DIR / 'hmm_regime_asset_summary.csv'}")
    return summary


def build_regime_covariance_matrices(merged_df: pd.DataFrame) -> dict[int, pd.DataFrame]:
    cov_dict = {}

    for regime in sorted(merged_df["regime"].dropna().unique()):
        sub = merged_df.loc[merged_df["regime"] == regime, ASSET_COLUMNS].copy()
        cov = sub.cov()
        cov_dict[regime] = cov

        cov.to_csv(PROCESSED_DIR / f"hmm_regime_cov_regime{regime}.csv")
        print(f"Covariance matrix saved to {PROCESSED_DIR / f'hmm_regime_cov_regime{regime}.csv'}")

    return cov_dict


def build_regime_correlation_matrices(merged_df: pd.DataFrame) -> dict[int, pd.DataFrame]:
    corr_dict = {}

    for regime in sorted(merged_df["regime"].dropna().unique()):
        sub = merged_df.loc[merged_df["regime"] == regime, ASSET_COLUMNS].copy()
        corr = sub.corr()
        corr_dict[regime] = corr

        corr.to_csv(PROCESSED_DIR / f"hmm_regime_corr_regime{regime}.csv")
        print(f"Correlation matrix saved to {PROCESSED_DIR / f'hmm_regime_corr_regime{regime}.csv'}")

    return corr_dict


def build_regime_overview(merged_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for regime in sorted(merged_df["regime"].dropna().unique()):
        sub = merged_df[merged_df["regime"] == regime].copy()

        rows.append({
            "regime": regime,
            "regime_name": sub["regime_name"].iloc[0],
            "n_months": len(sub),
            "start_date": sub["date"].min(),
            "end_date": sub["date"].max(),
        })

    overview = pd.DataFrame(rows)
    overview.to_csv(PROCESSED_DIR / "hmm_regime_overview.csv", index=False)
    print(f"Regime overview saved to {PROCESSED_DIR / 'hmm_regime_overview.csv'}")
    return overview


def main():
    returns_df, regime_df = load_returns_and_regimes()
    merged_df = merge_returns_with_regimes(returns_df, regime_df)

    overview = build_regime_overview(merged_df)
    summary = build_regime_asset_summary(merged_df)
    cov_dict = build_regime_covariance_matrices(merged_df)
    corr_dict = build_regime_correlation_matrices(merged_df)

    print("\n=== Regime Overview ===")
    print(overview)

    print("\n=== Regime Asset Summary (head) ===")
    print(summary.head(15))

    for regime, cov in cov_dict.items():
        print(f"\n=== Covariance Matrix: Regime {regime} ===")
        print(cov)

    for regime, corr in corr_dict.items():
        print(f"\n=== Correlation Matrix: Regime {regime} ===")
        print(corr)


if __name__ == "__main__":
    main()