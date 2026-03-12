import os
import pandas as pd
import numpy as np
from src.utils.paths import (
    RAW_DIR,
    PROCESSED_DIR,
    INTERIM_DIR,
    FIGURES_DIR
)
import matplotlib.pyplot as plt

ERROR_TOKENS = {
    "#N/A",
    "#N/A N/A",
    "#NAME?",
    "#N/A Requesting Data...",
    ""
}

COLUMN_MAP = {
    "ASA52 Index": "AEQ",
    "NDDLWI Index": "ILE_H",
    "NDLEEGF Index": "ILE_UH",
    "RDAU Index": "ALP",
    "FDCIISAH Index": "ILP_H",
    "HEDGNAV Index": "ILI_H",
    "BACM0 Index": "AFI",
    "H03432AU Index": "IFI_H",
    "BAUBIL Index": "CASH"
}

ASSET_COLUMNS = [asset_name for asset_name in COLUMN_MAP.values()]

# ==============
# Data Cleaning
# ==============
def load_index_levels(
    filename: str,
    sheet_name: str = "Market Data Construction",
    usecols: str = "L:U",
    skiprows: int = 2
) -> pd.DataFrame:
    """
    Load raw index level data from the Market Data Construction sheet.

    Parameters
    ----------
    filename : str
        Excel file name under data/raw/.
    sheet_name : str
        Sheet containing cleaned benchmark index levels.
    usecols : str
        Excel-style column range for the level block.
    skiprows : int
        Number of rows to skip before the header row.

    Returns
    -------
    pd.DataFrame
        Cleaned levels dataframe.
    """
    file_path = RAW_DIR / filename
    df = pd.read_excel(
        file_path,
        sheet_name=sheet_name,
        usecols=usecols,
        skiprows=skiprows
    )

    # Clean
    df = df.copy()
    df.columns = df.iloc[0, :].fillna("date")
    df = df.rename(columns=COLUMN_MAP)
    df = df.drop(index=df.iloc[0:3, :].index)
    df = df.replace(list(ERROR_TOKENS), np.nan).infer_objects(copy=False)

    # Ensure data type consistent
    for col in ASSET_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Ensure dates are sorted in ascending order
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values("date").reset_index(drop=True)

    # Save to processed directory
    df.to_csv(PROCESSED_DIR / f"index_levels_cleaned.csv")
    print(f"Index levels saved to {PROCESSED_DIR}")
    return df

def apply_manual_pre_inception_rules(df: pd.DataFrame) -> pd.DataFrame:
    """
    First version: manually blank out known placeholder history.
    Adjust once exact cutoffs are verified.
    """
    df = df.copy()

    # Example placeholder rule for ILP_H:
    # if level equals 100 for early history, treat as missing
    early_mask = df["ILP_H"] == 1000
    df.loc[early_mask, "ILP_H"] = np.nan

    return df


# =========================
# Generate returns datasets
# =========================
def compute_simple_returns(levels_df: pd.DataFrame) -> pd.DataFrame:

    df = levels_df.copy()

    returns = df[ASSET_COLUMNS].pct_change(fill_method=None)
    returns.insert(0, "date", df["date"])

    returns.to_csv(PROCESSED_DIR / "monthly_returns_wide.csv", index=False)
    returns.to_csv(INTERIM_DIR / "returns_wide.csv", index=False)

    print(f"Returns saved to {PROCESSED_DIR / 'monthly_returns_wide.csv'}")
    return returns

def build_long_returns(returns_wide: pd.DataFrame, levels_df: pd.DataFrame) -> pd.DataFrame:
    long_df = returns_wide.melt(
        id_vars="date",
        var_name="asset_code",
        value_name="simple_return"
    )
    long_df["log_return"] = np.where(
        long_df["simple_return"].notna(),
        np.log1p(long_df["simple_return"]),
        np.nan
    )
    long_df["is_missing"] = long_df["simple_return"].isna()
    level_long = levels_df.melt(
        id_vars="date",
        var_name="asset_code",
        value_name="level"
    )

    long_df = long_df.merge(level_long, on=["date", "asset_code"], how="left")
    long_df["is_pre_inception"] = long_df["level"].isna() & long_df["simple_return"].isna()
    long_df["source"] = "Bloomberg PX_LAST"
    long_df = long_df.drop(columns=["level"])

    long_df.to_csv(PROCESSED_DIR / "monthly_returns.csv", index=False)
    print(f"Long-format returns saved to {PROCESSED_DIR / 'monthly_returns.csv'}")

    return long_df


# =========================================
# Exploratory Data Analysis and diagnostics
# =========================================
def build_missingness_report(levels_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for col in ASSET_COLUMNS:
        series = levels_df[["date", col]].copy()
        first_valid = series.loc[series[col].notna(), "date"].min()
        last_valid = series.loc[series[col].notna(), "date"].max()

        missing_count = series[col].isna().sum()
        missing_pct = series[col].isna().mean()

        rows.append({
            "asset_code": col,
            "first_valid_date": first_valid,
            "last_valid_date": last_valid,
            "missing_count": missing_count,
            "missing_pct": missing_pct
        })

    report = pd.DataFrame(rows)
    report.to_csv(PROCESSED_DIR / "missingness_report.csv", index=False)
    print(f"Missingness report saved to {PROCESSED_DIR / 'missingness_report.csv'}")

    report["first_valid_date"] = pd.to_datetime(report["first_valid_date"])
    report["last_valid_date"] = pd.to_datetime(report["last_valid_date"])

    plt.figure(figsize=(10, 5))
    for i, row in report.iterrows():
        plt.plot(
            [row["first_valid_date"], row["last_valid_date"]],
            [row["asset_code"], row["asset_code"]],
            linewidth=6
        )

    plt.title("Asset Data Availability")
    plt.xlabel("Date")
    plt.ylabel("Asset")
    plt.tight_layout()

    # Save figure
    os.makedirs(FIGURES_DIR / "eda", exist_ok=True)
    save_path = os.path.join(FIGURES_DIR, "eda/asset_availability.png")
    plt.savefig(fname=save_path)
    print(f"Missingness saved to {save_path}")
    return report

def build_summary_stats(returns_wide: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in ASSET_COLUMNS:
        s = returns_wide[col].dropna()

        mean_monthly = s.mean()
        vol_monthly = s.std()
        ann_return = (1 + mean_monthly) ** 12 - 1
        ann_vol = vol_monthly * np.sqrt(12)

        rows.append({
            "asset_code": col,
            "n_obs": s.shape[0],
            "mean_monthly_return": mean_monthly,
            "vol_monthly": vol_monthly,
            "ann_return": ann_return,
            "ann_vol": ann_vol,
            "min_return": s.min(),
            "max_return": s.max(),
            "skewness": s.skew(),
            "kurtosis": s.kurt()
        })

    stats = pd.DataFrame(rows)
    stats.to_csv(PROCESSED_DIR / "summary_stats.csv", index=False)
    print(f"Summary stats saved to {PROCESSED_DIR / 'summary_stats.csv'}")
    return stats

def build_correlation_matrix(returns_wide: pd.DataFrame) -> pd.DataFrame:
    corr = returns_wide[ASSET_COLUMNS].corr()
    corr.to_csv(PROCESSED_DIR / "correlation_matrix.csv", index=True)
    print(f"Correlation matrix saved to {PROCESSED_DIR / 'correlation_matrix.csv'}")

    plt.figure(figsize=(8, 6))
    im = plt.imshow(corr, aspect="auto")
    plt.colorbar(im)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45)
    plt.yticks(range(len(corr.index)), corr.index)
    plt.title("Correlation Matrix of Monthly Returns")
    plt.tight_layout()
    os.makedirs(FIGURES_DIR / "eda", exist_ok=True)
    save_path = os.path.join(FIGURES_DIR, "eda/correlation_matrix.png")
    plt.savefig(fname=save_path)
    print(f"Correlation matrix saved to {save_path}")
    return corr

def build_growth_of_100(returns_wide: pd.DataFrame) -> pd.DataFrame:
    df = returns_wide.copy()
    growth = (1 + df[ASSET_COLUMNS].fillna(0)).cumprod() * 100
    growth.insert(0, "date", df["date"])
    growth.to_csv(PROCESSED_DIR / "growth_of_100.csv", index=False)
    print(f"Growth of 100 saved to {PROCESSED_DIR / 'growth_of_100.csv'}")
    growth["date"] = pd.to_datetime(growth["date"])

    plt.figure(figsize=(12, 6))
    for col in growth.columns[1:]:
        plt.plot(growth["date"], growth[col], label=col)

    # Save plot
    plt.title("Growth of $100 Across Asset Classes")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.legend()
    plt.tight_layout()

    # Save figure
    os.makedirs(FIGURES_DIR / "eda", exist_ok=True)
    save_path = os.path.join(FIGURES_DIR, "eda/growth_of_100.png")
    plt.savefig(fname=save_path)
    print(f"Growth of 100 plot saved to {save_path}")
    return growth

def build_rolling_volatility(returns_wide: pd.DataFrame, window: int = 12) -> pd.DataFrame:
    df = returns_wide.copy()
    rolling_vol = df[ASSET_COLUMNS].rolling(window=window).std() * np.sqrt(12)
    rolling_vol.insert(0, "date", df["date"])
    rolling_vol.to_csv(PROCESSED_DIR / f"rolling_vol_{window}m.csv", index=False)
    print(f"Rolling volatility saved to {PROCESSED_DIR / f'rolling_vol_{window}m.csv'}")

    rolling_vol["date"] = pd.to_datetime(rolling_vol["date"])

    plt.figure(figsize=(12, 6))
    for col in ["AEQ", "ILE_H", "ILE_UH", "AFI", "CASH"]:
        plt.plot(rolling_vol["date"], rolling_vol[col], label=col)

    plt.title("12-Month Rolling Annualised Volatility")
    plt.xlabel("Date")
    plt.ylabel("Volatility")
    plt.legend()
    plt.tight_layout()

    # Save figure
    os.makedirs(FIGURES_DIR / "eda", exist_ok=True)
    save_path = os.path.join(FIGURES_DIR, "eda/rolling_volatility.png")
    plt.savefig(fname=save_path)
    print(f"Rolling volatility plot saved to {save_path}")
    return rolling_vol


# =======================
# Main execution function
# =======================
def main():
    # Handle raw dataset
    levels = load_index_levels("BBG Data (2000-2025).xlsx")
    levels = apply_manual_pre_inception_rules(levels)

    # Obtain returns
    returns_wide = compute_simple_returns(levels)
    returns_long = build_long_returns(returns_wide, levels)

    # EDA
    missingness = build_missingness_report(levels)
    summary_stats = build_summary_stats(returns_wide)
    corr = build_correlation_matrix(returns_wide)
    growth = build_growth_of_100(returns_wide)
    rolling_vol = build_rolling_volatility(returns_wide, window=12)

    print(levels.head())
    print(returns_wide.head())
    print(returns_long.head())
    print(missingness)
    print(summary_stats)
    print(corr)
    print(growth.head())
    print(rolling_vol.head(15))

if __name__ == "__main__":
    main()