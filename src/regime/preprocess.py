import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.utils.paths import PROCESSED_DIR

DEFAULT_HMM_FEATURES = [
    "growth_proxy_ret",
    "defensive_proxy_ret",
    "gd_spread",
    "growth_vol_12m",
    "avg_corr_all_12m",
    "growth_drawdown",
    "worst_asset_ret",
    "num_negative_assets",
]

def load_regime_features(filename: str = "regime_features.csv") -> pd.DataFrame:
    file_path = PROCESSED_DIR / filename
    df = pd.read_csv(file_path)
    df["date"] = pd.to_datetime(df["date"])
    return df

def prepare_regime_input(
    features_df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    standardise: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame, object]:
    """
    Prepare regime-detection input matrix.

    Returns
    -------
    clean_df : pd.DataFrame
        DataFrame containing date + selected feature columns after dropna.
    X_df : pd.DataFrame
        Feature matrix as DataFrame (scaled if standardize=True).
    scaler : object
        Fitted StandardScaler if standardize=True, else None.
    """
    if feature_cols is None:
        feature_cols = DEFAULT_HMM_FEATURES

    required_cols = ["date"] + feature_cols
    clean_df = features_df[required_cols].copy()
    clean_df = clean_df.dropna().reset_index(drop=True)

    X_df = clean_df[feature_cols].copy()

    scaler = None
    if standardise:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_df)
        X_df = pd.DataFrame(X_scaled, columns=feature_cols, index=clean_df.index)

    return clean_df, X_df, scaler