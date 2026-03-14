import pandas as pd
import numpy as np
from src.regime.models.baselines import KMeansRegimeModel
from src.regime.preprocess import (
    load_regime_features,
    prepare_regime_input,
    DEFAULT_HMM_FEATURES
)
from src.regime.plotting import (
    plot_regime_strip,
    plot_regime_probabilities,
    plot_growth_with_regime_shading
)
from src.utils.paths import PROCESSED_DIR


def build_regime_label_table(
    clean_df: pd.DataFrame,
    labels: np.ndarray,
    probs: np.ndarray,
    n_states: int,
) -> pd.DataFrame:
    result = pd.DataFrame({
        "date": clean_df["date"],
        "regime": labels,
    })

    for k in range(n_states):
        result[f"prob_regime_{k}"] = probs[:, k]

    return result


def build_regime_summary(
    regime_df: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    summary = (
        regime_df.groupby("regime")[feature_cols]
        .agg(["mean", "std", "count"])
    )
    return summary


def relabel_regimes_by_risk(regime_df: pd.DataFrame) -> pd.DataFrame:
    """
    Relabel regimes for interpretability using average growth_vol_12m.
    Lowest vol -> lower regime id.
    Highest vol -> higher regime id.
    """
    temp = regime_df.copy()

    risk_order = (
        temp.groupby("regime")["growth_vol_12m"]
        .mean()
        .sort_values()
        .index
        .tolist()
    )

    mapping = {old_label: new_label for new_label, old_label in enumerate(risk_order)}
    temp["regime"] = temp["regime"].map(mapping)

    prob_cols = [c for c in temp.columns if c.startswith("prob_regime_")]
    reordered = pd.DataFrame(index=temp.index)

    for old_label, new_label in mapping.items():
        reordered[f"prob_regime_{new_label}"] = temp[f"prob_regime_{old_label}"]

    temp = temp.drop(columns=prob_cols)
    temp = pd.concat([temp, reordered], axis=1)

    return temp


def assign_regime_names(regime_df: pd.DataFrame) -> pd.DataFrame:
    """
    Name regimes using simple economic logic:
    - highest growth_proxy_ret -> Risk-On
    - lowest growth_proxy_ret -> Risk-Off
    - middle -> Neutral
    """
    temp = regime_df.copy()

    ranking = (
        temp.groupby("regime")["growth_proxy_ret"]
        .mean()
        .sort_values()
    )

    ordered = ranking.index.tolist()
    if len(ordered) != 3:
        name_map = {r: f"Regime_{r}" for r in ordered}
    else:
        name_map = {
            ordered[0]: "Risk-Off",
            ordered[1]: "Neutral",
            ordered[2]: "Risk-On",
        }

    temp["regime_name"] = temp["regime"].map(name_map)
    return temp


def main():
    n_states = 3

    features_df = load_regime_features("regime_features.csv")
    clean_df, X_df, scaler = prepare_regime_input(
        features_df,
        feature_cols=DEFAULT_HMM_FEATURES,
        standardise=True,
    )

    X = X_df.values

    model = KMeansRegimeModel(
        n_states=n_states,
        random_state=42,
    )
    model.fit(X)

    labels = model.predict(X)
    probs = model.predict_proba(X)

    regime_df = build_regime_label_table(clean_df, labels, probs, n_states=n_states)

    interpret_cols = [
        "growth_proxy_ret",
        "defensive_proxy_ret",
        "gd_spread",
        "growth_vol_12m",
        "avg_corr_all_12m",
        "growth_drawdown",
        "worst_asset_ret",
        "num_negative_assets",
    ]

    regime_df = regime_df.merge(
        clean_df[["date"] + interpret_cols],
        on="date",
        how="left"
    )

    regime_df = relabel_regimes_by_risk(regime_df)
    regime_df = assign_regime_names(regime_df)

    summary = build_regime_summary(
        regime_df,
        feature_cols=interpret_cols,
    )

    regime_df.to_csv(PROCESSED_DIR / "kmeans_regime_labels.csv", index=False)
    summary.to_csv(PROCESSED_DIR / "kmeans_regime_summary.csv")

    print(f"KMeans regime labels saved to {PROCESSED_DIR / 'kmeans_regime_labels.csv'}")
    print(f"KMeans regime summary saved to {PROCESSED_DIR / 'kmeans_regime_summary.csv'}")
    print(regime_df.head(15))
    print(summary)

    plot_regime_strip(regime_df, model_name="KMeans")
    plot_regime_probabilities(regime_df, model_name="KMeans")
    plot_growth_with_regime_shading(
        regime_df,
        value_col="growth_proxy_ret",
        model_name="KMeans"
    )


if __name__ == "__main__":
    main()