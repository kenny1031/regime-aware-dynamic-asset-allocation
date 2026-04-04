import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.mixture import GaussianMixture
from src.regime.preprocess import (
    load_regime_features,
    prepare_regime_input,
    DEFAULT_HMM_FEATURES
)
from src.regime.plotting import (
    plot_regime_timeline,
    plot_regime_strip,
    plot_regime_probabilities,
    plot_growth_with_regime_shading
)
from src.utils.paths import PROCESSED_DIR


# ===================
# Hidden Markov Model
# ===================
class HMMRegimeModel:
    def __init__(
        self,
        n_states=3,
        covariance_type="full",
        random_state=42
    ):
        self.n_states = n_states
        self.model = GaussianHMM(
            n_components=n_states,
            covariance_type=covariance_type,
            random_state=random_state
        )

    def fit(self, X):
        self.model.fit(X)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)


# ======================
# Gaussian Mixture Model
# ======================
class GMMRegimeModel:
    def __init__(self, n_states=3, random_state=42):
        self.n_states = n_states
        self.model = GaussianMixture(
            n_components=n_states,
            random_state=random_state
        )

    def fit(self, X):
        self.model.fit(X)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)


models = {
    "hmm": HMMRegimeModel, # Primary regime model
    "gmm": GMMRegimeModel  # Baseline comparator
}


def build_regime_label_table(
    clean_df: pd.DataFrame,
    labels,
    probs,
    n_states: int
) -> pd.DataFrame:
    result = pd.DataFrame({
        "date": clean_df["date"],
        "regime": labels
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


def build_transition_matrix_df(model, regime_name_map: dict[int, str]) -> pd.DataFrame:
    transmat = model.model.transmat_

    state_labels = [regime_name_map[i] for i in range(transmat.shape[0])]

    df_trans = pd.DataFrame(
        transmat,
        index=[f"from_{s}" for s in state_labels],
        columns=[f"to_{s}" for s in state_labels],
    )
    return df_trans


def build_regime_duration_summary(
    regime_df: pd.DataFrame,
    regime_name_map: dict[int, str]
) -> pd.DataFrame:
    temp = regime_df[["date", "regime"]].copy()
    temp = temp.sort_values("date").reset_index(drop=True)

    temp["block"] = (temp["regime"] != temp["regime"].shift(1)).cumsum()

    durations = (
        temp.groupby(["block", "regime"])
        .size()
        .reset_index(name="duration")
    )

    summary = (
        durations.groupby("regime")["duration"]
        .agg(
            n_spells="count",
            mean_duration="mean",
            median_duration="median",
            max_duration="max",
            min_duration="min",
            std_duration="std",
        )
        .reset_index()
    )

    summary["regime_name"] = summary["regime"].map(regime_name_map)
    summary = summary[
        ["regime", "regime_name", "n_spells", "mean_duration",
         "median_duration", "max_duration", "min_duration", "std_duration"]
    ]

    return summary


def relabel_regimes_by_risk(regime_df: pd.DataFrame) -> pd.DataFrame:
    """
    Optional: relabel regimes so that regime 0/1/2 have a more stable ordering
    based on average growth_vol_12m, in ascending order by risk.
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

    prob_cols = [col for col in temp.columns if col.startswith("prob_regime_")]
    old_prob_cols = prob_cols.copy()

    reordered = pd.DataFrame(index=temp.index)
    for old_label, new_label in mapping.items():
        reordered[f"prob_regime_{new_label}"] = temp[f"prob_regime_{old_label}"]

    temp = temp.drop(columns=prob_cols)
    temp = pd.concat([temp, reordered], axis=1)

    return temp


def main() -> None:
    model_names = models.keys()
    n_state = 3
    # Load and preprocess
    features_df = load_regime_features("regime_features.csv")
    clean_df, X_df, scaler = prepare_regime_input(
        features_df,
        feature_cols=DEFAULT_HMM_FEATURES,
        standardise=True
    )

    X = X_df.values

    # Fit HMM and GMM
    for model_name in model_names:
        model = models[model_name](n_states=n_state)
        model.fit(X)

        labels = model.predict(X)
        probs = model.predict_proba(X)

        # save regime labels and posterior probabilities
        regime_df = build_regime_label_table(clean_df, labels, probs, n_state)

        # Keep some key original features for interpretation
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

        # Optional relabelling for interpretability
        regime_df = relabel_regimes_by_risk(regime_df)

        regime_name_map = {
            0: "Neutral",
            1: "Risk-On",
            2: "Risk-Off"
        }

        regime_df["regime_name"] = regime_df["regime"].map(regime_name_map)

        #plot_regime_timeline(regime_df, model_name=model_name)
        plot_regime_strip(regime_df, model_name=model_name)
        plot_regime_probabilities(regime_df, model_name=model_name)
        plot_growth_with_regime_shading(
            regime_df=regime_df,
            value_col="growth_vol_12m",
            model_name=model_name
        )

        summary = build_regime_summary(
            regime_df,
            feature_cols=interpret_cols
        )

        regime_df.to_csv(PROCESSED_DIR / f"{model_name}_regime_labels.csv", index=False)
        summary.to_csv(PROCESSED_DIR / f"{model_name}_regime_summary.csv")

        print(f"{model_name.upper()} regime labels saved to "
              f"{PROCESSED_DIR / f'{model_name}_regime_labels.csv'}")
        print(f"{model_name.upper()} regime summary saved to "
              f"{PROCESSED_DIR / f'{model_name}_regime_summary.csv'}")
        print(regime_df.head(15))
        print(summary)
        print(regime_df["regime"].value_counts().sort_index())

        if model_name == "hmm":
            transition_df = build_transition_matrix_df(model, regime_name_map)
            transition_df.to_csv(PROCESSED_DIR / "hmm_transition_matrix.csv")
            duration_summary = build_regime_duration_summary(regime_df, regime_name_map)
            duration_summary.to_csv(PROCESSED_DIR / "hmm_regime_duration_summary.csv", index=False)
            print("HMM Transition Matrix:")
            print(transition_df)
            print("HMM Regime Duration Summary:")
            print(duration_summary)


if __name__ == "__main__":
    main()