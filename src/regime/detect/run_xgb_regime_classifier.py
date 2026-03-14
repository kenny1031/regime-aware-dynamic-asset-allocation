import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from src.regime.preprocess import load_regime_features
from src.regime.regime_models import  XGBRegimeClassifier
from src.utils.paths import PROCESSED_DIR


DEFAULT_XGB_FEATURES = [
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


def load_hmm_labels(filename: str = "hmm_regime_labels.csv") -> pd.DataFrame:
    df = pd.read_csv(PROCESSED_DIR / filename)
    df["date"] = pd.to_datetime(df["date"])
    return df


def build_supervised_dataset(
    feature_cols: list[str] | None = None,
    feature_filename: str = "regime_features.csv",
    label_filename: str = "hmm_regime_labels.csv",
) -> pd.DataFrame:
    if feature_cols is None:
        feature_cols = DEFAULT_XGB_FEATURES

    features_df = load_regime_features(feature_filename)
    labels_df = load_hmm_labels(label_filename)

    keep_label_cols = ["date", "regime", "regime_name"]
    merged = features_df.merge(
        labels_df[keep_label_cols],
        on="date",
        how="inner"
    )

    merged = merged[["date"] + feature_cols + ["regime", "regime_name"]].copy()
    merged = merged.dropna().sort_values("date").reset_index(drop=True)

    return merged


def time_based_split(
    df: pd.DataFrame,
    train_frac: float = 0.7,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_idx = int(len(df) * train_frac)

    train_df = df.iloc[:split_idx].copy().reset_index(drop=True)
    test_df = df.iloc[split_idx:].copy().reset_index(drop=True)

    return train_df, test_df


def build_prediction_table(
    test_df: pd.DataFrame,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    n_classes: int = 3,
) -> pd.DataFrame:
    out = test_df[["date", "regime", "regime_name"]].copy()
    out = out.rename(columns={"regime": "true_regime", "regime_name": "true_regime_name"})
    out["pred_regime"] = y_pred

    name_map = {
        0: "Neutral",
        1: "Risk-On",
        2: "Risk-Off",
    }
    out["pred_regime_name"] = out["pred_regime"].map(name_map)

    for k in range(n_classes):
        out[f"prob_regime_{k}"] = y_prob[:, k]

    return out


def main():
    feature_cols = DEFAULT_XGB_FEATURES

    df = build_supervised_dataset(feature_cols=feature_cols)
    train_df, test_df = time_based_split(df, train_frac=0.7)

    X_train = train_df[feature_cols].values
    y_train = train_df["regime"].values

    X_test = test_df[feature_cols].values
    y_test = test_df["regime"].values

    clf = XGBRegimeClassifier(
        n_classes=3,
        n_estimators=300,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=42,
    )

    clf.fit(X_train, y_train, feature_names=feature_cols)

    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(
        cm,
        index=["true_0", "true_1", "true_2"],
        columns=["pred_0", "pred_1", "pred_2"],
    )

    report = classification_report(y_test, y_pred, digits=4)

    pred_df = build_prediction_table(test_df, y_pred, y_prob, n_classes=3)
    importance_df = clf.get_feature_importance()

    pred_df.to_csv(PROCESSED_DIR / "xgb_regime_predictions.csv", index=False)
    importance_df.to_csv(PROCESSED_DIR / "xgb_regime_feature_importance.csv", index=False)
    cm_df.to_csv(PROCESSED_DIR / "xgb_regime_confusion_matrix.csv", index=True)

    with open(PROCESSED_DIR / "xgb_regime_classification_report.txt", "w") as f:
        f.write(report)

    print(f"XGBoost predictions saved to {PROCESSED_DIR / 'xgb_regime_predictions.csv'}")
    print(f"XGBoost feature importance saved to {PROCESSED_DIR / 'xgb_regime_feature_importance.csv'}")
    print(f"XGBoost confusion matrix saved to {PROCESSED_DIR / 'xgb_regime_confusion_matrix.csv'}")
    print(f"XGBoost classification report saved to {PROCESSED_DIR / 'xgb_regime_classification_report.txt'}")

    print("\n=== Train/Test Split ===")
    print(f"Train months: {len(train_df)}")
    print(f"Test months:  {len(test_df)}")
    print(f"Train start:  {train_df['date'].min().date()}")
    print(f"Train end:    {train_df['date'].max().date()}")
    print(f"Test start:   {test_df['date'].min().date()}")
    print(f"Test end:     {test_df['date'].max().date()}")

    print("\n=== XGBoost Regime Classification Metrics ===")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Macro F1:  {macro_f1:.4f}")

    print("\n=== Confusion Matrix ===")
    print(cm_df)

    print("\n=== Classification Report ===")
    print(report)

    print("\n=== Top Feature Importances ===")
    print(importance_df.head(10))

    print("\n=== Prediction Sample ===")
    print(pred_df.head(15))


if __name__ == "__main__":
    main()