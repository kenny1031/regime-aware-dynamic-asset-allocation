from xgboost import XGBClassifier
import pandas as pd


class XGBRegimeClassifier:
    def __init__(
        self,
        n_classes: int = 3,
        n_estimators: int = 300,
        max_depth: int = 3,
        learning_rate: float = 0.05,
        subsample: float = 0.9,
        colsample_bytree: float = 0.9,
        reg_lambda: float = 1.0,
        random_state: int = 42,
    ):
        self.n_classes = n_classes
        self.model = XGBClassifier(
            objective="multi:softprob",
            num_class=n_classes,
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_lambda=reg_lambda,
            random_state=random_state,
            eval_metric="mlogloss",
            use_label_encoder=False,
        )
        self.feature_names_ = None

    def fit(self, X, y, feature_names=None):
        self.model.fit(X, y)
        self.feature_names_ = feature_names
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def get_feature_importance(self) -> pd.DataFrame:
        if self.feature_names_ is None:
            feature_names = [f"feature_{i}" for i in range(len(self.model.feature_importances_))]
        else:
            feature_names = self.feature_names_

        importance_df = pd.DataFrame({
            "feature": feature_names,
            "importance": self.model.feature_importances_,
        }).sort_values("importance", ascending=False).reset_index(drop=True)

        return importance_df