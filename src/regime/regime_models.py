import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from hmmlearn.hmm import GaussianHMM
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from xgboost import XGBClassifier


# ======================
# LSTM Regime Classifier
# ======================
class LSTMRegimeClassifier(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        num_classes: int = 3,
        dropout: float = 0.2
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape: (batch_size, seq_len, input_size)

        Returns
        -------
        logits : torch.Tensor
            Shape: (batch_size, num_classes)
        """
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Use final hidden state from the last LSTM layer
        last_hidden = h_n[-1] # shape: (batch_size, hidden_state)

        logits = self.classifier(last_hidden)
        return logits


# ==================
# XGBoost Classifier
# ==================
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


# ==================
# K-Means Clustering
# ==================
class KMeansRegimeModel:
    def __init__(self, n_states=3, random_state=42):
        self.n_states = n_states
        self.model = KMeans(
            n_clusters=n_states,
            random_state=random_state,
            n_init=20
        )

    def fit(self, X):
        self.model.fit(X)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        # KMeans has no actual posterior probability
        labels = self.model.predict(X)
        probs = np.zeros((len(labels), self.model.n_clusters))
        probs[np.arange(len(labels)), labels] = 1.0

        return probs