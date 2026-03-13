from hmmlearn.hmm import GaussianHMM
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
import numpy as np


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