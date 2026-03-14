import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from src.utils.paths import PROCESSED_DIR
from src.data.load_data import ASSET_COLUMNS
from src.data.build_regime_features import FEATURE_COLUMNS


PROB_COLUMNS = ["prob_regime_0", "prob_regime_1", "prob_regime_2"]

class RegimeAllocationEnv(gym.Env):
    """
    Monthly multi-asset allocation environment.

    Observation:
        [regime features, HMM regime probabilities, current weights]

    Action:
        unconstrained vector in R^n
        mapped to long-only fully-invested weights via softmax

    Reward:
        next-month portfolio return
        - transaction cost penalty
        - optional risk penalty
    """
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        transaction_cost_bps: float = 10.0,
        risk_penalty: float = 0.0,
        start_index: int | None = None,
        end_index: int | None = None,
    ):
        super().__init__()

        self.transaction_cost_bps = transaction_cost_bps
        self.risk_penalty = risk_penalty

        returns_df = pd.read_csv(PROCESSED_DIR / "monthly_returns_wide.csv")
        returns_df["date"] = pd.to_datetime(returns_df["date"])

        features_df = pd.read_csv(PROCESSED_DIR / "regime_features.csv")
        features_df["date"] = pd.to_datetime(features_df["date"])

        hmm_df = pd.read_csv(PROCESSED_DIR / "hmm_regime_labels.csv")
        hmm_df["date"] = pd.to_datetime(hmm_df["date"])

        self.df = (
            returns_df
            .merge(features_df[["date"] + FEATURE_COLUMNS], on="date", how="inner")
            .merge(hmm_df[["date"] + PROB_COLUMNS], on="date", how="inner")
            .dropna()
            .sort_values("date")
            .reset_index(drop=True)
        )

        if start_index is None:
            start_index = 0
        if end_index is None:
            end_index = len(self.df) - 1

        self.start_index = start_index
        self.end_index = end_index

        self.n_assets = len(ASSET_COLUMNS)
        self.obs_dim = len(FEATURE_COLUMNS) + len(PROB_COLUMNS) + self.n_assets

        self.action_space = spaces.Box(
            low=-5.0,
            high=5.0,
            shape=(self.n_assets,),
            dtype=np.float32,
        )

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_dim,),
            dtype=np.float32,
        )

        self.current_step = None
        self.current_weights = None
        self.wealth = None
        self.done = None

    def _get_observation(self) -> np.ndarray:
        row = self.df.iloc[self.current_step]

        feature_vals = row[FEATURE_COLUMNS].astype(float).values
        prob_vals = row[PROB_COLUMNS].astype(float).values
        weight_vals = self.current_weights.astype(float)

        obs = np.concatenate([feature_vals, prob_vals, weight_vals])
        return obs.astype(np.float32)

    @staticmethod
    def softmax(x: np.ndarray) -> np.ndarray:
        z = x - np.max(x)
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z)

    def _action_to_weights(self, action: np.ndarray) -> np.ndarray:
        return self.softmax(action)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_step = self.start_index
        self.current_weights = np.array([1.0 / self.n_assets] * self.n_assets, dtype=float)
        self.wealth = 100.0
        self.done = False

        obs = self._get_observation()
        info = {
            "date": self.df.iloc[self.current_step]["date"],
            "wealth": self.wealth,
        }
        return obs, info

    def step(self, action):
        if self.done:
            raise RuntimeError("Episode is done. Call reset() first.")

        next_weights = self._action_to_weights(np.asarray(action, dtype=float))

        row = self.df.iloc[self.current_step]
        asset_returns = row[ASSET_COLUMNS].astype(float).values

        turnover = np.abs(next_weights - self.current_weights).sum()
        transaction_cost = turnover * (self.transaction_cost_bps / 10000.0)

        portfolio_return = float(np.dot(next_weights, asset_returns))
        risk_penalty_term = self.risk_penalty * float(np.var(asset_returns))
        reward = portfolio_return - transaction_cost - risk_penalty_term

        self.wealth *= (1.0 + reward)
        self.current_weights = next_weights

        self.current_step += 1
        if self.current_step >= self.end_index:
            self.done = True

        if not self.done:
            obs = self._get_observation()
        else:
            obs = np.zeros(self.obs_dim, dtype=np.float32)

        info = {
            "date": row["date"],
            "portfolio_return": portfolio_return,
            "transaction_cost": transaction_cost,
            "turnover": turnover,
            "wealth": self.wealth,
            "weights": self.current_weights.copy(),
        }

        terminated = self.done
        truncated = False

        return obs, reward, terminated, truncated, info

    def render(self):
        row = self.df.iloc[min(self.current_step, self.end_index - 1)]
        print(
            f"Step={self.current_step}, "
            f"Date={row['date']}, "
            f"Wealth={self.wealth:.2f}, "
            f"Weights={np.round(self.current_weights, 3)}"
        )

