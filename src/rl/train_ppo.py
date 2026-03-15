import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from src.rl.env import RegimeAllocationEnv
from src.utils.paths import PROCESSED_DIR


def make_train_env():
    return RegimeAllocationEnv(
        transaction_cost_bps=10.0,
        risk_penalty=0.0,
    )


def evaluate_agent(model, env: RegimeAllocationEnv) -> pd.DataFrame:
    obs, info = env.reset()

    done = False
    rows = []

    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        rows.append({
            "date": info["date"],
            "reward": reward,
            "portfolio_return": info["portfolio_return"],
            "transaction_cost": info["transaction_cost"],
            "turnover": info["turnover"],
            "wealth": info["wealth"],
            **{f"w_{i}": info["weights"][i] for i in range(len(info["weights"]))},
        })

        done = terminated or truncated

    out = pd.DataFrame(rows)
    return out


def annualised_return(returns: pd.Series) -> float:
    returns = returns.dropna()
    if len(returns) == 0:
        return np.nan
    return float((1 + returns).prod() ** (12 / len(returns)) - 1)


def annualised_volatility(returns: pd.Series) -> float:
    returns = returns.dropna()
    if len(returns) == 0:
        return np.nan
    return float(returns.std() * np.sqrt(12))


def sharpe_ratio(returns: pd.Series) -> float:
    vol = annualised_volatility(returns)
    if pd.isna(vol) or vol == 0:
        return np.nan
    return annualised_return(returns) / vol


def max_drawdown(wealth: pd.Series) -> float:
    running_max = wealth.cummax()
    dd = wealth / running_max - 1
    return float(dd.min())


def build_summary(eval_df: pd.DataFrame) -> pd.DataFrame:
    summary = pd.DataFrame([{
        "strategy": "PPO_RL",
        "n_months": len(eval_df),
        "ann_return": annualised_return(eval_df["reward"]),
        "ann_vol": annualised_volatility(eval_df["reward"]),
        "sharpe": sharpe_ratio(eval_df["reward"]),
        "max_drawdown": max_drawdown(eval_df["wealth"]),
        "avg_turnover": float(eval_df["turnover"].mean()),
        "final_wealth": float(eval_df["wealth"].iloc[-1]),
    }])
    return summary




def main():
    # Vectorise training env
    vec_env = make_vec_env(make_train_env, n_envs=1)
    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=3e-4,
        n_steps=128,
        batch_size=32,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        verbose=1,
    )

    print("\n=== Training PPO Agent ===")
    model.learn(total_timesteps=20000)

    model.save(str(PROCESSED_DIR / "ppo_regime_allocation_agent"))
    print(f"PPO model saved to {PROCESSED_DIR / 'ppo_regime_allocation_agent.zip'}")

    # Evaluate on a fresh env
    eval_env = RegimeAllocationEnv(
        transaction_cost_bps=10.0,
        risk_penalty=0.0,
    )
    eval_df = evaluate_agent(model, eval_env)
    eval_df.to_csv(PROCESSED_DIR / "ppo_rl_evaluation_paths.csv", index=False)

    summary = build_summary(eval_df)
    summary.to_csv(PROCESSED_DIR / "ppo_rl_evaluation_summary.csv", index=False)

    print(f"PPO evaluation paths saved to {PROCESSED_DIR / 'ppo_rl_evaluation_paths.csv'}")
    print(f"PPO evaluation summary saved to {PROCESSED_DIR / 'ppo_rl_evaluation_summary.csv'}")

    print("\n=== PPO Evaluation Summary ===")
    print(summary)

    print("\n=== Evaluation Sample ===")
    print(eval_df.head(10))


if __name__ == "__main__":
    main()