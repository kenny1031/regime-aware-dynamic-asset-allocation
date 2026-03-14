import numpy as np
from src.rl.env import RegimeAllocationEnv

env = RegimeAllocationEnv(transaction_cost_bps=10.0, risk_penalty=0.0)
obs, info = env.reset()

print("Initial obs shape:", obs.shape)
print("Initial info:", info)

done = False
step_count = 0

while not done and step_count < 5:
    action = np.random.randn(env.n_assets)
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Step {step_count}: reward={reward:.4f}, wealth={info['wealth']:.2f}")
    done = terminated or truncated
    step_count += 1