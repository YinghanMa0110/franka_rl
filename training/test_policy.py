"""
Policy Validation Script
Loads a trained checkpoint and tests it in simulation.
"""

import time
import gymnasium as gym
import panda_gym
from stable_baselines3 import PPO

# ============ Configuration ============
CHECKPOINT_PATH = '../checkpoints/panda_reach_ppo_40000_steps'
ENV_NAME = 'PandaReach-v3'
REWARD_TYPE = 'dense'
NUM_EPISODES = 10
RENDER = True
STEP_DELAY = 0.05  # seconds between steps (for visualization)

# ============ Load Model ============
print(f"Loading checkpoint: {CHECKPOINT_PATH}")
model = PPO.load(CHECKPOINT_PATH)
print("Checkpoint loaded!")

# ============ Create Environment ============
render_mode = 'human' if RENDER else None
env = gym.make(ENV_NAME, reward_type=REWARD_TYPE, render_mode=render_mode)

# ============ Test ============
successes = 0

for i in range(NUM_EPISODES):
    obs, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward

        if RENDER:
            env.render()
            time.sleep(STEP_DELAY)

    success = info['is_success']
    successes += int(success)
    print(f'Episode {i+1}: reward={total_reward:.1f}, success={success}')

print(f'\nResults: {successes}/{NUM_EPISODES} episodes successful '
      f'({successes/NUM_EPISODES*100:.0f}%)')

env.close()
