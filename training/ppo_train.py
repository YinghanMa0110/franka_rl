"""
PPO Training Script for Franka Panda
Trains a PPO policy on PandaReach using panda-gym and Stable Baselines3.
"""

import gymnasium as gym
import panda_gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

# ============ Configuration ============
ENV_NAME = 'PandaReach-v3'
REWARD_TYPE = 'dense'
TOTAL_TIMESTEPS = 3_000_000
CHECKPOINT_FREQ = 10_000
CHECKPOINT_DIR = '../checkpoints/'
CHECKPOINT_PREFIX = 'panda_reach_ppo'
FINAL_SAVE_PATH = '../checkpoints/panda_reach_final'

# ============ Create Environment ============
env = gym.make(ENV_NAME, reward_type=REWARD_TYPE)

# Print environment info
print(f"Environment: {ENV_NAME}")
print(f"Observation space: {env.observation_space}")
print(f"Action space: {env.action_space}")
print(f"Reward type: {REWARD_TYPE}")

# ============ Create Model ============
# MultiInputPolicy is required because panda-gym uses dict observations
model = PPO('MultiInputPolicy', env, verbose=1)

# ============ Setup Checkpoint Callback ============
# Auto-saves every CHECKPOINT_FREQ steps to prevent losing progress
checkpoint_callback = CheckpointCallback(
    save_freq=CHECKPOINT_FREQ,
    save_path=CHECKPOINT_DIR,
    name_prefix=CHECKPOINT_PREFIX
)

# ============ Train ============
print(f"\nStarting training for {TOTAL_TIMESTEPS} timesteps...")
model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=checkpoint_callback)

# ============ Save Final Model ============
model.save(FINAL_SAVE_PATH)
print(f"\nTraining complete! Final model saved to {FINAL_SAVE_PATH}")
