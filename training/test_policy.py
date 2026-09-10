"""
Policy Validation Script
Test the same SB3 PandaReach checkpoint used on the real Franka.
"""

import time
import gymnasium as gym
import panda_gym
import numpy as np

from stable_baselines3 import PPO


# ============ Configuration ============

CHECKPOINT_PATH = "checkpoints/panda_reach_ppo_40000_steps"

ENV_NAME = "PandaReach-v3"
REWARD_TYPE = "dense"

NUM_EPISODES = 20

RENDER = True
STEP_DELAY = 0.05


# ============ Load Model ============

print(f"Loading checkpoint: {CHECKPOINT_PATH}")

model = PPO.load(
    CHECKPOINT_PATH
)

print("Checkpoint loaded!")


# ============ Create Environment ============

render_mode = "human" if RENDER else None

env = gym.make(
    ENV_NAME,
    reward_type=REWARD_TYPE,
    render_mode=render_mode
)


# ============ Test ============

successes = 0
final_distances = []


for i in range(NUM_EPISODES):

    obs, info = env.reset(
        seed=i
    )

    done = False
    total_reward = 0.0
    step = 0

    while not done:

        action, _ = model.predict(
            obs,
            deterministic=True
        )

        obs, reward, terminated, truncated, info = env.step(
            action
        )

        done = terminated or truncated

        total_reward += reward
        step += 1

        if RENDER:
            time.sleep(STEP_DELAY)

    achieved = obs["achieved_goal"]
    desired = obs["desired_goal"]

    final_distance = np.linalg.norm(
        achieved - desired
    )

    final_distances.append(
        final_distance
    )

    success = bool(
        info.get(
            "is_success",
            final_distance < 0.05
        )
    )

    successes += int(success)

    print(
        f"Episode {i + 1}: "
        f"reward={total_reward:.2f}, "
        f"success={success}, "
        f"final_error={final_distance * 100:.2f} cm, "
        f"steps={step}"
    )


# ============ Summary ============

success_rate = (
    successes
    / NUM_EPISODES
    * 100
)

mean_error = (
    np.mean(final_distances)
    * 100
)

print("\n" + "=" * 60)

print(
    f"Results: "
    f"{successes}/{NUM_EPISODES} successful "
    f"({success_rate:.1f}%)"
)

print(
    f"Mean final error: "
    f"{mean_error:.2f} cm"
)

print("=" * 60)

env.close()
