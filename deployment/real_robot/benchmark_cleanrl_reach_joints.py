import os
import sys
import csv
import time
import argparse
from datetime import datetime

import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces

import panda_py


# ============================================================
# Project paths
# ============================================================

ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

TRAIN_DIR = os.path.join(
    ROOT,
    "training",
    "cleanrl",
)

sys.path.insert(0, TRAIN_DIR)

from ppo_continuous_action import Agent as PPOAgent
from sac_reach_joints import Actor as SACActor


# ============================================================
# Robot
# ============================================================

HOSTNAME = "192.168.1.8"

CONTROL_FREQ = 20
DT = 1.0 / CONTROL_FREQ

MAX_EP_STEPS = 50

GOAL_THRESHOLD = 0.02  # 2 cm


# ============================================================
# IMPORTANT:
# panda-gym joint action semantics:
#
# action ∈ [-1, 1]^7
# delta_q = action * 0.05 rad
# q_target = q_current + delta_q
# ============================================================

JOINT_ACTION_SCALE = 0.05


# ============================================================
# Franka joint limits
#
# Keep a margin away from hard limits.
# Verify against lab configuration tomorrow.
# ============================================================

JOINT_LOW = np.array([
    -2.8973,
    -1.7628,
    -2.8973,
    -3.0718,
    -2.8973,
    -0.0175,
    -2.8973,
])

JOINT_HIGH = np.array([
    2.8973,
    1.7628,
    2.8973,
    -0.0698,
    2.8973,
    3.7525,
    2.8973,
])

JOINT_MARGIN = 0.10

SAFE_JOINT_LOW = JOINT_LOW + JOINT_MARGIN
SAFE_JOINT_HIGH = JOINT_HIGH - JOINT_MARGIN


# ============================================================
# Workspace monitoring
#
# Same idea as your old real-robot script.
# Adjust tomorrow if needed.
# ============================================================

WORKSPACE_LOW = np.array([
    0.25,
    -0.25,
    0.00,
])

WORKSPACE_HIGH = np.array([
    0.65,
    0.25,
    0.60,
])


# ============================================================
# Coordinate conversion
#
# IMPORTANT:
# This needs to match the coordinate convention used by the
# simulation policy.
#
# -1.0 is copied from your previous real deployment.
# Verify this tomorrow before benchmark.
# ============================================================

BASE_OFFSET = np.array([
    -1.0,
    0.0,
    0.0,
])


def real_to_sim(real_pos):
    return np.asarray(real_pos) + BASE_OFFSET


# ============================================================
# Model paths
# ============================================================

PPO_RUN_DIR = os.path.join(
    ROOT,
    "runs",
    "PandaReach-v3__ppo_continuous_action__1__1789070756",
)

PPO_MODEL_PATH = os.path.join(
    PPO_RUN_DIR,
    "ppo_continuous_action.cleanrl_model",
)

PPO_OBS_RMS_PATH = os.path.join(
    PPO_RUN_DIR,
    "ppo_continuous_action.obs_rms.npz",
)


SAC_RUN_DIR = os.path.join(
    ROOT,
    "runs",
    "PandaReach-v3__sac_reach_joints__1__1789075275",
)

SAC_MODEL_PATH = os.path.join(
    SAC_RUN_DIR,
    "sac_reach_joints.actor.pt",
)

SAC_OBS_RMS_PATH = os.path.join(
    SAC_RUN_DIR,
    "sac_reach_joints.obs_rms.npz",
)


# ============================================================
# Fixed benchmark goals
#
# SAME offsets for PPO and SAC.
#
# Start conservatively at +/- 8 cm for first real benchmark.
# ============================================================

N_EPISODES = 20

GOAL_SEED = 12345


def make_fixed_goal_offsets():

    rng = np.random.default_rng(
        GOAL_SEED
    )

    offsets = []

    while len(offsets) < N_EPISODES:

        offset = rng.uniform(
            low=-0.08,
            high=0.08,
            size=3,
        )

        distance = np.linalg.norm(
            offset
        )

        # Avoid trivially close goals
        if distance < 0.04:
            continue

        offsets.append(offset)

    return np.asarray(
        offsets,
        dtype=np.float64,
    )


GOAL_OFFSETS = make_fixed_goal_offsets()


# ============================================================
# Minimal environment spec for constructing CleanRL networks
# ============================================================

class DummyEnvSpec:

    def __init__(self):

        self.single_observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(12,),
            dtype=np.float32,
        )

        self.single_action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(7,),
            dtype=np.float32,
        )


# ============================================================
# Observation preprocessing
# ============================================================

class ObservationNormalizer:

    def __init__(self, path):

        data = np.load(path)

        self.mean = np.asarray(
            data["mean"],
            dtype=np.float32,
        )

        self.var = np.asarray(
            data["var"],
            dtype=np.float32,
        )

        self.count = float(
            data["count"]
        )

        if self.mean.shape != (12,):
            raise RuntimeError(
                f"Expected obs mean shape (12,), "
                f"got {self.mean.shape}"
            )

        if self.var.shape != (12,):
            raise RuntimeError(
                f"Expected obs var shape (12,), "
                f"got {self.var.shape}"
            )

    def normalize(self, obs):

        obs = np.asarray(
            obs,
            dtype=np.float32,
        )

        obs = (
            obs - self.mean
        ) / np.sqrt(
            self.var + 1e-8
        )

        obs = np.clip(
            obs,
            -10.0,
            10.0,
        )

        return obs.astype(
            np.float32
        )


# ============================================================
# Build the same 12D flattened observation as training
#
# Gymnasium Dict flattening for the PandaReach dict gives:
#
# achieved_goal : 3
# desired_goal  : 3
# observation   : 6
#
# total = 12
# ============================================================

def build_flat_obs(
    current_pos_real,
    current_vel_real,
    target_real,
):

    current_sim = real_to_sim(
        current_pos_real
    )

    target_sim = real_to_sim(
        target_real
    )

    achieved_goal = current_sim.astype(
        np.float32
    )

    desired_goal = target_sim.astype(
        np.float32
    )

    observation = np.concatenate([
        current_sim,
        current_vel_real,
    ]).astype(
        np.float32
    )

    flat = np.concatenate([
        achieved_goal,
        desired_goal,
        observation,
    ])

    assert flat.shape == (12,)

    return flat


# ============================================================
# Policy wrappers
# ============================================================

class PPOPolicy:

    def __init__(
        self,
        model_path,
        obs_rms_path,
        device,
    ):

        self.device = device

        dummy_env = DummyEnvSpec()

        self.agent = PPOAgent(
            dummy_env
        ).to(device)

        state_dict = torch.load(
            model_path,
            map_location=device,
        )

        self.agent.load_state_dict(
            state_dict
        )

        self.agent.eval()

        self.normalizer = (
            ObservationNormalizer(
                obs_rms_path
            )
        )

        print(
            f"PPO loaded: {model_path}"
        )

    def predict(self, raw_obs):

        obs = self.normalizer.normalize(
            raw_obs
        )

        obs_t = torch.as_tensor(
            obs,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        with torch.no_grad():

            action = self.agent.actor_mean(
                obs_t
            )

        action = (
            action
            .cpu()
            .numpy()[0]
        )

        return np.clip(
            action,
            -1.0,
            1.0,
        )


class SACPolicy:

    def __init__(
        self,
        model_path,
        obs_rms_path,
        device,
    ):

        self.device = device

        dummy_env = DummyEnvSpec()

        self.actor = SACActor(
            dummy_env
        ).to(device)

        state_dict = torch.load(
            model_path,
            map_location=device,
        )

        self.actor.load_state_dict(
            state_dict
        )

        self.actor.eval()

        self.normalizer = (
            ObservationNormalizer(
                obs_rms_path
            )
        )

        print(
            f"SAC loaded: {model_path}"
        )

    def predict(self, raw_obs):

        obs = self.normalizer.normalize(
            raw_obs
        )

        obs_t = torch.as_tensor(
            obs,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        with torch.no_grad():

            mean, _ = self.actor(
                obs_t
            )

            action = (
                torch.tanh(mean)
                * self.actor.action_scale
                + self.actor.action_bias
            )

        action = (
            action
            .cpu()
            .numpy()[0]
        )

        return np.clip(
            action,
            -1.0,
            1.0,
        )


# ============================================================
# Real robot benchmark
# ============================================================

class FrankaReachJointBenchmark:

    def __init__(
        self,
        panda,
        policy,
        algo_name,
    ):

        self.panda = panda
        self.policy = policy
        self.algo_name = algo_name

        self.prev_pos = None

        self.target_real = None

        self.step_count = 0

        self.ep_return = 0.0

        self.minimum_distance = np.inf


    # ========================================================
    # Read robot state
    # ========================================================

    def get_q(self):

        state = self.panda.get_state()

        q = np.asarray(
            state.q,
            dtype=np.float64,
        )[:7]

        return q


    def get_position(self):

        return np.asarray(
            self.panda.get_position(),
            dtype=np.float64,
        )


    # ========================================================
    # TODO TOMORROW:
    # Fill in the actual panda_py joint controller.
    # ========================================================

    def _send_joint_target(
        self,
        q_target,
    ):

        raise NotImplementedError(
            "\nJoint realtime controller has not "
            "been connected yet.\n"
            "Tomorrow on Jetson run:\n\n"
            "from panda_py import controllers\n"
            "print(dir(controllers))\n\n"
            "Then implement only "
            "_send_joint_target()."
        )


    # ========================================================
    # Apply panda-gym-equivalent action
    # ========================================================

    def apply_action(
        self,
        action,
    ):

        action = np.asarray(
            action,
            dtype=np.float64,
        ).flatten()

        if action.shape != (7,):
            raise RuntimeError(
                f"Expected action shape (7,), "
                f"got {action.shape}"
            )

        action = np.clip(
            action,
            -1.0,
            1.0,
        )

        q_current = self.get_q()

        # EXACT panda-gym joint semantics:
        #
        # q_target = q_current + action * 0.05
        #
        delta_q = (
            action
            * JOINT_ACTION_SCALE
        )

        q_target = (
            q_current
            + delta_q
        )

        # Joint safety layer
        q_target = np.clip(
            q_target,
            SAFE_JOINT_LOW,
            SAFE_JOINT_HIGH,
        )

        self._send_joint_target(
            q_target
        )

        time.sleep(DT)


    # ========================================================
    # Reset
    # ========================================================

    def reset(
        self,
        goal_offset,
    ):

        print(
            "\nReturning robot to start..."
        )

        try:

            self.panda.move_to_start()

        except Exception:

            print(
                "move_to_start failed, recovering..."
            )

            self.panda.recover()

            self.panda.move_to_start()

        time.sleep(0.5)

        x0 = self.get_position()

        self.prev_pos = x0.copy()

        target = (
            x0
            + np.asarray(goal_offset)
        )

        target = np.clip(
            target,
            WORKSPACE_LOW,
            WORKSPACE_HIGH,
        )

        self.target_real = target

        self.step_count = 0

        self.ep_return = 0.0

        initial_distance = np.linalg.norm(
            x0 - target
        )

        self.minimum_distance = (
            initial_distance
        )

        return (
            x0,
            initial_distance,
        )


    # ========================================================
    # Run one benchmark episode
    # ========================================================

    def run_episode(
        self,
        goal_offset,
    ):

        (
            current_pos,
            initial_distance,
        ) = self.reset(
            goal_offset
        )

        print(
            f"Initial distance: "
            f"{initial_distance * 100:.2f} cm"
        )

        print(
            "Goal real:",
            np.round(
                self.target_real,
                4,
            ),
        )


        success = False


        for step in range(
            MAX_EP_STEPS
        ):

            current_pos = self.get_position()

            velocity = (
                current_pos
                - self.prev_pos
            ) * CONTROL_FREQ

            self.prev_pos = (
                current_pos.copy()
            )


            raw_obs = build_flat_obs(
                current_pos,
                velocity,
                self.target_real,
            )


            action = self.policy.predict(
                raw_obs
            )


            print(
                f"step={step + 1:02d} "
                f"action="
                f"{np.round(action, 3)}"
            )


            self.apply_action(
                action
            )


            new_pos = self.get_position()


            # Workspace emergency check
            if np.any(
                new_pos < WORKSPACE_LOW
            ) or np.any(
                new_pos > WORKSPACE_HIGH
            ):

                raise RuntimeError(
                    "EE left safe workspace."
                )


            distance = np.linalg.norm(
                new_pos
                - self.target_real
            )


            reward = -float(
                distance
            )


            self.ep_return += (
                reward
            )


            self.minimum_distance = min(
                self.minimum_distance,
                distance,
            )


            self.step_count += 1


            print(
                f"      distance="
                f"{distance * 100:.2f} cm"
            )


            if distance < GOAL_THRESHOLD:

                success = True

                break


        final_pos = self.get_position()

        final_distance = np.linalg.norm(
            final_pos
            - self.target_real
        )


        return {
            "success": int(success),

            "initial_distance_cm":
                initial_distance * 100,

            "final_distance_cm":
                final_distance * 100,

            "minimum_distance_cm":
                self.minimum_distance * 100,

            "steps":
                self.step_count,

            "return":
                self.ep_return,
        }


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--algo",
        choices=[
            "ppo",
            "sac",
        ],
        required=True,
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=20,
    )

    args = parser.parse_args()


    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


    # ========================================================
    # Load policy
    # ========================================================

    if args.algo == "ppo":

        policy = PPOPolicy(
            PPO_MODEL_PATH,
            PPO_OBS_RMS_PATH,
            device,
        )

    else:

        policy = SACPolicy(
            SAC_MODEL_PATH,
            SAC_OBS_RMS_PATH,
            device,
        )


    # ========================================================
    # Robot
    # ========================================================

    print(
        f"Connecting to {HOSTNAME}"
    )

    panda = panda_py.Panda(
        HOSTNAME
    )

    print(
        "Robot connected."
    )


    benchmark = FrankaReachJointBenchmark(
        panda,
        policy,
        args.algo,
    )


    # ========================================================
    # Results
    # ========================================================

    os.makedirs(
        os.path.join(
            ROOT,
            "results",
        ),
        exist_ok=True,
    )


    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )


    csv_path = os.path.join(
        ROOT,
        "results",
        f"real_{args.algo}_joint_"
        f"reach_benchmark_{timestamp}.csv",
    )


    results = []


    print(
        "=" * 70
    )

    print(
        f"REAL ROBOT "
        f"{args.algo.upper()} "
        f"JOINT REACH BENCHMARK"
    )

    print(
        f"Threshold: "
        f"{GOAL_THRESHOLD * 100:.1f} cm"
    )

    print(
        f"Joint scale: "
        f"{JOINT_ACTION_SCALE} rad"
    )

    print(
        f"Episodes: "
        f"{args.episodes}"
    )

    print(
        "=" * 70
    )


    input(
        "\n检查机器人、workspace和急停。"
        "Enter 继续："
    )


    try:

        for episode in range(
            args.episodes
        ):

            print(
                "\n"
                + "=" * 70
            )

            print(
                f"Episode "
                f"{episode + 1}/"
                f"{args.episodes}"
            )

            print(
                "=" * 70
            )


            result = benchmark.run_episode(
                GOAL_OFFSETS[
                    episode
                    % len(GOAL_OFFSETS)
                ]
            )


            result["episode"] = (
                episode + 1
            )


            result["algo"] = (
                args.algo
            )


            results.append(
                result
            )


            print(
                f"\n"
                f"{'SUCCESS' if result['success'] else 'FAIL'}"
                f" | final="
                f"{result['final_distance_cm']:.2f} cm"
                f" | min="
                f"{result['minimum_distance_cm']:.2f} cm"
                f" | steps="
                f"{result['steps']}"
            )


    except KeyboardInterrupt:

        print(
            "\nBenchmark interrupted."
        )


    finally:

        try:
            panda.stop_controller()
        except Exception:
            pass

        try:
            panda.move_to_start()
        except Exception:
            pass


    # ========================================================
    # Save
    # ========================================================

    if len(results) == 0:

        print(
            "No completed episodes."
        )

        return


    fieldnames = [
        "episode",
        "algo",
        "success",
        "initial_distance_cm",
        "final_distance_cm",
        "minimum_distance_cm",
        "steps",
        "return",
    ]


    with open(
        csv_path,
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            results
        )


    success = np.array([
        r["success"]
        for r in results
    ])


    final_errors = np.array([
        r["final_distance_cm"]
        for r in results
    ])


    min_errors = np.array([
        r["minimum_distance_cm"]
        for r in results
    ])


    steps = np.array([
        r["steps"]
        for r in results
    ])


    print(
        "\n"
        + "=" * 70
    )

    print(
        "BENCHMARK SUMMARY"
    )

    print(
        "=" * 70
    )

    print(
        f"Algorithm:                 "
        f"{args.algo.upper()}"
    )

    print(
        f"Episodes:                  "
        f"{len(results)}"
    )

    print(
        f"Success threshold:         "
        f"{GOAL_THRESHOLD * 100:.1f} cm"
    )

    print(
        f"Success rate:              "
        f"{success.mean() * 100:.1f}%"
    )

    print(
        f"Mean final error:          "
        f"{final_errors.mean():.2f} cm"
    )

    print(
        f"Median final error:        "
        f"{np.median(final_errors):.2f} cm"
    )

    print(
        f"Mean minimum error:        "
        f"{min_errors.mean():.2f} cm"
    )

    print(
        f"Mean steps:                "
        f"{steps.mean():.1f}"
    )

    print(
        f"Median steps:              "
        f"{np.median(steps):.1f}"
    )

    print(
        "=" * 70
    )

    print(
        f"CSV: {csv_path}"
    )


if __name__ == "__main__":
    main()