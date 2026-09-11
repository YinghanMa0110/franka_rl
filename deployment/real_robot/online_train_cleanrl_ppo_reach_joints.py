"""
Online PPO fine-tuning for Franka Panda Reach task.

Starts from the simulation-trained CleanRL PPO checkpoint and
continues training using real-robot transitions.

Real environment:
    deployment/real_robot/reach_joint_env.py

Simulation policy:
    PandaReach-v3
    dense reward
    control_type="joints"
    success threshold = 0.02 m

Observation:
    12D flattened PandaReach observation

Action:
    7D Gaussian PPO action

The real environment reproduces panda-gym semantics:
    action -> clip to [-1, 1]
    delta_q = action * 0.05 rad
    q_target = q_current + delta_q
"""

import argparse
import os
import random
import sys
import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ============================================================
# Repository paths
# ============================================================

ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "../..",
    )
)

if ROOT not in sys.path:
    sys.path.insert(
        0,
        ROOT,
    )

REAL_ROBOT_DIR = os.path.dirname(
    __file__
)

if REAL_ROBOT_DIR not in sys.path:
    sys.path.insert(
        0,
        REAL_ROBOT_DIR,
    )


# ============================================================
# Project imports
# ============================================================

from training.cleanrl.ppo_continuous_action import Agent
from reach_joint_env import FrankaReachJointEnv


# panda_py is only expected on the real-robot machine.
try:
    import panda_py
except ImportError:
    panda_py = None


# ============================================================
# Simulation checkpoint
# ============================================================

PPO_RUN = os.path.join(
    ROOT,
    "runs",
    "PandaReach-v3__ppo_continuous_action__1__1789070756",
)

MODEL_PATH = os.path.join(
    PPO_RUN,
    "ppo_continuous_action.cleanrl_model",
)

OBS_RMS_PATH = os.path.join(
    PPO_RUN,
    "ppo_continuous_action.obs_rms.npz",
)


# ============================================================
# Default real-robot configuration
# ============================================================

# Change from CLI if Jetson/robot network uses another address.
DEFAULT_HOSTNAME = "192.168.1.8"

DEFAULT_TOTAL_REAL_STEPS = 1000

# Real transitions collected before each PPO update.
DEFAULT_ROLLOUT_STEPS = 200

# Save checkpoints at these interaction intervals.
DEFAULT_SAVE_EVERY = 200


# ============================================================
# PPO fine-tuning hyperparameters
# ============================================================

# Lower than simulation training because this is fine-tuning
# an already-trained policy on a real robot.
LEARNING_RATE = 1e-4

GAMMA = 0.99
GAE_LAMBDA = 0.95

# Keep PPO objective parameters aligned with simulation.
CLIP_COEF = 0.2

ENT_COEF = 0.01
VF_COEF = 0.5

UPDATE_EPOCHS = 10

MAX_GRAD_NORM = 0.5

NORM_ADV = True
CLIP_VLOSS = True

# 200 rollout transitions -> 4 minibatches of 50.
MINIBATCH_SIZE = 50


# ============================================================
# Experiment
# ============================================================

SEED = 1


# ============================================================
# Frozen simulation observation normalization
# ============================================================

class FrozenObsNormalizer:
    """
    Reproduce the NormalizeObservation statistics used during
    simulation training.

    We intentionally freeze these statistics during the first
    real-robot online-learning experiment so PPO and SAC do not
    introduce an additional normalization adaptation variable.
    """

    def __init__(
        self,
        path,
    ):
        data = np.load(
            path
        )

        self.mean = np.asarray(
            data["mean"],
            dtype=np.float32,
        )

        self.var = np.asarray(
            data["var"],
            dtype=np.float32,
        )

        if "count" in data:
            self.count = float(
                np.asarray(
                    data["count"]
                )
            )
        else:
            self.count = None

        self.epsilon = 1e-8

        if self.mean.shape != (12,):
            raise ValueError(
                "Expected observation mean shape (12,), "
                f"got {self.mean.shape}"
            )

        if self.var.shape != (12,):
            raise ValueError(
                "Expected observation var shape (12,), "
                f"got {self.var.shape}"
            )

    def normalize(
        self,
        obs,
    ):
        obs = np.asarray(
            obs,
            dtype=np.float32,
        )

        if obs.shape != (12,):
            raise ValueError(
                "Expected raw observation shape (12,), "
                f"got {obs.shape}"
            )

        normalized = (
            obs - self.mean
        ) / np.sqrt(
            self.var
            + self.epsilon
        )

        # Same TransformObservation used in simulation.
        normalized = np.clip(
            normalized,
            -10.0,
            10.0,
        )

        return normalized.astype(
            np.float32
        )


# ============================================================
# Dummy vector environment interface
#
# CleanRL Agent expects:
#     envs.single_observation_space
#     envs.single_action_space
#
# We only need these spaces to reconstruct the network.
# ============================================================

class DummyVectorEnv:

    def __init__(self):

        self.single_observation_space = (
            gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(12,),
                dtype=np.float32,
            )
        )

        self.single_action_space = (
            gym.spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(7,),
                dtype=np.float32,
            )
        )


# ============================================================
# Agent loading
# ============================================================

def load_agent(
    model_path,
    device,
):
    dummy_env = DummyVectorEnv()

    agent = Agent(
        dummy_env
    ).to(
        device
    )

    state_dict = torch.load(
        model_path,
        map_location=device,
    )

    # Normal simulation checkpoint is a plain state_dict.
    # Also support our online training checkpoint format.
    if (
        isinstance(
            state_dict,
            dict,
        )
        and "agent_state_dict" in state_dict
    ):
        state_dict = state_dict[
            "agent_state_dict"
        ]

    agent.load_state_dict(
        state_dict
    )

    return agent


# ============================================================
# Checkpoint saving
# ============================================================

def save_checkpoint(
    agent,
    optimizer,
    real_steps,
    output_dir,
):
    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Plain CleanRL-compatible model.
    #
    # Benchmark scripts can load this directly with
    # agent.load_state_dict(torch.load(...)).
    # --------------------------------------------------------

    model_path = os.path.join(
        output_dir,
        (
            "ppo_online_step_"
            f"{real_steps:06d}"
            ".cleanrl_model"
        ),
    )

    torch.save(
        agent.state_dict(),
        model_path,
    )

    # --------------------------------------------------------
    # Full training state.
    # --------------------------------------------------------

    training_path = os.path.join(
        output_dir,
        (
            "ppo_online_step_"
            f"{real_steps:06d}"
            ".training.pt"
        ),
    )

    torch.save(
        {
            "real_steps":
                real_steps,

            "agent_state_dict":
                agent.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "learning_rate":
                LEARNING_RATE,

            "gamma":
                GAMMA,

            "gae_lambda":
                GAE_LAMBDA,

            "clip_coef":
                CLIP_COEF,

            "ent_coef":
                ENT_COEF,

            "vf_coef":
                VF_COEF,
        },
        training_path,
    )

    print(
        "\nCheckpoint saved:"
    )

    print(
        f"  model:    {model_path}"
    )

    print(
        f"  training: {training_path}"
    )

    return model_path


# ============================================================
# PPO update
# ============================================================

def ppo_update(
    agent,
    optimizer,
    b_obs,
    b_actions,
    b_logprobs,
    b_advantages,
    b_returns,
    b_values,
):
    """
    PPO update kept close to the simulation CleanRL implementation.
    """

    batch_size = (
        b_obs.shape[0]
    )

    if batch_size == 0:
        raise ValueError(
            "Empty PPO rollout."
        )

    b_inds = np.arange(
        batch_size
    )

    clipfracs = []

    last_pg_loss = np.nan
    last_v_loss = np.nan
    last_entropy = np.nan
    last_approx_kl = np.nan

    for epoch in range(
        UPDATE_EPOCHS
    ):
        np.random.shuffle(
            b_inds
        )

        for start in range(
            0,
            batch_size,
            MINIBATCH_SIZE,
        ):
            end = min(
                start
                + MINIBATCH_SIZE,
                batch_size,
            )

            mb_inds = b_inds[
                start:end
            ]

            if len(mb_inds) == 0:
                continue

            (
                _,
                newlogprob,
                entropy,
                newvalue,
            ) = agent.get_action_and_value(
                b_obs[
                    mb_inds
                ],
                b_actions[
                    mb_inds
                ],
            )

            # =================================================
            # PPO probability ratio
            # =================================================

            logratio = (
                newlogprob
                - b_logprobs[
                    mb_inds
                ]
            )

            ratio = logratio.exp()

            with torch.no_grad():

                old_approx_kl = (
                    -logratio
                ).mean()

                approx_kl = (
                    (
                        ratio - 1
                    )
                    - logratio
                ).mean()

                clipfrac = (
                    (
                        (
                            ratio - 1.0
                        ).abs()
                        > CLIP_COEF
                    )
                    .float()
                    .mean()
                    .item()
                )

                clipfracs.append(
                    clipfrac
                )

            # =================================================
            # Advantage normalization
            # =================================================

            mb_advantages = (
                b_advantages[
                    mb_inds
                ]
            )

            if (
                NORM_ADV
                and len(mb_inds) > 1
            ):
                mb_advantages = (
                    mb_advantages
                    - mb_advantages.mean()
                ) / (
                    mb_advantages.std(
                        unbiased=False
                    )
                    + 1e-8
                )

            # =================================================
            # Policy loss
            # =================================================

            pg_loss1 = (
                -mb_advantages
                * ratio
            )

            pg_loss2 = (
                -mb_advantages
                * torch.clamp(
                    ratio,
                    1.0
                    - CLIP_COEF,
                    1.0
                    + CLIP_COEF,
                )
            )

            pg_loss = torch.max(
                pg_loss1,
                pg_loss2,
            ).mean()

            # =================================================
            # Value loss
            # =================================================

            newvalue = (
                newvalue.view(
                    -1
                )
            )

            if CLIP_VLOSS:

                v_loss_unclipped = (
                    newvalue
                    - b_returns[
                        mb_inds
                    ]
                ) ** 2

                v_clipped = (
                    b_values[
                        mb_inds
                    ]
                    + torch.clamp(
                        newvalue
                        - b_values[
                            mb_inds
                        ],
                        -CLIP_COEF,
                        CLIP_COEF,
                    )
                )

                v_loss_clipped = (
                    v_clipped
                    - b_returns[
                        mb_inds
                    ]
                ) ** 2

                v_loss_max = (
                    torch.max(
                        v_loss_unclipped,
                        v_loss_clipped,
                    )
                )

                v_loss = (
                    0.5
                    * v_loss_max.mean()
                )

            else:

                v_loss = (
                    0.5
                    * (
                        (
                            newvalue
                            - b_returns[
                                mb_inds
                            ]
                        )
                        ** 2
                    ).mean()
                )

            # =================================================
            # Entropy / total loss
            # =================================================

            entropy_loss = (
                entropy.mean()
            )

            loss = (
                pg_loss
                - ENT_COEF
                * entropy_loss
                + VF_COEF
                * v_loss
            )

            optimizer.zero_grad()

            loss.backward()

            nn.utils.clip_grad_norm_(
                agent.parameters(),
                MAX_GRAD_NORM,
            )

            optimizer.step()

            last_pg_loss = (
                pg_loss.item()
            )

            last_v_loss = (
                v_loss.item()
            )

            last_entropy = (
                entropy_loss.item()
            )

            last_approx_kl = (
                approx_kl.item()
            )

    return {
        "policy_loss":
            float(
                last_pg_loss
            ),

        "value_loss":
            float(
                last_v_loss
            ),

        "entropy":
            float(
                last_entropy
            ),

        "approx_kl":
            float(
                last_approx_kl
            ),

        "clipfrac":
            float(
                np.mean(
                    clipfracs
                )
            )
            if clipfracs
            else np.nan,
    }


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Online PPO fine-tuning "
            "for Franka Panda Reach."
        )
    )

    parser.add_argument(
        "--hostname",
        type=str,
        default=DEFAULT_HOSTNAME,
    )

    parser.add_argument(
        "--model-path",
        type=str,
        default=MODEL_PATH,
    )

    parser.add_argument(
        "--obs-rms-path",
        type=str,
        default=OBS_RMS_PATH,
    )

    parser.add_argument(
        "--total-real-steps",
        type=int,
        default=DEFAULT_TOTAL_REAL_STEPS,
    )

    parser.add_argument(
        "--rollout-steps",
        type=int,
        default=DEFAULT_ROLLOUT_STEPS,
    )

    parser.add_argument(
        "--save-every",
        type=int,
        default=DEFAULT_SAVE_EVERY,
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(
            ROOT,
            "checkpoints",
            "online_ppo",
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
    )

    args = parser.parse_args()

    # ========================================================
    # Validation
    # ========================================================

    if panda_py is None:
        raise ImportError(
            "panda_py is not installed in this "
            "Python environment. Run this script "
            "on the Franka/Jetson machine."
        )

    if not os.path.exists(
        args.model_path
    ):
        raise FileNotFoundError(
            "PPO model not found:\n"
            f"{args.model_path}"
        )

    if not os.path.exists(
        args.obs_rms_path
    ):
        raise FileNotFoundError(
            "Observation statistics not found:\n"
            f"{args.obs_rms_path}"
        )

    if args.rollout_steps <= 0:
        raise ValueError(
            "--rollout-steps must be > 0"
        )

    if args.total_real_steps <= 0:
        raise ValueError(
            "--total-real-steps must be > 0"
        )

    # ========================================================
    # Seeds
    # ========================================================

    random.seed(
        args.seed
    )

    np.random.seed(
        args.seed
    )

    torch.manual_seed(
        args.seed
    )

    # ========================================================
    # Device
    # ========================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device: {device}"
    )

    # ========================================================
    # Load normalization + PPO
    # ========================================================

    normalizer = FrozenObsNormalizer(
        args.obs_rms_path
    )

    agent = load_agent(
        args.model_path,
        device,
    )

    optimizer = optim.Adam(
        agent.parameters(),
        lr=LEARNING_RATE,
        eps=1e-5,
    )

    # ========================================================
    # Connect robot
    # ========================================================

    print(
        f"Connecting to Franka: "
        f"{args.hostname}"
    )

    panda = panda_py.Panda(
        args.hostname
    )

    env = FrankaReachJointEnv(
        panda,
        goal_threshold=0.02,
        max_ep_steps=50,
    )

    # ========================================================
    # Experiment information
    # ========================================================

    print(
        "\n"
        + "=" * 72
    )

    print(
        "REAL ROBOT PPO ONLINE FINE-TUNING"
    )

    print(
        "=" * 72
    )

    print(
        f"Simulation model:\n"
        f"  {args.model_path}"
    )

    print(
        f"Observation RMS:\n"
        f"  {args.obs_rms_path}"
    )

    print(
        f"Total real interactions: "
        f"{args.total_real_steps}"
    )

    print(
        f"Rollout size: "
        f"{args.rollout_steps}"
    )

    print(
        f"Learning rate: "
        f"{LEARNING_RATE}"
    )

    print(
        f"PPO clip coefficient: "
        f"{CLIP_COEF}"
    )

    print(
        f"GAE lambda: "
        f"{GAE_LAMBDA}"
    )

    print(
        f"Success threshold: "
        f"{env.goal_threshold * 100:.1f} cm"
    )

    print(
        f"Joint action scale: "
        f"{env.joint_action_scale} rad"
    )

    print(
        "=" * 72
    )

    input(
        "\nCheck robot, workspace, controller and E-stop. "
        "Press Enter to start online training: "
    )

    # ========================================================
    # Initial episode
    # ========================================================

    obs_raw, reset_info = (
        env.reset(
            seed=args.seed
        )
    )

    obs = normalizer.normalize(
        obs_raw
    )

    # Matches CleanRL's next_done.
    next_done = 0.0

    real_steps = 0

    episode_number = 1

    start_time = time.time()

    next_save_step = (
        args.save_every
    )

    print(
        f"\nEpisode {episode_number}"
        f" | initial distance="
        f"{reset_info['initial_distance'] * 100:.2f} cm"
    )

    # ========================================================
    # Online training
    # ========================================================

    try:

        while (
            real_steps
            < args.total_real_steps
        ):

            rollout_obs = []

            rollout_actions = []

            rollout_logprobs = []

            rollout_rewards = []

            rollout_dones = []

            rollout_values = []

            # =================================================
            # Collect a real-robot rollout
            # =================================================

            for _ in range(
                args.rollout_steps
            ):

                if (
                    real_steps
                    >= args.total_real_steps
                ):
                    break

                # ---------------------------------------------
                # Store observation and previous done flag.
                #
                # This matches:
                #
                # obs[step] = next_obs
                # dones[step] = next_done
                #
                # in the original CleanRL implementation.
                # ---------------------------------------------

                rollout_obs.append(
                    obs.copy()
                )

                rollout_dones.append(
                    float(
                        next_done
                    )
                )

                obs_tensor = torch.as_tensor(
                    obs,
                    dtype=torch.float32,
                    device=device,
                ).unsqueeze(
                    0
                )

                # ---------------------------------------------
                # Sample PPO action.
                # ---------------------------------------------

                with torch.no_grad():

                    (
                        action,
                        logprob,
                        _,
                        value,
                    ) = agent.get_action_and_value(
                        obs_tensor
                    )

                # IMPORTANT:
                #
                # Store the ORIGINAL sampled Gaussian action.
                #
                # In simulation, ClipAction clips it only when
                # sending it into panda-gym. The PPO logprob is
                # associated with the un-clipped sampled action.
                #
                # reach_joint_env.step() performs the same
                # [-1,1] clipping before converting it to dq.
                #
                action_np = (
                    action
                    .cpu()
                    .numpy()[0]
                    .astype(
                        np.float32
                    )
                )

                rollout_actions.append(
                    action_np.copy()
                )

                rollout_logprobs.append(
                    float(
                        logprob.item()
                    )
                )

                rollout_values.append(
                    float(
                        value.item()
                    )
                )

                # ---------------------------------------------
                # Execute on real robot.
                # ---------------------------------------------

                (
                    next_obs_raw,
                    reward,
                    terminated,
                    truncated,
                    step_info,
                ) = env.step(
                    action_np
                )

                next_done = float(
                    terminated
                    or truncated
                )

                rollout_rewards.append(
                    float(
                        reward
                    )
                )

                real_steps += 1

                # Actual action executed by the environment.
                executed_action = np.clip(
                    action_np,
                    -1.0,
                    1.0,
                )

                clipped = bool(
                    np.any(
                        np.abs(
                            action_np
                        )
                        > 1.0
                    )
                )

                print(
                    f"real_step={real_steps:05d}"
                    f" | ep={episode_number:03d}"
                    f" | ep_step="
                    f"{step_info['step_count']:02d}"
                    f" | distance="
                    f"{step_info['distance_cm']:.2f} cm"
                    f" | reward="
                    f"{reward:.4f}"
                    f" | success="
                    f"{int(step_info['is_success'])}"
                    f" | clipped="
                    f"{int(clipped)}"
                )

                if clipped:

                    print(
                        "  sampled="
                        f"{np.round(action_np, 3)}"
                    )

                    print(
                        "  executed="
                        f"{np.round(executed_action, 3)}"
                    )

                # ---------------------------------------------
                # Episode ended.
                # ---------------------------------------------

                if next_done:

                    print(
                        "\nEpisode finished"
                    )

                    print(
                        f"  episode: "
                        f"{episode_number}"
                    )

                    print(
                        f"  success: "
                        f"{bool(step_info['is_success'])}"
                    )

                    print(
                        f"  steps: "
                        f"{step_info['step_count']}"
                    )

                    print(
                        f"  final error: "
                        f"{step_info['distance_cm']:.2f} cm"
                    )

                    print(
                        f"  minimum error: "
                        f"{step_info['minimum_distance_cm']:.2f} cm"
                    )

                    print(
                        f"  return: "
                        f"{step_info['episode_return']:.4f}"
                    )

                    if (
                        step_info.get(
                            "safety_reason"
                        )
                        not in (
                            None,
                            "time_limit",
                        )
                    ):
                        print(
                            "  SAFETY EVENT: "
                            f"{step_info['safety_reason']}"
                        )

                    episode_number += 1

                    # Reset creates a new goal and returns the
                    # robot to the start configuration.
                    next_obs_raw, reset_info = (
                        env.reset()
                    )

                    print(
                        f"\nEpisode "
                        f"{episode_number}"
                        f" | initial distance="
                        f"{reset_info['initial_distance'] * 100:.2f} cm"
                    )

                # Whether this was an ordinary transition or
                # an auto/manual reset transition, obs now
                # points to the next state used by PPO.
                obs = normalizer.normalize(
                    next_obs_raw
                )

            # =================================================
            # Convert rollout to tensors
            # =================================================

            if len(
                rollout_rewards
            ) == 0:
                break

            b_obs = torch.as_tensor(
                np.asarray(
                    rollout_obs,
                    dtype=np.float32,
                ),
                dtype=torch.float32,
                device=device,
            )

            b_actions = torch.as_tensor(
                np.asarray(
                    rollout_actions,
                    dtype=np.float32,
                ),
                dtype=torch.float32,
                device=device,
            )

            b_logprobs = torch.as_tensor(
                np.asarray(
                    rollout_logprobs,
                    dtype=np.float32,
                ),
                dtype=torch.float32,
                device=device,
            )

            b_rewards_np = np.asarray(
                rollout_rewards,
                dtype=np.float32,
            )

            b_dones_np = np.asarray(
                rollout_dones,
                dtype=np.float32,
            )

            b_values_np = np.asarray(
                rollout_values,
                dtype=np.float32,
            )

            # =================================================
            # Value bootstrap
            # =================================================

            obs_tensor = torch.as_tensor(
                obs,
                dtype=torch.float32,
                device=device,
            ).unsqueeze(
                0
            )

            with torch.no_grad():

                next_value = (
                    agent.get_value(
                        obs_tensor
                    )
                    .cpu()
                    .numpy()
                    .reshape(
                        -1
                    )[0]
                )

            # =================================================
            # GAE
            #
            # This follows your original CleanRL PPO:
            #
            # if final step:
            #     nextnonterminal = 1 - next_done
            # else:
            #     nextnonterminal = 1 - dones[t+1]
            # =================================================

            advantages = np.zeros_like(
                b_rewards_np,
                dtype=np.float32,
            )

            lastgaelam = 0.0

            for t in reversed(
                range(
                    len(
                        b_rewards_np
                    )
                )
            ):

                if (
                    t
                    == len(
                        b_rewards_np
                    )
                    - 1
                ):

                    nextnonterminal = (
                        1.0
                        - float(
                            next_done
                        )
                    )

                    nextvalues = (
                        float(
                            next_value
                        )
                    )

                else:

                    nextnonterminal = (
                        1.0
                        - b_dones_np[
                            t + 1
                        ]
                    )

                    nextvalues = (
                        b_values_np[
                            t + 1
                        ]
                    )

                delta = (
                    b_rewards_np[t]
                    + GAMMA
                    * nextvalues
                    * nextnonterminal
                    - b_values_np[t]
                )

                lastgaelam = (
                    delta
                    + GAMMA
                    * GAE_LAMBDA
                    * nextnonterminal
                    * lastgaelam
                )

                advantages[t] = (
                    lastgaelam
                )

            returns = (
                advantages
                + b_values_np
            )

            b_advantages = torch.as_tensor(
                advantages,
                dtype=torch.float32,
                device=device,
            )

            b_returns = torch.as_tensor(
                returns,
                dtype=torch.float32,
                device=device,
            )

            b_values = torch.as_tensor(
                b_values_np,
                dtype=torch.float32,
                device=device,
            )

            # =================================================
            # PPO update
            # =================================================

            metrics = ppo_update(
                agent=agent,
                optimizer=optimizer,
                b_obs=b_obs,
                b_actions=b_actions,
                b_logprobs=b_logprobs,
                b_advantages=b_advantages,
                b_returns=b_returns,
                b_values=b_values,
            )

            elapsed = (
                time.time()
                - start_time
            )

            print(
                "\n"
                + "-" * 72
            )

            print(
                "PPO UPDATE"
            )

            print(
                "-" * 72
            )

            print(
                f"real interactions: "
                f"{real_steps}"
            )

            print(
                f"rollout samples: "
                f"{len(rollout_rewards)}"
            )

            print(
                f"policy loss: "
                f"{metrics['policy_loss']:.6f}"
            )

            print(
                f"value loss: "
                f"{metrics['value_loss']:.6f}"
            )

            print(
                f"entropy: "
                f"{metrics['entropy']:.6f}"
            )

            print(
                f"approx KL: "
                f"{metrics['approx_kl']:.6f}"
            )

            print(
                f"clip fraction: "
                f"{metrics['clipfrac']:.4f}"
            )

            print(
                f"elapsed: "
                f"{elapsed:.1f} s"
            )

            print(
                "-" * 72
            )

            # =================================================
            # Save checkpoint
            # =================================================

            while (
                real_steps
                >= next_save_step
            ):

                save_checkpoint(
                    agent=agent,
                    optimizer=optimizer,
                    real_steps=real_steps,
                    output_dir=args.output_dir,
                )

                next_save_step += (
                    args.save_every
                )

    except KeyboardInterrupt:

        print(
            "\n\nOnline PPO training interrupted "
            "by user."
        )

    finally:

        print(
            "\nSaving final PPO state..."
        )

        try:

            save_checkpoint(
                agent=agent,
                optimizer=optimizer,
                real_steps=real_steps,
                output_dir=args.output_dir,
            )

        except Exception as e:

            print(
                f"[save] warning: {e}"
            )

        print(
            "\nClosing real-robot environment..."
        )

        try:

            env.close()

        except Exception as e:

            print(
                f"[close] warning: {e}"
            )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()