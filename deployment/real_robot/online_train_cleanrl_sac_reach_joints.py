"""
Online SAC fine-tuning on the real Franka Panda.

Experiment design
-----------------
Warm start from simulation:
    - Actor
    - QF1
    - QF2
    - frozen simulation observation normalization

Reinitialize for real-world adaptation:
    - actor optimizer
    - Q optimizer
    - alpha optimizer
    - target Q networks are copied from Q1/Q2
    - replay buffer starts EMPTY and contains REAL data only

This gives a clean comparison against PPO:
both algorithms inherit learned simulation networks, but begin
real-world adaptation with zero real-world training history.

Task
----
PandaReach-v3 equivalent
reward_type = dense
control_type = joints
success threshold = 0.02 m

Action semantics:
    action in [-1, 1]^7
    delta_q = 0.05 * action
    q_target = q_current + delta_q
"""

import argparse
import math
import os
import random
import sys
import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim


# ============================================================
# Paths
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

CLEANRL_DIR = os.path.join(
    ROOT,
    "training",
    "cleanrl",
)

if CLEANRL_DIR not in sys.path:
    sys.path.insert(
        0,
        CLEANRL_DIR,
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

from training.cleanrl.sac_reach_joints import (
    Actor,
    SoftQNetwork,
)

from training.cleanrl.replay_buffer import (
    ReplayBuffer,
)

from reach_joint_env import (
    FrankaReachJointEnv,
)


try:
    import panda_py
except ImportError:
    panda_py = None


# ============================================================
# Simulation checkpoint
# ============================================================

SAC_RUN = os.path.join(
    ROOT,
    "runs",
    "PandaReach-v3__sac_reach_joints__1__1789075275",
)

ACTOR_PATH = os.path.join(
    SAC_RUN,
    "sac_reach_joints.actor.pt",
)

QF1_PATH = os.path.join(
    SAC_RUN,
    "sac_reach_joints.qf1.pt",
)

QF2_PATH = os.path.join(
    SAC_RUN,
    "sac_reach_joints.qf2.pt",
)

OBS_RMS_PATH = os.path.join(
    SAC_RUN,
    "sac_reach_joints.obs_rms.npz",
)


# ============================================================
# Real robot defaults
# ============================================================

DEFAULT_HOSTNAME = "192.168.1.8"

DEFAULT_TOTAL_REAL_STEPS = 1000

DEFAULT_SAVE_EVERY = 200


# ============================================================
# SAC hyperparameters
# ============================================================

GAMMA = 0.99

TAU = 0.005

BATCH_SIZE = 256

BUFFER_SIZE = 20_000

POLICY_LR = 3e-4

Q_LR = 1e-3

POLICY_FREQUENCY = 2

TARGET_NETWORK_FREQUENCY = 1

UPDATES_PER_STEP = 1


# ------------------------------------------------------------
# No simulation-style random learning_starts.
#
# Robot is controlled by the pretrained SAC actor immediately.
#
# We only delay gradient updates until enough real transitions
# have entered the replay buffer.
# ------------------------------------------------------------

DEFAULT_UPDATE_AFTER = 256


# ============================================================
# Entropy tuning
# ============================================================

AUTOTUNE = True

# Simulation final alpha was not saved.
# Start from a documented value and continue autotuning.
INITIAL_ALPHA = 0.2


# ============================================================
# Seed
# ============================================================

DEFAULT_SEED = 1


# ============================================================
# Observation normalization
# ============================================================

class FrozenObsNormalizer:

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
                "Expected obs mean shape (12,), "
                f"got {self.mean.shape}"
            )


        if self.var.shape != (12,):
            raise ValueError(
                "Expected obs var shape (12,), "
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
                "Expected raw obs shape (12,), "
                f"got {obs.shape}"
            )


        obs = (
            obs - self.mean
        ) / np.sqrt(
            self.var
            + self.epsilon
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
# Dummy vector-env interface
#
# Actor and SoftQNetwork expect:
#     single_observation_space
#     single_action_space
# ============================================================

class DummyVectorEnv:

    def __init__(
        self,
    ):

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
# Load pretrained simulation networks
# ============================================================

def load_networks(
    actor_path,
    qf1_path,
    qf2_path,
    device,
):

    dummy_env = DummyVectorEnv()


    actor = Actor(
        dummy_env
    ).to(
        device
    )


    qf1 = SoftQNetwork(
        dummy_env
    ).to(
        device
    )


    qf2 = SoftQNetwork(
        dummy_env
    ).to(
        device
    )


    actor.load_state_dict(
        torch.load(
            actor_path,
            map_location=device,
        )
    )


    qf1.load_state_dict(
        torch.load(
            qf1_path,
            map_location=device,
        )
    )


    qf2.load_state_dict(
        torch.load(
            qf2_path,
            map_location=device,
        )
    )


    # --------------------------------------------------------
    # Target Q networks
    #
    # Simulation target Q checkpoints were not saved.
    # Initialize them from the trained online critics.
    # --------------------------------------------------------

    qf1_target = SoftQNetwork(
        dummy_env
    ).to(
        device
    )


    qf2_target = SoftQNetwork(
        dummy_env
    ).to(
        device
    )


    qf1_target.load_state_dict(
        qf1.state_dict()
    )


    qf2_target.load_state_dict(
        qf2.state_dict()
    )


    qf1_target.eval()
    qf2_target.eval()


    return (
        actor,
        qf1,
        qf2,
        qf1_target,
        qf2_target,
        dummy_env,
    )


# ============================================================
# Replay buffer size helper
# ============================================================

def get_replay_size(
    rb,
):

    if hasattr(
        rb,
        "size",
    ):

        size_attr = rb.size

        if callable(
            size_attr
        ):
            return int(
                size_attr()
            )

        return int(
            size_attr
        )


    if hasattr(
        rb,
        "pos",
    ):

        if getattr(
            rb,
            "full",
            False,
        ):
            return int(
                rb.buffer_size
            )

        return int(
            rb.pos
        )


    raise AttributeError(
        "Unable to determine replay-buffer size."
    )


# ============================================================
# Target-network update
# ============================================================

def soft_update(
    source,
    target,
    tau,
):

    with torch.no_grad():

        for (
            param,
            target_param,
        ) in zip(
            source.parameters(),
            target.parameters(),
        ):

            target_param.data.copy_(
                tau
                * param.data
                + (
                    1.0
                    - tau
                )
                * target_param.data
            )


# ============================================================
# Save checkpoint
# ============================================================

def save_checkpoint(
    *,
    actor,
    qf1,
    qf2,
    qf1_target,
    qf2_target,
    actor_optimizer,
    q_optimizer,
    log_alpha,
    alpha_optimizer,
    alpha,
    real_steps,
    update_steps,
    output_dir,
):

    os.makedirs(
        output_dir,
        exist_ok=True,
    )


    prefix = os.path.join(
        output_dir,
        f"sac_online_step_{real_steps:06d}",
    )


    # --------------------------------------------------------
    # Individual model files
    #
    # These remain easy to use with benchmark scripts.
    # --------------------------------------------------------

    actor_path = (
        prefix
        + ".actor.pt"
    )

    qf1_path = (
        prefix
        + ".qf1.pt"
    )

    qf2_path = (
        prefix
        + ".qf2.pt"
    )


    torch.save(
        actor.state_dict(),
        actor_path,
    )


    torch.save(
        qf1.state_dict(),
        qf1_path,
    )


    torch.save(
        qf2.state_dict(),
        qf2_path,
    )


    # --------------------------------------------------------
    # Complete training state
    # --------------------------------------------------------

    training_path = (
        prefix
        + ".training.pt"
    )


    state = {
        "real_steps":
            int(
                real_steps
            ),

        "update_steps":
            int(
                update_steps
            ),

        "actor_state_dict":
            actor.state_dict(),

        "qf1_state_dict":
            qf1.state_dict(),

        "qf2_state_dict":
            qf2.state_dict(),

        "qf1_target_state_dict":
            qf1_target.state_dict(),

        "qf2_target_state_dict":
            qf2_target.state_dict(),

        "actor_optimizer_state_dict":
            actor_optimizer.state_dict(),

        "q_optimizer_state_dict":
            q_optimizer.state_dict(),

        "alpha":
            float(
                alpha
            ),

        "gamma":
            GAMMA,

        "tau":
            TAU,

        "batch_size":
            BATCH_SIZE,

        "policy_lr":
            POLICY_LR,

        "q_lr":
            Q_LR,
    }


    if (
        AUTOTUNE
        and log_alpha is not None
    ):

        state[
            "log_alpha"
        ] = (
            log_alpha
            .detach()
            .cpu()
            .clone()
        )


    if (
        AUTOTUNE
        and alpha_optimizer is not None
    ):

        state[
            "alpha_optimizer_state_dict"
        ] = (
            alpha_optimizer
            .state_dict()
        )


    torch.save(
        state,
        training_path,
    )


    print(
        "\nSAC checkpoint saved:"
    )

    print(
        f"  actor:    {actor_path}"
    )

    print(
        f"  qf1:      {qf1_path}"
    )

    print(
        f"  qf2:      {qf2_path}"
    )

    print(
        f"  training: {training_path}"
    )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Online SAC fine-tuning "
            "on the real Franka Panda."
        )
    )


    parser.add_argument(
        "--hostname",
        type=str,
        default=DEFAULT_HOSTNAME,
    )


    parser.add_argument(
        "--actor-path",
        type=str,
        default=ACTOR_PATH,
    )


    parser.add_argument(
        "--qf1-path",
        type=str,
        default=QF1_PATH,
    )


    parser.add_argument(
        "--qf2-path",
        type=str,
        default=QF2_PATH,
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
        "--update-after",
        type=int,
        default=DEFAULT_UPDATE_AFTER,
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
            "online_sac",
        ),
    )


    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
    )


    args = parser.parse_args()


    # ========================================================
    # Validate
    # ========================================================

    if panda_py is None:

        raise ImportError(
            "panda_py is not installed in this "
            "environment. Run this script on "
            "the Franka/Jetson machine."
        )


    required_paths = [
        args.actor_path,
        args.qf1_path,
        args.qf2_path,
        args.obs_rms_path,
    ]


    for path in required_paths:

        if not os.path.exists(
            path
        ):

            raise FileNotFoundError(
                path
            )


    if args.total_real_steps <= 0:

        raise ValueError(
            "--total-real-steps must be > 0"
        )


    if args.update_after < BATCH_SIZE:

        print(
            "[warning] "
            "--update-after is smaller than "
            f"BATCH_SIZE={BATCH_SIZE}. "
            "Updates will still wait until the "
            "buffer contains one full batch."
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
    # Normalization
    # ========================================================

    normalizer = FrozenObsNormalizer(
        args.obs_rms_path
    )


    # ========================================================
    # Networks
    # ========================================================

    (
        actor,
        qf1,
        qf2,
        qf1_target,
        qf2_target,
        dummy_env,
    ) = load_networks(
        actor_path=
            args.actor_path,

        qf1_path=
            args.qf1_path,

        qf2_path=
            args.qf2_path,

        device=
            device,
    )


    # ========================================================
    # Fresh optimizers for real adaptation
    # ========================================================

    q_optimizer = optim.Adam(
        list(
            qf1.parameters()
        )
        + list(
            qf2.parameters()
        ),
        lr=Q_LR,
    )


    actor_optimizer = optim.Adam(
        actor.parameters(),
        lr=POLICY_LR,
    )


    # ========================================================
    # Fresh entropy tuning state
    # ========================================================

    action_dim = int(
        np.prod(
            dummy_env
            .single_action_space
            .shape
        )
    )


    target_entropy = float(
        -action_dim
    )


    if AUTOTUNE:

        log_alpha = torch.tensor(
            [
                math.log(
                    INITIAL_ALPHA
                )
            ],
            dtype=torch.float32,
            requires_grad=True,
            device=device,
        )


        alpha_optimizer = optim.Adam(
            [
                log_alpha
            ],
            lr=Q_LR,
        )


        alpha = float(
            log_alpha
            .exp()
            .item()
        )


    else:

        log_alpha = None

        alpha_optimizer = None

        alpha = float(
            INITIAL_ALPHA
        )


    # ========================================================
    # Fresh REAL replay buffer
    # ========================================================

    dummy_env.single_observation_space.dtype = (
        np.float32
    )


    rb = ReplayBuffer(
        BUFFER_SIZE,
        dummy_env.single_observation_space,
        dummy_env.single_action_space,
        device,
        n_envs=1,
    )


    # ========================================================
    # Robot
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
    # Experiment info
    # ========================================================

    print(
        "\n"
        + "=" * 72
    )

    print(
        "REAL ROBOT SAC ONLINE FINE-TUNING"
    )

    print(
        "=" * 72
    )


    print(
        "Warm start:"
    )

    print(
        "  Actor: simulation"
    )

    print(
        "  Q1/Q2: simulation"
    )

    print(
        "  Target Q: copied from Q1/Q2"
    )

    print(
        "  Optimizers: fresh"
    )

    print(
        "  Replay buffer: fresh REAL data only"
    )


    print(
        f"\nTotal real interactions: "
        f"{args.total_real_steps}"
    )


    print(
        f"Update after: "
        f"{args.update_after}"
    )


    print(
        f"Batch size: "
        f"{BATCH_SIZE}"
    )


    print(
        f"Replay buffer capacity: "
        f"{BUFFER_SIZE}"
    )


    print(
        f"Actor LR: "
        f"{POLICY_LR}"
    )


    print(
        f"Q LR: "
        f"{Q_LR}"
    )


    print(
        f"Initial alpha: "
        f"{alpha:.4f}"
    )


    print(
        f"Target entropy: "
        f"{target_entropy:.1f}"
    )


    print(
        f"Success threshold: "
        f"{env.goal_threshold * 100:.1f} cm"
    )


    print(
        f"Joint action scale: "
        f"{env.joint_action_scale:.3f} rad"
    )


    print(
        "=" * 72
    )


    input(
        "\nCheck robot, workspace, controller "
        "and E-stop. Press Enter to start: "
    )


    # ========================================================
    # Initial episode
    # ========================================================

    obs_raw, reset_info = env.reset(
        seed=args.seed
    )


    obs = normalizer.normalize(
        obs_raw
    )


    real_steps = 0

    update_steps = 0

    episode_number = 1

    next_save_step = (
        args.save_every
    )

    start_time = time.time()


    print(
        f"\nEpisode {episode_number}"
        f" | initial distance="
        f"{reset_info['initial_distance'] * 100:.2f} cm"
    )


    # ========================================================
    # Online loop
    # ========================================================

    try:

        while (
            real_steps
            < args.total_real_steps
        ):

            # =================================================
            # SAC stochastic action
            #
            # NO RANDOM learning-start phase.
            #
            # Exploration comes from the pretrained
            # stochastic SAC actor itself.
            # =================================================

            obs_tensor = torch.as_tensor(
                obs,
                dtype=torch.float32,
                device=device,
            ).unsqueeze(
                0
            )


            with torch.no_grad():

                (
                    action_tensor,
                    _,
                    _
                ) = actor.get_action(
                    obs_tensor
                )


            action = (
                action_tensor
                .cpu()
                .numpy()[0]
                .astype(
                    np.float32
                )
            )


            # SAC tanh actor should already satisfy bounds,
            # but this is an explicit safety guard.
            action = np.clip(
                action,
                -1.0,
                1.0,
            )


            # =================================================
            # Real step
            # =================================================

            (
                next_obs_raw,
                reward,
                terminated,
                truncated,
                step_info,
            ) = env.step(
                action
            )


            next_obs = normalizer.normalize(
                next_obs_raw
            )


            # =================================================
            # Termination semantics
            #
            # Success = true terminal.
            #
            # Normal 50-step time limit is NOT a true
            # environment terminal for SAC bootstrap.
            #
            # Safety/controller error IS treated terminal.
            # =================================================

            safety_reason = (
                step_info.get(
                    "safety_reason"
                )
            )


            safety_terminal = bool(
                truncated
                and safety_reason
                not in (
                    None,
                    "time_limit",
                )
            )


            buffer_done = float(
                terminated
                or safety_terminal
            )


            # =================================================
            # Store REAL transition only
            # =================================================

            rb.add(
                obs.reshape(
                    1,
                    -1,
                ),
                next_obs.reshape(
                    1,
                    -1,
                ),
                action.reshape(
                    1,
                    -1,
                ),
                np.array(
                    [
                        reward
                    ],
                    dtype=np.float32,
                ),
                np.array(
                    [
                        buffer_done
                    ],
                    dtype=np.float32,
                ),
            )


            real_steps += 1


            replay_size = (
                get_replay_size(
                    rb
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
                f" | buffer="
                f"{replay_size}"
                f" | alpha="
                f"{alpha:.4f}"
            )


            # =================================================
            # Episode finished
            # =================================================

            episode_finished = bool(
                terminated
                or truncated
            )


            if episode_finished:

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
                    safety_reason
                    not in (
                        None,
                        "time_limit",
                    )
                ):

                    print(
                        "  SAFETY EVENT: "
                        f"{safety_reason}"
                    )


            # =================================================
            # SAC updates
            # =================================================

            can_update = (
                real_steps
                >= args.update_after
                and replay_size
                >= BATCH_SIZE
            )


            if can_update:

                for _ in range(
                    UPDATES_PER_STEP
                ):

                    update_steps += 1


                    data = rb.sample(
                        BATCH_SIZE
                    )


                    # =========================================
                    # Critic target
                    # =========================================

                    with torch.no_grad():

                        (
                            next_state_actions,
                            next_state_log_pi,
                            _,
                        ) = actor.get_action(
                            data.next_observations
                        )


                        qf1_next_target = (
                            qf1_target(
                                data.next_observations,
                                next_state_actions,
                            )
                        )


                        qf2_next_target = (
                            qf2_target(
                                data.next_observations,
                                next_state_actions,
                            )
                        )


                        min_qf_next_target = (
                            torch.min(
                                qf1_next_target,
                                qf2_next_target,
                            )
                            - alpha
                            * next_state_log_pi
                        )


                        next_q_value = (
                            data.rewards.flatten()
                            + (
                                1.0
                                - data.dones.flatten()
                            )
                            * GAMMA
                            * min_qf_next_target.view(
                                -1
                            )
                        )


                    # =========================================
                    # Q losses
                    # =========================================

                    qf1_a_values = (
                        qf1(
                            data.observations,
                            data.actions,
                        ).view(
                            -1
                        )
                    )


                    qf2_a_values = (
                        qf2(
                            data.observations,
                            data.actions,
                        ).view(
                            -1
                        )
                    )


                    qf1_loss = F.mse_loss(
                        qf1_a_values,
                        next_q_value,
                    )


                    qf2_loss = F.mse_loss(
                        qf2_a_values,
                        next_q_value,
                    )


                    qf_loss = (
                        qf1_loss
                        + qf2_loss
                    )


                    q_optimizer.zero_grad()

                    qf_loss.backward()

                    q_optimizer.step()


                    actor_loss_value = np.nan

                    alpha_loss_value = np.nan


                    # =========================================
                    # Actor update
                    # =========================================

                    if (
                        update_steps
                        % POLICY_FREQUENCY
                        == 0
                    ):

                        (
                            pi,
                            log_pi,
                            _
                        ) = actor.get_action(
                            data.observations
                        )


                        qf1_pi = qf1(
                            data.observations,
                            pi,
                        )


                        qf2_pi = qf2(
                            data.observations,
                            pi,
                        )


                        min_qf_pi = torch.min(
                            qf1_pi,
                            qf2_pi,
                        )


                        actor_loss = (
                            alpha
                            * log_pi
                            - min_qf_pi
                        ).mean()


                        actor_optimizer.zero_grad()

                        actor_loss.backward()

                        actor_optimizer.step()


                        actor_loss_value = float(
                            actor_loss.item()
                        )


                        # =====================================
                        # Alpha autotuning
                        # =====================================

                        if AUTOTUNE:

                            with torch.no_grad():

                                (
                                    _,
                                    log_pi_alpha,
                                    _
                                ) = actor.get_action(
                                    data.observations
                                )


                            alpha_loss = (
                                -log_alpha.exp()
                                * (
                                    log_pi_alpha
                                    + target_entropy
                                )
                            ).mean()


                            alpha_optimizer.zero_grad()

                            alpha_loss.backward()

                            alpha_optimizer.step()


                            alpha = float(
                                log_alpha
                                .exp()
                                .item()
                            )


                            alpha_loss_value = float(
                                alpha_loss.item()
                            )


                    # =========================================
                    # Target Q update
                    # =========================================

                    if (
                        update_steps
                        % TARGET_NETWORK_FREQUENCY
                        == 0
                    ):

                        soft_update(
                            qf1,
                            qf1_target,
                            TAU,
                        )


                        soft_update(
                            qf2,
                            qf2_target,
                            TAU,
                        )


                    if (
                        update_steps % 25
                        == 0
                    ):

                        print(
                            "\nSAC UPDATE"
                            f" | update="
                            f"{update_steps}"
                            f" | q_loss="
                            f"{qf_loss.item():.6f}"
                            f" | actor_loss="
                            f"{actor_loss_value:.6f}"
                            f" | alpha="
                            f"{alpha:.5f}"
                            f" | alpha_loss="
                            f"{alpha_loss_value:.6f}"
                        )


            # =================================================
            # Save checkpoint
            # =================================================

            while (
                real_steps
                >= next_save_step
            ):

                save_checkpoint(
                    actor=
                        actor,

                    qf1=
                        qf1,

                    qf2=
                        qf2,

                    qf1_target=
                        qf1_target,

                    qf2_target=
                        qf2_target,

                    actor_optimizer=
                        actor_optimizer,

                    q_optimizer=
                        q_optimizer,

                    log_alpha=
                        log_alpha,

                    alpha_optimizer=
                        alpha_optimizer,

                    alpha=
                        alpha,

                    real_steps=
                        real_steps,

                    update_steps=
                        update_steps,

                    output_dir=
                        args.output_dir,
                )


                next_save_step += (
                    args.save_every
                )


            # =================================================
            # New episode / next state
            # =================================================

            if episode_finished:

                episode_number += 1


                next_obs_raw, reset_info = (
                    env.reset()
                )


                obs = normalizer.normalize(
                    next_obs_raw
                )


                print(
                    f"\nEpisode "
                    f"{episode_number}"
                    f" | initial distance="
                    f"{reset_info['initial_distance'] * 100:.2f} cm"
                )


            else:

                obs = next_obs


    except KeyboardInterrupt:

        print(
            "\n\nOnline SAC training interrupted."
        )


    finally:

        # ====================================================
        # Always save current state
        # ====================================================

        print(
            "\nSaving final SAC state..."
        )


        try:

            save_checkpoint(
                actor=
                    actor,

                qf1=
                    qf1,

                qf2=
                    qf2,

                qf1_target=
                    qf1_target,

                qf2_target=
                    qf2_target,

                actor_optimizer=
                    actor_optimizer,

                q_optimizer=
                    q_optimizer,

                log_alpha=
                    log_alpha,

                alpha_optimizer=
                    alpha_optimizer,

                alpha=
                    alpha,

                real_steps=
                    real_steps,

                update_steps=
                    update_steps,

                output_dir=
                    args.output_dir,
            )

        except Exception as e:

            print(
                f"[save] warning: {e}"
            )


        print(
            "\nClosing real robot environment..."
        )


        try:

            env.close()

        except Exception as e:

            print(
                f"[close] warning: {e}"
            )


        elapsed = (
            time.time()
            - start_time
        )


        print(
            f"\nReal interactions: "
            f"{real_steps}"
        )


        print(
            f"SAC gradient updates: "
            f"{update_steps}"
        )


        print(
            f"Final alpha: "
            f"{alpha:.6f}"
        )


        print(
            f"Elapsed: "
            f"{elapsed:.1f} s"
        )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()