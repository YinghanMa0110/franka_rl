# docs:
# https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy

import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter

import panda_gym  # noqa: F401

from replay_buffer import ReplayBuffer


# ============================================================
# Configuration
# ============================================================

DISTANCE_THRESHOLD = 0.02  # 2 cm


# ============================================================
# Arguments
# ============================================================

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[:-len(".py")]
    """experiment name"""

    seed: int = 1
    """random seed"""

    torch_deterministic: bool = True
    """torch deterministic mode"""

    cuda: bool = True
    """use CUDA if available"""

    track: bool = False
    """Weights & Biases tracking"""

    wandb_project_name: str = "cleanRL"
    wandb_entity: str = None

    capture_video: bool = False
    """record simulation videos"""

    save_model: bool = True
    """save actor, critics and obs normalization"""

    # --------------------------------------------------------
    # Environment
    # --------------------------------------------------------

    env_id: str = "PandaReach-v3"
    """PandaReach environment"""

    total_timesteps: int = 200_000
    """same order of training budget as PPO"""

    num_envs: int = 1

    # --------------------------------------------------------
    # SAC
    # --------------------------------------------------------

    buffer_size: int = 200_000

    gamma: float = 0.99

    tau: float = 0.005

    batch_size: int = 256

    learning_starts: int = 5_000

    policy_lr: float = 3e-4

    q_lr: float = 1e-3

    policy_frequency: int = 2

    target_network_frequency: int = 1

    alpha: float = 0.2

    autotune: bool = True


# ============================================================
# Environment
# ============================================================

def make_env(
    env_id,
    seed,
    idx,
    capture_video,
    run_name,
):

    def thunk():

        if capture_video and idx == 0:

            env = gym.make(
                env_id,
                reward_type="dense",
                control_type="joints",
                render_mode="rgb_array",
            )

            env = gym.wrappers.RecordVideo(
                env,
                f"videos/{run_name}",
            )

        else:

            # Your panda-gym build requires an explicit render mode.
            env = gym.make(
                env_id,
                reward_type="dense",
                control_type="joints",
                render_mode="rgb_array",
            )

        # ----------------------------------------------------
        # Same 2 cm task definition as PPO
        # ----------------------------------------------------

        env.unwrapped.task.distance_threshold = DISTANCE_THRESHOLD

        # ----------------------------------------------------
        # Same observation preprocessing as PPO
        # ----------------------------------------------------

        env = gym.wrappers.FlattenObservation(env)

        env = gym.wrappers.RecordEpisodeStatistics(env)

        # Important:
        # Do NOT use ClipAction here.
        # SAC Actor needs finite env action bounds.

        env = gym.wrappers.NormalizeObservation(env)

        env = gym.wrappers.TransformObservation(
            env,
            lambda obs: np.clip(obs, -10, 10),
            env.observation_space,
        )

        env.action_space.seed(seed)

        return env

    return thunk


# ============================================================
# Q Network
# ============================================================

class SoftQNetwork(nn.Module):

    def __init__(self, env):
        super().__init__()

        obs_dim = int(
            np.array(
                env.single_observation_space.shape
            ).prod()
        )

        action_dim = int(
            np.prod(
                env.single_action_space.shape
            )
        )

        self.fc1 = nn.Linear(
            obs_dim + action_dim,
            256,
        )

        self.fc2 = nn.Linear(
            256,
            256,
        )

        self.fc3 = nn.Linear(
            256,
            1,
        )

    def forward(self, x, a):

        x = torch.cat(
            [x, a],
            dim=1,
        )

        x = F.relu(
            self.fc1(x)
        )

        x = F.relu(
            self.fc2(x)
        )

        x = self.fc3(x)

        return x


# ============================================================
# Actor
# ============================================================

LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):

    def __init__(self, env):
        super().__init__()

        obs_dim = int(
            np.array(
                env.single_observation_space.shape
            ).prod()
        )

        action_dim = int(
            np.prod(
                env.single_action_space.shape
            )
        )

        self.fc1 = nn.Linear(
            obs_dim,
            256,
        )

        self.fc2 = nn.Linear(
            256,
            256,
        )

        self.fc_mean = nn.Linear(
            256,
            action_dim,
        )

        self.fc_logstd = nn.Linear(
            256,
            action_dim,
        )

        # ----------------------------------------------------
        # SAC action rescaling
        # ----------------------------------------------------

        self.register_buffer(
            "action_scale",
            torch.tensor(
                (
                    env.single_action_space.high
                    - env.single_action_space.low
                ) / 2.0,
                dtype=torch.float32,
            ),
        )

        self.register_buffer(
            "action_bias",
            torch.tensor(
                (
                    env.single_action_space.high
                    + env.single_action_space.low
                ) / 2.0,
                dtype=torch.float32,
            ),
        )

    def forward(self, x):

        x = F.relu(
            self.fc1(x)
        )

        x = F.relu(
            self.fc2(x)
        )

        mean = self.fc_mean(x)

        log_std = self.fc_logstd(x)

        log_std = torch.tanh(log_std)

        log_std = (
            LOG_STD_MIN
            + 0.5
            * (LOG_STD_MAX - LOG_STD_MIN)
            * (log_std + 1)
        )

        return mean, log_std

    def get_action(self, x):

        mean, log_std = self(x)

        std = log_std.exp()

        normal = torch.distributions.Normal(
            mean,
            std,
        )

        x_t = normal.rsample()

        y_t = torch.tanh(
            x_t
        )

        action = (
            y_t
            * self.action_scale
            + self.action_bias
        )

        log_prob = normal.log_prob(
            x_t
        )

        log_prob -= torch.log(
            self.action_scale
            * (1 - y_t.pow(2))
            + 1e-6
        )

        log_prob = log_prob.sum(
            1,
            keepdim=True,
        )

        mean_action = (
            torch.tanh(mean)
            * self.action_scale
            + self.action_bias
        )

        return (
            action,
            log_prob,
            mean_action,
        )


# ============================================================
# Wrapper utility
# ============================================================

def find_wrapper(env, cls):

    w = env

    while isinstance(w, gym.Wrapper):

        if isinstance(w, cls):
            return w

        w = w.env

    return None


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    args = tyro.cli(Args)

    run_name = (
        f"{args.env_id}"
        f"__{args.exp_name}"
        f"__{args.seed}"
        f"__{int(time.time())}"
    )


    # ========================================================
    # Tracking
    # ========================================================

    if args.track:

        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )


    # ========================================================
    # TensorBoard
    # ========================================================

    writer = SummaryWriter(
        f"runs/{run_name}"
    )

    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % (
            "\n".join(
                [
                    f"|{key}|{value}|"
                    for key, value
                    in vars(args).items()
                ]
            )
        ),
    )


    # ========================================================
    # Seeding
    # ========================================================

    random.seed(args.seed)

    np.random.seed(args.seed)

    torch.manual_seed(args.seed)

    torch.backends.cudnn.deterministic = (
        args.torch_deterministic
    )


    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        and args.cuda
        else "cpu"
    )


    print(
        "Device:",
        device,
    )


    # ========================================================
    # Environments
    # ========================================================

    envs = gym.vector.SyncVectorEnv(
        [
            make_env(
                args.env_id,
                args.seed + i,
                i,
                args.capture_video,
                run_name,
            )
            for i
            in range(args.num_envs)
        ]
    )


    assert isinstance(
        envs.single_action_space,
        gym.spaces.Box,
    ), "Only continuous Box action space supported."


    # ========================================================
    # Sanity checks
    # ========================================================

    print(
        "\n" + "=" * 72
    )

    print(
        "SAC PANDA REACH TRAINING"
    )

    print(
        "=" * 72
    )

    print(
        "Environment:",
        args.env_id,
    )

    print(
        "Observation space:",
        envs.single_observation_space,
    )

    print(
        "Action space:",
        envs.single_action_space,
    )

    print(
        "Action shape:",
        envs.single_action_space.shape,
    )

    print(
        "Distance threshold:",
        envs.envs[0]
        .unwrapped
        .task
        .distance_threshold,
    )

    print(
        "Total timesteps:",
        args.total_timesteps,
    )

    print(
        "Learning starts:",
        args.learning_starts,
    )

    print(
        "=" * 72
    )


    # --------------------------------------------------------
    # SAC requires finite action bounds
    # --------------------------------------------------------

    if not np.all(
        np.isfinite(
            envs.single_action_space.low
        )
    ):

        raise RuntimeError(
            "Action-space lower bounds are not finite."
        )

    if not np.all(
        np.isfinite(
            envs.single_action_space.high
        )
    ):

        raise RuntimeError(
            "Action-space upper bounds are not finite."
        )


    if envs.single_action_space.shape != (7,):

        raise RuntimeError(
            f"Expected 7D joint action space, "
            f"got {envs.single_action_space.shape}"
        )


    # ========================================================
    # Networks
    # ========================================================

    actor = Actor(
        envs
    ).to(device)


    qf1 = SoftQNetwork(
        envs
    ).to(device)

    qf2 = SoftQNetwork(
        envs
    ).to(device)


    qf1_target = SoftQNetwork(
        envs
    ).to(device)

    qf2_target = SoftQNetwork(
        envs
    ).to(device)


    qf1_target.load_state_dict(
        qf1.state_dict()
    )

    qf2_target.load_state_dict(
        qf2.state_dict()
    )


    # ========================================================
    # Optimizers
    # ========================================================

    q_optimizer = optim.Adam(
        list(qf1.parameters())
        + list(qf2.parameters()),
        lr=args.q_lr,
    )


    actor_optimizer = optim.Adam(
        actor.parameters(),
        lr=args.policy_lr,
    )


    # ========================================================
    # Entropy coefficient
    # ========================================================

    if args.autotune:

        target_entropy = -torch.prod(
            torch.Tensor(
                envs.single_action_space.shape
            ).to(device)
        ).item()


        log_alpha = torch.zeros(
            1,
            requires_grad=True,
            device=device,
        )


        alpha = (
            log_alpha.exp().item()
        )


        a_optimizer = optim.Adam(
            [log_alpha],
            lr=args.q_lr,
        )

    else:

        alpha = args.alpha


    # ========================================================
    # Replay Buffer
    # ========================================================

    envs.single_observation_space.dtype = np.float32


    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        n_envs=args.num_envs,
        handle_timeout_termination=False,
    )


    # ========================================================
    # Start
    # ========================================================

    start_time = time.time()

    obs, _ = envs.reset(seed=args.seed)

    episode_returns = np.zeros(args.num_envs, dtype=np.float32)
    episode_lengths = np.zeros(args.num_envs, dtype=np.int32)
    episode_count = 0


    # ========================================================
    # Training loop
    # ========================================================

    for global_step in range(
        args.total_timesteps
    ):


        # ----------------------------------------------------
        # Action selection
        # ----------------------------------------------------

        if global_step < args.learning_starts:

            actions = np.array(
                [
                    envs.single_action_space.sample()
                    for _
                    in range(envs.num_envs)
                ]
            )

        else:

            with torch.no_grad():

                actions, _, _ = (
                    actor.get_action(
                        torch.tensor(
                            obs,
                            dtype=torch.float32,
                            device=device,
                        )
                    )
                )

            actions = (
                actions
                .cpu()
                .numpy()
            )


        # ----------------------------------------------------
        # Environment interaction
        # ----------------------------------------------------

        (next_obs,rewards,terminations,truncations,infos,) = envs.step(actions)
        episode_returns += rewards
        episode_lengths += 1

        dones = np.logical_or(
            terminations,
            truncations,
        )

        for i, done in enumerate(dones):
            if done:
                episode_count += 1

                print(
                    f"global_step={global_step}, "
                    f"episode={episode_count}, "
                    f"episodic_return={episode_returns[i]:.4f}, "
                    f"episodic_length={episode_lengths[i]}"
                )

                writer.add_scalar(
                    "charts/episodic_return",
                    episode_returns[i],
                    global_step,
                )

                writer.add_scalar(
                    "charts/episodic_length",
                    episode_lengths[i],
                    global_step,
                )

                episode_returns[i] = 0.0
                episode_lengths[i] = 0


        # ----------------------------------------------------
        # Episode logging
        # ----------------------------------------------------

        # if "final_info" in infos:

        #     for info in infos["final_info"]:

        #         if info is not None:

        #             print(
        #                 f"global_step={global_step}, "
        #                 f"episodic_return="
        #                 f"{info['episode']['r']}"
        #             )

        #             writer.add_scalar(
        #                 "charts/episodic_return",
        #                 info["episode"]["r"],
        #                 global_step,
        #             )

        #             writer.add_scalar(
        #                 "charts/episodic_length",
        #                 info["episode"]["l"],
        #                 global_step,
        #             )

        #             break


        # ----------------------------------------------------
        # Correct next obs for truncated episodes
        # ----------------------------------------------------

        real_next_obs = next_obs.copy()
        if "final_observation" in infos:
            for idx, trunc in enumerate(truncations):
                if trunc:
                    real_next_obs[idx] = (infos["final_observation"][idx])


        # ----------------------------------------------------
        # Replay buffer
        # ----------------------------------------------------

        rb.add(
            obs,
            real_next_obs,
            actions,
            rewards,
            terminations,
            infos,
        )


        obs = next_obs


        # ====================================================
        # SAC updates
        # ====================================================

        if global_step > args.learning_starts:

            data = rb.sample(
                args.batch_size
            )


            # ------------------------------------------------
            # Target Q
            # ------------------------------------------------

            with torch.no_grad():

                (
                    next_state_actions,
                    next_state_log_pi,
                    _,
                ) = actor.get_action(
                    data.next_observations
                )


                qf1_next_target = qf1_target(
                    data.next_observations,
                    next_state_actions,
                )


                qf2_next_target = qf2_target(
                    data.next_observations,
                    next_state_actions,
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
                        1
                        - data.dones.flatten()
                    )
                    * args.gamma
                    * min_qf_next_target.view(-1)
                )


            # ------------------------------------------------
            # Critic loss
            # ------------------------------------------------

            qf1_a_values = qf1(
                data.observations,
                data.actions,
            ).view(-1)


            qf2_a_values = qf2(
                data.observations,
                data.actions,
            ).view(-1)


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


            # ------------------------------------------------
            # Actor
            # ------------------------------------------------

            if (
                global_step
                % args.policy_frequency
                == 0
            ):

                for _ in range(
                    args.policy_frequency
                ):

                    pi, log_pi, _ = (
                        actor.get_action(
                            data.observations
                        )
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
                        alpha * log_pi
                        - min_qf_pi
                    ).mean()


                    actor_optimizer.zero_grad()

                    actor_loss.backward()

                    actor_optimizer.step()


                    # ----------------------------------------
                    # Alpha autotuning
                    # ----------------------------------------

                    if args.autotune:

                        with torch.no_grad():

                            _, log_pi, _ = (
                                actor.get_action(
                                    data.observations
                                )
                            )


                        alpha_loss = (
                            -log_alpha.exp()
                            * (
                                log_pi
                                + target_entropy
                            )
                        ).mean()


                        a_optimizer.zero_grad()

                        alpha_loss.backward()

                        a_optimizer.step()


                        alpha = (
                            log_alpha.exp().item()
                        )


            # ------------------------------------------------
            # Target networks
            # ------------------------------------------------

            if (
                global_step
                % args.target_network_frequency
                == 0
            ):

                for param, target_param in zip(
                    qf1.parameters(),
                    qf1_target.parameters(),
                ):

                    target_param.data.copy_(
                        args.tau
                        * param.data
                        + (
                            1
                            - args.tau
                        )
                        * target_param.data
                    )


                for param, target_param in zip(
                    qf2.parameters(),
                    qf2_target.parameters(),
                ):

                    target_param.data.copy_(
                        args.tau
                        * param.data
                        + (
                            1
                            - args.tau
                        )
                        * target_param.data
                    )


            # ------------------------------------------------
            # TensorBoard metrics
            # ------------------------------------------------

            if global_step % 100 == 0:

                writer.add_scalar(
                    "losses/qf1_values",
                    qf1_a_values.mean().item(),
                    global_step,
                )

                writer.add_scalar(
                    "losses/qf2_values",
                    qf2_a_values.mean().item(),
                    global_step,
                )

                writer.add_scalar(
                    "losses/qf1_loss",
                    qf1_loss.item(),
                    global_step,
                )

                writer.add_scalar(
                    "losses/qf2_loss",
                    qf2_loss.item(),
                    global_step,
                )

                writer.add_scalar(
                    "losses/qf_loss",
                    qf_loss.item() / 2.0,
                    global_step,
                )

                writer.add_scalar(
                    "losses/actor_loss",
                    actor_loss.item(),
                    global_step,
                )

                writer.add_scalar(
                    "losses/alpha",
                    alpha,
                    global_step,
                )


                if args.autotune:

                    writer.add_scalar(
                        "losses/alpha_loss",
                        alpha_loss.item(),
                        global_step,
                    )


                sps = int(
                    global_step
                    / (
                        time.time()
                        - start_time
                    )
                )


                print(
                    "SPS:",
                    sps,
                )


                writer.add_scalar(
                    "charts/SPS",
                    sps,
                    global_step,
                )


    # ========================================================
    # Save
    # ========================================================

    if args.save_model:

        run_dir = f"runs/{run_name}"

        os.makedirs(
            run_dir,
            exist_ok=True,
        )


        actor_path = os.path.join(
            run_dir,
            f"{args.exp_name}.actor.pt",
        )


        qf1_path = os.path.join(
            run_dir,
            f"{args.exp_name}.qf1.pt",
        )


        qf2_path = os.path.join(
            run_dir,
            f"{args.exp_name}.qf2.pt",
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


        print(
            f"Actor saved to: {actor_path}"
        )


        print(
            f"QF1 saved to: {qf1_path}"
        )


        print(
            f"QF2 saved to: {qf2_path}"
        )


        # ----------------------------------------------------
        # Save NormalizeObservation statistics
        # ----------------------------------------------------

        norm_w = find_wrapper(
            envs.envs[0],
            gym.wrappers.NormalizeObservation,
        )


        assert norm_w is not None, (
            "NormalizeObservation wrapper not found"
        )


        obs_rms_path = os.path.join(
            run_dir,
            f"{args.exp_name}.obs_rms.npz",
        )


        np.savez(
            obs_rms_path,
            mean=norm_w.obs_rms.mean,
            var=norm_w.obs_rms.var,
            count=norm_w.obs_rms.count,
        )


        print(
            f"obs_rms saved to: "
            f"{obs_rms_path}"
        )


    # ========================================================
    # Close
    # ========================================================

    envs.close()

    writer.close()