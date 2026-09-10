# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
import os
import random
import time
from collections import deque
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
import panda_gym


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """whether to track with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb project's name"""
    wandb_entity: str = None
    """the wandb entity/team"""
    capture_video: bool = False
    """whether to capture videos"""
    save_model: bool = True
    """whether to save the model"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user/org name of the Hugging Face repository"""

    # Environment / Push shaping
    env_id: str = "PandaPush-v3"
    """Panda environment id; reward_type is forced to dense below"""
    push_w_reach: float = 0.20
    """weight for EE-to-cube distance shaping"""
    push_w_prepush: float = 0.40
    """weight for EE-to-prepush-position shaping before the EE reaches the pushing region"""
    push_w_prepush_near: float = 0.10
    """reduced pre-push weight once the EE is already near the useful pushing region"""
    prepush_distance: float = 0.05
    """desired XY distance behind cube relative to cube->goal direction (m)"""
    prepush_threshold: float = 0.04
    """distance threshold (m) for switching from approach shaping to push shaping"""
    action_scale: float = 0.40
    """scale applied after clipping; 1.0 keeps panda-gym's original action magnitude"""

    # Algorithm specific arguments
    total_timesteps: int = 200000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """keep 1 while exporting a single NormalizeObservation statistic for real deployment"""
    num_steps: int = 2048
    """steps per policy rollout"""
    anneal_lr: bool = False
    """toggle learning rate annealing"""
    gamma: float = 0.99
    """discount factor"""
    gae_lambda: float = 0.95
    """GAE lambda"""
    num_minibatches: int = 32
    """number of mini-batches"""
    update_epochs: int = 10
    """epochs per PPO update"""
    norm_adv: bool = True
    """normalize advantages"""
    clip_coef: float = 0.2
    """PPO clipping coefficient"""
    clip_vloss: bool = True
    """use clipped value loss"""
    ent_coef: float = 0.001
    """entropy coefficient; reduced for finer Push control"""
    vf_coef: float = 0.5
    """value loss coefficient"""
    max_grad_norm: float = 0.5
    """gradient norm clipping"""
    target_kl: float = None
    """optional target KL threshold"""
    init_logstd: float = -1.0
    """initial Gaussian policy log std; -1.0 -> std ~= 0.368"""
    min_logstd: float = -2.0
    """minimum learned policy log std"""
    max_logstd: float = -0.5
    """maximum learned policy log std; prevents exploration variance from growing too large"""
    debug_log_interval: int = 1000
    """steps between debug metrics"""

    # to be filled in runtime
    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0


class PushShapedReward(gym.Wrapper):
    """
    Dense Push shaping for PPO.

    Base PandaPush dense reward rewards cube->goal progress.
    This wrapper additionally gives the policy a signal for:
      1) approaching the cube;
      2) approaching a pre-push point BEHIND the cube, opposite the cube->goal direction.

    It also places useful diagnostics in info:
      cube_goal_distance, ee_cube_distance, prepush_distance, is_success.
    """

    def __init__(
        self,
        env,
        w_reach=0.20,
        w_prepush=0.40,
        w_prepush_near=0.10,
        prepush_distance=0.05,
        prepush_threshold=0.04,
    ):
        super().__init__(env)
        self.w_reach = w_reach
        self.w_prepush = w_prepush
        self.w_prepush_near = w_prepush_near
        self.prepush_distance = prepush_distance
        self.prepush_threshold = prepush_threshold

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        base = self.env.unwrapped
        ee = np.asarray(base.robot.get_ee_position(), dtype=np.float32)
        obj = np.asarray(base.task.get_achieved_goal(), dtype=np.float32)
        goal = np.asarray(obs["desired_goal"], dtype=np.float32)

        # Original Push task objective: move cube toward goal.
        d_goal = float(np.linalg.norm(obj - goal))

        # Helps PPO discover contact with the cube.
        d_reach = float(np.linalg.norm(ee - obj))

        # Desired XY pre-push point:
        #   prepush = cube - d * unit(cube -> goal)
        # so the EE is encouraged to get behind the cube before pushing.
        direction_xy = goal[:2] - obj[:2]
        direction_norm = float(np.linalg.norm(direction_xy))

        if direction_norm > 1e-8:
            direction_xy = direction_xy / direction_norm
            prepush_xy = obj[:2] - self.prepush_distance * direction_xy
            d_prepush = float(np.linalg.norm(ee[:2] - prepush_xy))
        else:
            d_prepush = 0.0

        # Two-stage shaping:
        #   far from the useful pushing configuration -> strongly reward reaching pre-push;
        #   once near it -> weaken the pre-push term so the base cube->goal reward can
        #   dominate and encourage continued pushing instead of hovering behind the cube.
        if d_prepush > self.prepush_threshold:
            prepush_weight = self.w_prepush
        else:
            prepush_weight = self.w_prepush_near

        shaped_reward = (
            float(reward)
            - self.w_reach * d_reach
            - prepush_weight * d_prepush
        )

        # Use panda-gym's own task success definition when available.
        try:
            success = float(np.asarray(base.task.is_success(obj, goal)).item())
        except Exception:
            # Diagnostic fallback only. Panda-gym Push commonly uses a 5 cm threshold.
            success = float(d_goal < 0.05)

        info = dict(info)
        info["cube_goal_distance"] = d_goal
        info["ee_cube_distance"] = d_reach
        info["prepush_distance"] = d_prepush
        info["prepush_weight"] = prepush_weight
        info["is_success"] = success

        return obs, shaped_reward, terminated, truncated, info


class ScaleAction(gym.ActionWrapper):
    """Scale clipped Panda actions before they reach the underlying environment."""

    def __init__(self, env, scale=0.40):
        super().__init__(env)
        if not (0.0 < scale <= 1.0):
            raise ValueError("action scale must be in (0, 1]")
        self.scale = float(scale)

    def action(self, action):
        # An outer ClipAction wrapper will normally already bound this to [-1, 1].
        # Clip again here defensively, then reduce the physical EE displacement.
        return np.clip(action, -1.0, 1.0) * self.scale


def make_env(env_id, idx, capture_video, run_name, gamma, args):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array", reward_type="dense")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id, reward_type="dense")

        if "Push" in env_id:
            env = PushShapedReward(
                env,
                w_reach=args.push_w_reach,
                w_prepush=args.push_w_prepush,
                w_prepush_near=args.push_w_prepush_near,
                prepush_distance=args.prepush_distance,
                prepush_threshold=args.prepush_threshold,
            )
            env = ScaleAction(env, scale=args.action_scale)

        # Keep the full GoalEnv information: observation + achieved_goal + desired_goal.
        env = gym.wrappers.FlattenObservation(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)

        # Panda end-effector actions are bounded; the Gaussian policy itself is unbounded.
        # ClipAction clips before passing the action to the underlying Panda environment.
        env = gym.wrappers.ClipAction(env)

        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(
            env,
            lambda obs: np.clip(obs, -10, 10),
            env.observation_space,
        )

        # Keep reward normalization disabled while diagnosing the shaped reward.
        # env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        # env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))

        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs, init_logstd=-1.0, min_logstd=-2.0, max_logstd=-0.5):
        super().__init__()

        obs_dim = np.array(envs.single_observation_space.shape).prod()
        action_dim = np.prod(envs.single_action_space.shape)

        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )

        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, action_dim), std=0.01),
        )

        # Original CleanRL version starts at logstd=0 -> std=1.
        # For Panda Push this can create a lot of saturated/clipped actions.
        self.actor_logstd = nn.Parameter(
            torch.full((1, action_dim), float(init_logstd))
        )
        self.min_logstd = float(min_logstd)
        self.max_logstd = float(max_logstd)

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = torch.clamp(
            self.actor_logstd,
            min=self.min_logstd,
            max=self.max_logstd,
        ).expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)

        if action is None:
            action = probs.sample()

        return (
            action,
            probs.log_prob(action).sum(1),
            probs.entropy().sum(1),
            self.critic(x),
        )


if __name__ == "__main__":
    args = tyro.cli(Args)

    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size

    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

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

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # Environment setup
    envs = gym.vector.SyncVectorEnv(
        [
            make_env(
                args.env_id,
                i,
                args.capture_video,
                run_name,
                args.gamma,
                args,
            )
            for i in range(args.num_envs)
        ]
    )

    assert isinstance(
        envs.single_action_space, gym.spaces.Box
    ), "only continuous action space is supported"

    agent = Agent(
        envs,
        init_logstd=args.init_logstd,
        min_logstd=args.min_logstd,
        max_logstd=args.max_logstd,
    ).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # PPO storage
    obs = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_observation_space.shape
    ).to(device)
    actions = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_action_space.shape
    ).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # Start training
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    # Rolling success rate over the last 100 completed episodes.
    success_window = deque(maxlen=100)
    last_debug_bucket = -1

    for iteration in range(1, args.num_iterations + 1):
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # Action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()

            actions[step] = action
            logprobs[step] = logprob

            # ---- Action saturation diagnostics ----
            # Panda EE action bounds are [-1, 1]; ClipAction executes the clipped action.
            raw_clip_fraction = (torch.abs(action) > 1.0).float().mean()
            raw_action_abs_mean = action.abs().mean()
            action_mean_now = agent.actor_mean(next_obs)
            action_mean_abs = action_mean_now.abs().mean()
            action_mean_max = action_mean_now.abs().max()
            clamped_logstd = torch.clamp(
                agent.actor_logstd,
                min=args.min_logstd,
                max=args.max_logstd,
            )
            policy_std_mean = torch.exp(clamped_logstd).mean()
            executed_action_abs_mean = (
                torch.clamp(action, -1.0, 1.0).abs().mean() * args.action_scale
            )

            # Execute the environment step
            next_obs, reward, terminations, truncations, infos = envs.step(
                action.cpu().numpy()
            )
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs = torch.Tensor(next_obs).to(device)
            next_done = torch.Tensor(next_done).to(device)

            # Log debug metrics only every N environment transitions.
            debug_bucket = global_step // args.debug_log_interval
            if debug_bucket != last_debug_bucket:
                last_debug_bucket = debug_bucket

                writer.add_scalar(
                    "debug/action_clip_fraction",
                    raw_clip_fraction.item(),
                    global_step,
                )
                writer.add_scalar(
                    "debug/action_abs_mean",
                    raw_action_abs_mean.item(),
                    global_step,
                )
                writer.add_scalar(
                    "debug/policy_std_mean",
                    policy_std_mean.item(),
                    global_step,
                )
                writer.add_scalar(
                    "debug/action_mean_abs",
                    action_mean_abs.item(),
                    global_step,
                )
                writer.add_scalar(
                    "debug/action_mean_max",
                    action_mean_max.item(),
                    global_step,
                )
                writer.add_scalar(
                    "debug/executed_action_abs_mean",
                    executed_action_abs_mean.item(),
                    global_step,
                )

                for key in (
                    "cube_goal_distance",
                    "ee_cube_distance",
                    "prepush_distance",
                    "prepush_weight",
                    "is_success",
                ):
                    if key in infos:
                        writer.add_scalar(
                            f"debug/{key}",
                            float(np.mean(infos[key])),
                            global_step,
                        )

            # Episode logging
            if "episode" in infos:
                mask = infos.get(
                    "_episode",
                    np.array([True] * len(infos["episode"]["r"])),
                )

                for idx, done in enumerate(mask):
                    if not done:
                        continue

                    episode_return = float(infos["episode"]["r"][idx])
                    episode_length = float(infos["episode"]["l"][idx])

                    print(
                        f"global_step={global_step}, "
                        f"episodic_return={episode_return:.3f}, "
                        f"episodic_length={episode_length:.0f}"
                    )

                    writer.add_scalar(
                        "charts/episodic_return",
                        episode_return,
                        global_step,
                    )
                    writer.add_scalar(
                        "charts/episodic_length",
                        episode_length,
                        global_step,
                    )

                    if "is_success" in infos:
                        episode_success = float(infos["is_success"][idx])
                        success_window.append(episode_success)

                        writer.add_scalar(
                            "charts/episode_success",
                            episode_success,
                            global_step,
                        )
                        writer.add_scalar(
                            "charts/success_rate_100",
                            np.mean(success_window),
                            global_step,
                        )

                        print(
                            f"  success={episode_success:.0f}, "
                            f"success_rate_100={np.mean(success_window):.3f}"
                        )

                    if "cube_goal_distance" in infos:
                        writer.add_scalar(
                            "charts/final_cube_goal_distance",
                            float(infos["cube_goal_distance"][idx]),
                            global_step,
                        )

        # Bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0

            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]

                delta = (
                    rewards[t]
                    + args.gamma * nextvalues * nextnonterminal
                    - values[t]
                )
                advantages[t] = lastgaelam = (
                    delta
                    + args.gamma
                    * args.gae_lambda
                    * nextnonterminal
                    * lastgaelam
                )

            returns = advantages + values

        # Flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # PPO optimization
        b_inds = np.arange(args.batch_size)
        clipfracs = []

        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)

            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds],
                    b_actions[mb_inds],
                )

                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [
                        (
                            (ratio - 1.0).abs() > args.clip_coef
                        ).float().mean().item()
                    ]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (
                        mb_advantages - mb_advantages.mean()
                    ) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio,
                    1 - args.clip_coef,
                    1 + args.clip_coef,
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (
                        newvalue - b_returns[mb_inds]
                    ) ** 2

                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )

                    v_loss_clipped = (
                        v_clipped - b_returns[mb_inds]
                    ) ** 2

                    v_loss_max = torch.max(
                        v_loss_unclipped,
                        v_loss_clipped,
                    )

                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * (
                        (newvalue - b_returns[mb_inds]) ** 2
                    ).mean()

                entropy_loss = entropy.mean()
                loss = (
                    pg_loss
                    - args.ent_coef * entropy_loss
                    + v_loss * args.vf_coef
                )

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    agent.parameters(),
                    args.max_grad_norm,
                )
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred = b_values.cpu().numpy()
        y_true = b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = (
            np.nan
            if var_y == 0
            else 1 - np.var(y_true - y_pred) / var_y
        )

        # Standard CleanRL metrics
        writer.add_scalar(
            "charts/learning_rate",
            optimizer.param_groups[0]["lr"],
            global_step,
        )
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar(
            "losses/old_approx_kl",
            old_approx_kl.item(),
            global_step,
        )
        writer.add_scalar(
            "losses/approx_kl",
            approx_kl.item(),
            global_step,
        )
        writer.add_scalar(
            "losses/clipfrac",
            np.mean(clipfracs),
            global_step,
        )
        writer.add_scalar(
            "losses/explained_variance",
            explained_var,
            global_step,
        )

        sps = int(global_step / (time.time() - start_time))
        print("SPS:", sps)
        writer.add_scalar("charts/SPS", sps, global_step)

    # Save model + observation normalization statistics
    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")

        def find_wrapper(env, cls):
            w = env
            while isinstance(w, gym.Wrapper):
                if isinstance(w, cls):
                    return w
                w = w.env
            return None

        norm_w = find_wrapper(
            envs.envs[0],
            gym.wrappers.NormalizeObservation,
        )
        assert norm_w is not None, "NormalizeObservation wrapper not found"

        obs_rms_path = f"runs/{run_name}/{args.exp_name}.obs_rms.npz"
        np.savez(
            obs_rms_path,
            mean=norm_w.obs_rms.mean,
            var=norm_w.obs_rms.var,
            count=norm_w.obs_rms.count,
        )

        print(f"obs_rms saved to {obs_rms_path}")
        print(f"  mean={norm_w.obs_rms.mean}")
        print(f"  var={norm_w.obs_rms.var}")
        print(f"  count={norm_w.obs_rms.count}")

        if args.upload_model:
            from cleanrl_utils.huggingface import push_to_hub

            repo_name = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
            repo_id = (
                f"{args.hf_entity}/{repo_name}"
                if args.hf_entity
                else repo_name
            )
            push_to_hub(
                args,
                [],
                repo_id,
                "PPO",
                f"runs/{run_name}",
                f"videos/{run_name}-eval",
            )

    envs.close()
    writer.close()