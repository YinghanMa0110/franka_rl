import os
import sys
import time

import gymnasium as gym
import numpy as np
import panda_gym  # noqa: F401
import torch


ROOT = os.path.dirname(os.path.abspath(__file__))

# Put this file in ~/franka_rl as eval_cleanrl.py
CLEANRL_DIR = os.path.join(ROOT, "training", "cleanrl")
if CLEANRL_DIR not in sys.path:
    sys.path.insert(0, CLEANRL_DIR)

from ppo_cont_2 import (  # noqa: E402
    Agent,
    Args,
    PushShapedReward,
    ScaleAction,
    make_env,
)


def find_wrapper(env, cls):
    w = env
    while isinstance(w, gym.Wrapper):
        if isinstance(w, cls):
            return w
        w = w.env
    return None


def make_render_env(env_id, args):
    """Mirror the training wrappers, but with render_mode='human'."""
    env = gym.make(
        env_id,
        reward_type="dense",
        render_mode="human",
    )

    if "Push" in env_id:
        env = PushShapedReward(
            env,
            w_reach=args.push_w_reach,
            w_prepush=args.push_w_prepush,
            w_prepush_near=args.push_w_prepush_near,
            prepush_distance=args.prepush_distance,
            prepush_threshold=args.prepush_threshold,
        )
        env = ScaleAction(
            env,
            scale=args.action_scale,
        )

    env = gym.wrappers.FlattenObservation(env)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.ClipAction(env)
    env = gym.wrappers.NormalizeObservation(env)
    env = gym.wrappers.TransformObservation(
        env,
        lambda obs: np.clip(obs, -10, 10),
        env.observation_space,
    )

    return env


def load_and_freeze_obs_rms(env, obs_rms_path):
    norm_w = find_wrapper(
        env,
        gym.wrappers.NormalizeObservation,
    )

    if norm_w is None:
        raise RuntimeError(
            "NormalizeObservation wrapper not found."
        )

    data = np.load(obs_rms_path)

    norm_w.obs_rms.mean = data["mean"].copy()
    norm_w.obs_rms.var = data["var"].copy()
    norm_w.obs_rms.count = data["count"].copy()

    if hasattr(norm_w, "update_running_mean"):
        norm_w.update_running_mean = False
    else:
        norm_w.obs_rms.update = lambda x: None

    print(
        "Loaded obs_rms and froze updates: "
        f"count={float(np.asarray(data['count'])):.1f}"
    )


def evaluate(
    model_path,
    obs_rms_path,
    env_id="PandaPush-v3",
    n_episodes=20,
    gamma=0.99,
    deterministic=True,
    seed=0,
    render=True,
    sleep=0.05,
    pause_between_episodes=False,
    debug_pos=True,
):
    args = Args()
    args.env_id = env_id
    args.gamma = gamma

    print("=" * 72)
    print("PPO Push evaluation")
    print(f"env:              {env_id}")
    print(f"model:            {model_path}")
    print(f"obs_rms:          {obs_rms_path}")
    print(f"episodes:         {n_episodes}")
    print(f"deterministic:     {deterministic}")
    print(f"action_scale:      {args.action_scale}")
    print(f"init_logstd:       {args.init_logstd}")
    print(f"logstd clamp:      [{args.min_logstd}, {args.max_logstd}]")
    print("=" * 72)

    # Dummy vector env only to reconstruct the network dimensions.
    dummy = gym.vector.SyncVectorEnv(
        [
            make_env(
                env_id,
                0,
                False,
                "eval_dummy",
                gamma,
                args,
            )
        ]
    )

    agent = Agent(
        dummy,
        init_logstd=args.init_logstd,
        min_logstd=args.min_logstd,
        max_logstd=args.max_logstd,
    )

    state_dict = torch.load(
        model_path,
        map_location="cpu",
    )
    agent.load_state_dict(state_dict)
    agent.eval()
    dummy.close()

    print("Model loaded successfully.")

    if render:
        env = make_render_env(
            env_id,
            args,
        )
    else:
        env = make_env(
            env_id,
            0,
            False,
            "eval",
            gamma,
            args,
        )()

    # Important: load training normalization statistics before reset.
    load_and_freeze_obs_rms(
        env,
        obs_rms_path,
    )

    if debug_pos:
        base = env.unwrapped

        print("\n--- task attributes ---")
        print(
            [
                x
                for x in dir(base.task)
                if not x.startswith("_")
            ]
        )

        print("\n--- robot attributes ---")
        print(
            [
                x
                for x in dir(base.robot)
                if not x.startswith("_")
            ]
        )
        print()

    successes = []
    returns = []
    final_distances = []

    for ep in range(n_episodes):
        obs, info = env.reset(
            seed=seed + ep,
        )

        done = False
        ep_ret = 0.0
        ever_success = False
        n_step = 0

        while not done:
            with torch.no_grad():
                x = torch.tensor(
                    obs,
                    dtype=torch.float32,
                ).unsqueeze(0)

                if deterministic:
                    action = agent.actor_mean(x)
                else:
                    action, _, _, _ = agent.get_action_and_value(x)

            raw_action = action.cpu().numpy()[0]

            if debug_pos and n_step % 10 == 0:
                clipped = np.clip(raw_action, -1.0, 1.0)
                executed = clipped * args.action_scale

                print(
                    f"  action[{n_step:3d}] "
                    f"raw={np.round(raw_action, 3)} "
                    f"clip={np.round(clipped, 3)} "
                    f"executed≈{np.round(executed, 3)}"
                )

            obs, reward, terminated, truncated, info = env.step(
                raw_action
            )

            ep_ret += float(reward)
            n_step += 1

            if bool(info.get("is_success", False)):
                ever_success = True

            if debug_pos and n_step % 10 == 0:
                base = env.unwrapped

                try:
                    goal = np.asarray(
                        base.task.get_goal(),
                        dtype=np.float32,
                    )
                    achieved = np.asarray(
                        base.task.get_achieved_goal(),
                        dtype=np.float32,
                    )
                    ee = np.asarray(
                        base.robot.get_ee_position(),
                        dtype=np.float32,
                    )

                    cube_goal_dist = float(
                        np.linalg.norm(achieved - goal)
                    )
                    ee_cube_dist = float(
                        np.linalg.norm(ee - achieved)
                    )

                    print(
                        f"  step {n_step:3d}: "
                        f"goal={np.round(goal, 3)} "
                        f"cube={np.round(achieved, 3)} "
                        f"ee={np.round(ee, 3)} "
                        f"cube_goal={cube_goal_dist * 100:.2f}cm "
                        f"ee_cube={ee_cube_dist * 100:.2f}cm"
                    )
                except Exception as e:
                    print(f"  position read failed: {e}")

            if render and sleep > 0:
                time.sleep(sleep)

            done = bool(terminated or truncated)

        base = env.unwrapped

        try:
            goal = np.asarray(
                base.task.get_goal(),
                dtype=np.float32,
            )
            achieved = np.asarray(
                base.task.get_achieved_goal(),
                dtype=np.float32,
            )
            ee = np.asarray(
                base.robot.get_ee_position(),
                dtype=np.float32,
            )

            final_distance = float(
                np.linalg.norm(achieved - goal)
            )
        except Exception:
            goal = None
            achieved = None
            ee = None
            final_distance = np.nan

        successes.append(float(ever_success))
        returns.append(float(ep_ret))
        final_distances.append(float(final_distance))

        print("\n" + "-" * 64)
        print(f"Episode {ep + 1}/{n_episodes}")
        print(f"  success: {ever_success}")
        print(f"  return:  {ep_ret:.3f}")
        print(f"  steps:   {n_step}")

        if goal is not None:
            print(f"  goal:    {np.round(goal, 3)}")
            print(f"  cube:    {np.round(achieved, 3)}")
            print(f"  ee:      {np.round(ee, 3)}")
            print(
                "  final cube-goal error: "
                f"{final_distance * 100:.2f} cm"
            )

        if pause_between_episodes:
            input("\nPress Enter for next episode...")

    env.close()

    success_rate = float(np.mean(successes) * 100.0)
    mean_return = float(np.mean(returns))
    mean_distance = float(np.nanmean(final_distances))

    print("\n" + "=" * 72)
    print("Evaluation results")
    print("=" * 72)
    print(
        f"Success rate: {success_rate:.1f}% "
        f"({int(sum(successes))}/{n_episodes})"
    )
    print(f"Mean return: {mean_return:.3f}")
    print(
        "Mean final cube-goal error: "
        f"{mean_distance * 100:.2f} cm"
    )
    print("=" * 72)

    return success_rate


if __name__ == "__main__":
    RUN = os.path.join(
        ROOT,
        "training",
        "cleanrl",
        "runs",
        "PandaPush-v3__ppo_cont_2__1__1788985317",
    )

    model_path = os.path.join(
        RUN,
        "ppo_cont_2.cleanrl_model",
    )

    obs_rms_path = os.path.join(
        RUN,
        "ppo_cont_2.obs_rms.npz",
    )

    env_id = "PandaPush-v3"

    print(f"Using run: {RUN}")
    print(f"env_id:    {env_id}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found:\n{model_path}"
        )

    if not os.path.exists(obs_rms_path):
        raise FileNotFoundError(
            f"obs_rms not found:\n{obs_rms_path}"
        )

    evaluate(
        model_path=model_path,
        obs_rms_path=obs_rms_path,
        env_id=env_id,
        n_episodes=20,
        deterministic=True,
        render=True,
        sleep=0.08,
        pause_between_episodes=False,
        debug_pos=True,
    )