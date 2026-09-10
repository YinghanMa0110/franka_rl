import os
import csv
from datetime import datetime

import gymnasium as gym
import numpy as np
import torch

import panda_gym  # noqa: F401

from sac_reach_joints import Actor


# ============================================================
# Configuration
# ============================================================

ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "../..",
    )
)

ENV_ID = "PandaReach-v3"
REWARD_TYPE = "dense"
CONTROL_TYPE = "joints"

DISTANCE_THRESHOLD = 0.02

N_EPISODES = 20

# SAC eval usually uses mean action
DETERMINISTIC = True

RECORD_VIDEO = True
SHOW_LIVE = False

DEBUG = True


# ============================================================
# IMPORTANT:
# Replace this with your actual SAC run directory
# ============================================================

RUN_DIR = os.path.join(
    ROOT,
    "runs",
    "PandaReach-v3__sac_reach_joints__1__1789075275",
)

ACTOR_PATH = os.path.join(
    RUN_DIR,
    "sac_reach_joints.actor.pt",
)

OBS_RMS_PATH = os.path.join(
    RUN_DIR,
    "sac_reach_joints.obs_rms.npz",
)


# ============================================================
# Output
# ============================================================

timestamp = datetime.now().strftime(
    "%Y%m%d_%H%M%S"
)

VIDEO_DIR = os.path.join(
    ROOT,
    "videos",
    f"sac_reach_joint_2cm_eval_{timestamp}",
)

RESULT_DIR = os.path.join(
    ROOT,
    "results",
)

os.makedirs(
    RESULT_DIR,
    exist_ok=True,
)

RESULT_CSV = os.path.join(
    RESULT_DIR,
    f"sac_reach_joint_2cm_eval_{timestamp}.csv",
)


# ============================================================
# Environment
# ============================================================

def build_env(
    record_video=False,
    show_live=False,
):

    if record_video and show_live:
        raise ValueError(
            "Use either RECORD_VIDEO=True "
            "or SHOW_LIVE=True, not both."
        )

    if show_live:
        render_mode = "human"
    else:
        render_mode = "rgb_array"

    env = gym.make(
        ENV_ID,
        reward_type=REWARD_TYPE,
        control_type=CONTROL_TYPE,
        render_mode=render_mode,
    )

    env.unwrapped.task.distance_threshold = (
        DISTANCE_THRESHOLD
    )

    if record_video:

        os.makedirs(
            VIDEO_DIR,
            exist_ok=True,
        )

        env = gym.wrappers.RecordVideo(
            env,
            video_folder=VIDEO_DIR,
            episode_trigger=lambda episode_id: True,
            name_prefix="sac_reach_joint_2cm",
        )

    env = gym.wrappers.FlattenObservation(env)

    env = gym.wrappers.RecordEpisodeStatistics(env)

    # Same preprocessing as training
    env = gym.wrappers.NormalizeObservation(env)

    env = gym.wrappers.TransformObservation(
        env,
        lambda obs: np.clip(
            obs,
            -10,
            10,
        ),
        env.observation_space,
    )

    return env


# ============================================================
# Wrapper finder
# ============================================================

def find_wrapper(
    env,
    cls,
):

    w = env

    while isinstance(
        w,
        gym.Wrapper,
    ):

        if isinstance(
            w,
            cls,
        ):
            return w

        w = w.env

    return None


# ============================================================
# Freeze observation normalization
# ============================================================

def load_and_freeze_obs_rms(
    env,
    obs_rms_path,
):

    norm_w = find_wrapper(
        env,
        gym.wrappers.NormalizeObservation,
    )

    if norm_w is None:
        raise RuntimeError(
            "NormalizeObservation wrapper not found."
        )

    data = np.load(
        obs_rms_path
    )

    norm_w.obs_rms.mean = (
        data["mean"].copy()
    )

    norm_w.obs_rms.var = (
        data["var"].copy()
    )

    norm_w.obs_rms.count = float(
        data["count"]
    )

    # Gymnasium normalization wrapper
    # updates stats by default.
    # Freeze it for evaluation.
    if hasattr(
        norm_w,
        "update_running_mean",
    ):
        norm_w.update_running_mean = False

    print(
        "Loaded observation normalization:"
    )

    print(
        " mean shape:",
        norm_w.obs_rms.mean.shape,
    )

    print(
        " var shape:",
        norm_w.obs_rms.var.shape,
    )

    print(
        " count:",
        norm_w.obs_rms.count,
    )


# ============================================================
# Helper: get physical distance
# ============================================================

def get_distance(env):

    base_env = env.unwrapped

    achieved_goal = np.asarray(
        base_env.task.get_achieved_goal(),
        dtype=np.float32,
    )

    desired_goal = np.asarray(
        base_env.task.get_goal(),
        dtype=np.float32,
    )

    distance = np.linalg.norm(
        achieved_goal
        - desired_goal
    )

    return float(distance)


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    print(
        "=" * 72
    )

    print(
        "SAC PANDA REACH EVALUATION"
    )

    print(
        "=" * 72
    )

    print(
        "Actor:",
        ACTOR_PATH,
    )

    print(
        "Obs RMS:",
        OBS_RMS_PATH,
    )

    print(
        "Threshold:",
        DISTANCE_THRESHOLD,
    )

    print(
        "Episodes:",
        N_EPISODES,
    )

    print(
        "=" * 72
    )


    # ========================================================
    # File checks
    # ========================================================

    if not os.path.exists(
        ACTOR_PATH
    ):
        raise FileNotFoundError(
            f"Actor not found:\n"
            f"{ACTOR_PATH}"
        )

    if not os.path.exists(
        OBS_RMS_PATH
    ):
        raise FileNotFoundError(
            f"obs_rms not found:\n"
            f"{OBS_RMS_PATH}"
        )


    # ========================================================
    # Dummy vector env used only to construct Actor
    # ========================================================

    dummy_envs = gym.vector.SyncVectorEnv(
        [
            lambda: build_env(
                record_video=False,
                show_live=False,
            )
        ]
    )


    print(
        "\nNetwork:"
    )

    print(
        "Observation space:",
        dummy_envs.single_observation_space,
    )

    print(
        "Action space:",
        dummy_envs.single_action_space,
    )


    if (
        dummy_envs.single_action_space.shape
        != (7,)
    ):
        raise RuntimeError(
            "Expected 7D joint action space, "
            f"got "
            f"{dummy_envs.single_action_space.shape}"
        )


    if not np.all(
        np.isfinite(
            dummy_envs.single_action_space.low
        )
    ):
        raise RuntimeError(
            "Action lower bounds are not finite."
        )


    if not np.all(
        np.isfinite(
            dummy_envs.single_action_space.high
        )
    ):
        raise RuntimeError(
            "Action upper bounds are not finite."
        )


    print(
        "Joint action dimension: 7 ✓"
    )


    # ========================================================
    # Load Actor
    # ========================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Device:",
        device,
    )


    actor = Actor(
        dummy_envs
    ).to(device)


    state_dict = torch.load(
        ACTOR_PATH,
        map_location=device,
    )


    actor.load_state_dict(
        state_dict
    )

    actor.eval()


    print(
        "Actor loaded successfully."
    )


    dummy_envs.close()


    # ========================================================
    # Evaluation env
    # ========================================================

    env = build_env(
        record_video=RECORD_VIDEO,
        show_live=SHOW_LIVE,
    )


    load_and_freeze_obs_rms(
        env,
        OBS_RMS_PATH,
    )


    # ========================================================
    # Results
    # ========================================================

    results = []

    success_count = 0


    # ========================================================
    # Evaluation loop
    # ========================================================

    for episode in range(
        N_EPISODES
    ):

        # Same deterministic seeds for fair comparison
        seed = episode

        obs, info = env.reset(
            seed=seed
        )


        initial_distance = get_distance(
            env
        )

        minimum_distance = (
            initial_distance
        )

        episode_return = 0.0

        episode_steps = 0


        if DEBUG:

            print(
                "\n"
                + "-" * 72
            )

            print(
                f"Episode {episode + 1}/{N_EPISODES}"
            )

            print(
                f"Seed: {seed}"
            )

            print(
                "Initial distance: "
                f"{initial_distance * 100:.2f} cm"
            )


        terminated = False
        truncated = False


        while not (
            terminated
            or truncated
        ):

            obs_tensor = torch.as_tensor(
                obs,
                dtype=torch.float32,
                device=device,
            ).unsqueeze(0)


            with torch.no_grad():

                if DETERMINISTIC:

                    # ------------------------------------------------
                    # SAC deterministic evaluation:
                    # tanh(mean) rescaled to action bounds
                    # ------------------------------------------------

                    mean, _ = actor(
                        obs_tensor
                    )

                    action = (
                        torch.tanh(mean)
                        * actor.action_scale
                        + actor.action_bias
                    )

                else:

                    action, _, _ = (
                        actor.get_action(
                            obs_tensor
                        )
                    )


            action = (
                action
                .cpu()
                .numpy()[0]
            )


            (
                obs,
                reward,
                terminated,
                truncated,
                info,
            ) = env.step(
                action
            )


            episode_return += float(
                reward
            )

            episode_steps += 1


            current_distance = (
                get_distance(env)
            )


            minimum_distance = min(
                minimum_distance,
                current_distance,
            )


            if DEBUG:

                print(
                    f"  step="
                    f"{episode_steps:02d} "
                    f"distance="
                    f"{current_distance * 100:.2f} cm "
                    f"reward="
                    f"{float(reward):.4f}"
                )


        # ====================================================
        # Episode result
        # ====================================================

        final_distance = get_distance(
            env
        )


        success = (
            minimum_distance
            < DISTANCE_THRESHOLD
        )


        if success:
            success_count += 1


        results.append(
            {
                "episode": episode + 1,
                "seed": seed,
                "success": int(success),
                "initial_distance_cm":
                    initial_distance * 100,
                "final_distance_cm":
                    final_distance * 100,
                "minimum_distance_cm":
                    minimum_distance * 100,
                "return":
                    episode_return,
                "steps":
                    episode_steps,
            }
        )


        print(
            f"\nEpisode {episode + 1}: "
            f"{'SUCCESS' if success else 'FAIL'} | "
            f"initial="
            f"{initial_distance * 100:.2f} cm | "
            f"final="
            f"{final_distance * 100:.2f} cm | "
            f"min="
            f"{minimum_distance * 100:.2f} cm | "
            f"steps="
            f"{episode_steps} | "
            f"return="
            f"{episode_return:.4f}"
        )


    # ========================================================
    # Close env
    # ========================================================

    env.close()


    # ========================================================
    # Save CSV
    # ========================================================

    with open(
        RESULT_CSV,
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "episode",
                "seed",
                "success",
                "initial_distance_cm",
                "final_distance_cm",
                "minimum_distance_cm",
                "return",
                "steps",
            ],
        )

        writer.writeheader()

        writer.writerows(
            results
        )


    # ========================================================
    # Statistics
    # ========================================================

    initial_errors = np.array(
        [
            r["initial_distance_cm"]
            for r in results
        ]
    )


    final_errors = np.array(
        [
            r["final_distance_cm"]
            for r in results
        ]
    )


    minimum_errors = np.array(
        [
            r["minimum_distance_cm"]
            for r in results
        ]
    )


    returns = np.array(
        [
            r["return"]
            for r in results
        ]
    )


    steps = np.array(
        [
            r["steps"]
            for r in results
        ]
    )


    success_rate = (
        success_count
        / N_EPISODES
        * 100.0
    )


    # ========================================================
    # Summary
    # ========================================================

    print(
        "\n"
        + "=" * 72
    )

    print(
        "EVALUATION SUMMARY"
    )

    print(
        "=" * 72
    )

    print(
        f"Episodes:                  "
        f"{N_EPISODES}"
    )

    print(
        f"Success threshold:         "
        f"{DISTANCE_THRESHOLD * 100:.1f} cm"
    )

    print(
        f"Success rate:              "
        f"{success_rate:.1f}% "
        f"({success_count}/{N_EPISODES})"
    )

    print(
        f"Mean initial error:        "
        f"{initial_errors.mean():.2f} cm"
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
        f"{minimum_errors.mean():.2f} cm"
    )

    print(
        f"Median minimum error:      "
        f"{np.median(minimum_errors):.2f} cm"
    )

    print(
        f"Best minimum error:        "
        f"{minimum_errors.min():.2f} cm"
    )

    print(
        f"Std final error:           "
        f"{final_errors.std():.2f} cm"
    )

    print(
        f"Mean episode return:       "
        f"{returns.mean():.3f}"
    )

    print(
        f"Mean episode steps:        "
        f"{steps.mean():.1f}"
    )

    print(
        f"Median episode steps:      "
        f"{np.median(steps):.1f}"
    )

    print(
        "=" * 72
    )


    print(
        "\nResults CSV:"
    )

    print(
        RESULT_CSV
    )


    if RECORD_VIDEO:

        print(
            "\nVideos:"
        )

        print(
            VIDEO_DIR
        )