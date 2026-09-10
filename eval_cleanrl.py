import os
import sys
import time
import csv

import gymnasium as gym
import numpy as np
import panda_gym  # noqa: F401
import torch


# ============================================================
# Path setup
# ============================================================

ROOT = os.path.dirname(os.path.abspath(__file__))

CLEANRL_DIR = os.path.join(
    ROOT,
    "training",
    "cleanrl",
)

if CLEANRL_DIR not in sys.path:
    sys.path.insert(0, CLEANRL_DIR)


# ============================================================
# Import EXACT Agent used for training
# ============================================================

from ppo_continuous_action import Agent  # noqa: E402


# ============================================================
# Configuration
# ============================================================

ENV_ID = "PandaReach-v3"

REWARD_TYPE = "dense"

CONTROL_TYPE = "joints"

# Must match training
DISTANCE_THRESHOLD = 0.02  # 2 cm


# ------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------

N_EPISODES = 20

DETERMINISTIC = True

SEED = 0

DEBUG = True

STEP_SLEEP = 0.00


# ------------------------------------------------------------
# Video
#
# RECORD_VIDEO=True:
#   Save MP4 evaluation videos.
#
# SHOW_LIVE=True:
#   Show PyBullet window live.
#
# You normally cannot use both modes with the same env because
# Gymnasium render_mode is selected when creating the env.
# ------------------------------------------------------------

RECORD_VIDEO = True

SHOW_LIVE = False

VIDEO_DIR = os.path.join(
    ROOT,
    "videos",
    "reach_joint_2cm_eval",
)

RESULT_CSV = os.path.join(
    ROOT,
    "results",
    "reach_joint_2cm_eval.csv",
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
# Build evaluation environment
# ============================================================

def build_env(
    record_video=False,
    show_live=False,
):
    """
    Build Reach evaluation environment.

    panda-gym version used here requires render_mode
    to be either "rgb_array" or "human".
    """

    if record_video and show_live:
        raise ValueError(
            "RECORD_VIDEO and SHOW_LIVE cannot both be True."
        )

    # IMPORTANT:
    # Always define render_mode
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

    # Same 2 cm threshold as training
    env.unwrapped.task.distance_threshold = DISTANCE_THRESHOLD

    # Video recording
    if record_video:
        os.makedirs(
            VIDEO_DIR,
            exist_ok=True,
        )

        env = gym.wrappers.RecordVideo(
            env,
            video_folder=VIDEO_DIR,
            episode_trigger=lambda episode_id: True,
            name_prefix="ppo_reach_joint_2cm",
        )

    # Same wrappers as training
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


# ============================================================
# Load observation normalization
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

    norm_w.obs_rms.count = (
        data["count"].copy()
    )


    # ========================================================
    # Freeze stats during evaluation
    # ========================================================

    if hasattr(
        norm_w,
        "update_running_mean",
    ):

        norm_w.update_running_mean = False

    else:

        norm_w.obs_rms.update = (
            lambda x: None
        )


    print(
        "\nLoaded training observation normalization:"
    )

    print(
        "mean:",
        np.round(
            data["mean"],
            4,
        ),
    )

    print(
        "var:",
        np.round(
            data["var"],
            4,
        ),
    )

    print(
        "count:",
        float(
            np.asarray(
                data["count"]
            )
        ),
    )

    print(
        "Observation normalization frozen."
    )


# ============================================================
# Evaluation
# ============================================================

def evaluate(
    model_path,
    obs_rms_path,
):


    print(
        "=" * 72
    )

    print(
        "PPO REACH JOINT-CONTROL EVALUATION"
    )

    print(
        "=" * 72
    )

    print(
        f"Environment:        {ENV_ID}"
    )

    print(
        f"Reward:             {REWARD_TYPE}"
    )

    print(
        f"Control:            {CONTROL_TYPE}"
    )

    print(
        f"Success threshold:  "
        f"{DISTANCE_THRESHOLD * 100:.1f} cm"
    )

    print(
        f"Episodes:           {N_EPISODES}"
    )

    print(
        f"Deterministic:      {DETERMINISTIC}"
    )

    print(
        f"Record video:       {RECORD_VIDEO}"
    )

    if RECORD_VIDEO:

        print(
            f"Video folder:       {VIDEO_DIR}"
        )

    print(
        "=" * 72
    )


    # ========================================================
    # Dummy vector env
    #
    # Needed only to reconstruct Agent dimensions
    # ========================================================

    dummy = gym.vector.SyncVectorEnv(
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
        dummy.single_observation_space,
    )

    print(
        "Action space:",
        dummy.single_action_space,
    )


    # ========================================================
    # Must be 7 joint actions
    # ========================================================

    action_shape = (
        dummy.single_action_space.shape
    )


    if action_shape != (7,):

        raise RuntimeError(
            f"Expected joint action shape (7,), "
            f"but got {action_shape}."
        )


    print(
        "Joint action dimension: 7 ✓"
    )


    # ========================================================
    # Construct network
    # ========================================================

    agent = Agent(
        dummy
    )


    # ========================================================
    # Load policy
    # ========================================================

    state_dict = torch.load(
        model_path,
        map_location="cpu",
    )


    agent.load_state_dict(
        state_dict
    )


    agent.eval()


    dummy.close()


    print(
        "Model loaded successfully."
    )


    # ========================================================
    # Evaluation environment
    # ========================================================

    env = build_env(
        record_video=RECORD_VIDEO,
        show_live=SHOW_LIVE,
    )


    # ========================================================
    # Restore training normalization
    # ========================================================

    load_and_freeze_obs_rms(
        env,
        obs_rms_path,
    )


    # ========================================================
    # Sanity check
    # ========================================================

    base = env.unwrapped


    print(
        "\nEnvironment check:"
    )

    print(
        "distance_threshold =",
        base.task.distance_threshold,
    )

    print(
        "control_type =",
        getattr(
            base.robot,
            "control_type",
            "unknown",
        ),
    )

    print(
        "goal_range_low =",
        base.task.goal_range_low,
    )

    print(
        "goal_range_high =",
        base.task.goal_range_high,
    )


    # ========================================================
    # Results
    # ========================================================

    successes = []

    returns = []

    initial_distances = []

    final_distances = []

    minimum_distances = []

    episode_steps = []

    rows = []


    # ========================================================
    # Episode loop
    # ========================================================

    for ep in range(
        N_EPISODES
    ):


        obs, info = env.reset(
            seed=SEED + ep
        )


        done = False

        ep_ret = 0.0

        ever_success = False

        n_step = 0


        # ====================================================
        # Initial state
        # ====================================================

        base = env.unwrapped


        goal = np.asarray(
            base.task.get_goal(),
            dtype=np.float32,
        )


        initial_ee = np.asarray(
            base.task.get_achieved_goal(),
            dtype=np.float32,
        )


        initial_distance = float(
            np.linalg.norm(
                initial_ee - goal
            )
        )


        min_distance = (
            initial_distance
        )


        print(
            "\n" + "-" * 72
        )

        print(
            f"Episode {ep + 1}/{N_EPISODES}"
        )

        print(
            "Initial EE:",
            np.round(
                initial_ee,
                3,
            ),
        )

        print(
            "Goal:",
            np.round(
                goal,
                3,
            ),
        )

        print(
            f"Initial error: "
            f"{initial_distance * 100:.2f} cm"
        )


        # ====================================================
        # Interaction loop
        # ====================================================

        while not done:


            # ------------------------------------------------
            # Policy inference
            # ------------------------------------------------

            with torch.no_grad():


                x = torch.tensor(
                    obs,
                    dtype=torch.float32,
                ).unsqueeze(0)


                if DETERMINISTIC:

                    action = (
                        agent.actor_mean(x)
                    )

                else:

                    action, _, _, _ = (
                        agent.get_action_and_value(
                            x
                        )
                    )


            raw_action = (
                action
                .cpu()
                .numpy()[0]
            )


            # ------------------------------------------------
            # Debug
            # ------------------------------------------------

            if (
                DEBUG
                and n_step % 10 == 0
            ):

                clipped = np.clip(
                    raw_action,
                    -1.0,
                    1.0,
                )


                print(
                    f"\nAction step {n_step:3d}"
                )

                print(
                    "raw:",
                    np.round(
                        raw_action,
                        3,
                    ),
                )

                print(
                    "clip:",
                    np.round(
                        clipped,
                        3,
                    ),
                )


            # ------------------------------------------------
            # Step environment
            # ------------------------------------------------

            obs, reward, terminated, truncated, info = (
                env.step(
                    raw_action
                )
            )


            ep_ret += float(
                reward
            )


            n_step += 1


            # =================================================
            # Current distance
            # =================================================

            base = env.unwrapped


            current_goal = np.asarray(
                base.task.get_goal(),
                dtype=np.float32,
            )


            current_ee = np.asarray(
                base.task.get_achieved_goal(),
                dtype=np.float32,
            )


            current_distance = float(
                np.linalg.norm(
                    current_ee
                    - current_goal
                )
            )


            # =================================================
            # Minimum distance reached
            # =================================================

            min_distance = min(
                min_distance,
                current_distance,
            )


            # =================================================
            # Success
            # =================================================

            if bool(
                info.get(
                    "is_success",
                    False,
                )
            ):

                ever_success = True


            # =================================================
            # Debug position
            # =================================================

            if (
                DEBUG
                and n_step % 10 == 0
            ):

                print(
                    f"step {n_step:3d}: "
                    f"EE={np.round(current_ee, 3)} "
                    f"goal={np.round(current_goal, 3)} "
                    f"error={current_distance * 100:.2f} cm "
                    f"min={min_distance * 100:.2f} cm"
                )


            # ------------------------------------------------
            # Optional delay
            # ------------------------------------------------

            if STEP_SLEEP > 0:

                time.sleep(
                    STEP_SLEEP
                )


            done = bool(
                terminated
                or truncated
            )


        # ====================================================
        # Final state
        # ====================================================

        base = env.unwrapped


        final_goal = np.asarray(
            base.task.get_goal(),
            dtype=np.float32,
        )


        final_ee = np.asarray(
            base.task.get_achieved_goal(),
            dtype=np.float32,
        )


        final_distance = float(
            np.linalg.norm(
                final_ee
                - final_goal
            )
        )


        # ====================================================
        # Save metrics
        # ====================================================

        successes.append(
            float(
                ever_success
            )
        )


        returns.append(
            ep_ret
        )


        initial_distances.append(
            initial_distance
        )


        final_distances.append(
            final_distance
        )


        minimum_distances.append(
            min_distance
        )


        episode_steps.append(
            n_step
        )


        rows.append(
            {
                "episode": ep + 1,

                "success":
                    int(ever_success),

                "initial_distance_cm":
                    initial_distance * 100,

                "final_distance_cm":
                    final_distance * 100,

                "minimum_distance_cm":
                    min_distance * 100,

                "return":
                    ep_ret,

                "steps":
                    n_step,
            }
        )


        # ====================================================
        # Episode result
        # ====================================================

        print(
            "\nEpisode result:"
        )


        print(
            f"success:       "
            f"{ever_success}"
        )


        print(
            f"return:        "
            f"{ep_ret:.3f}"
        )


        print(
            f"steps:         "
            f"{n_step}"
        )


        print(
            f"initial error: "
            f"{initial_distance * 100:.2f} cm"
        )


        print(
            f"final error:   "
            f"{final_distance * 100:.2f} cm"
        )


        print(
            f"minimum error: "
            f"{min_distance * 100:.2f} cm"
        )


    # ========================================================
    # Closing env finalises MP4 videos
    # ========================================================

    env.close()


    # ========================================================
    # Save CSV
    # ========================================================

    os.makedirs(
        os.path.dirname(
            RESULT_CSV
        ),
        exist_ok=True,
    )


    with open(
        RESULT_CSV,
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=rows[0].keys(),
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


    # ========================================================
    # Convert metrics to arrays
    # ========================================================

    successes = np.asarray(
        successes
    )

    returns = np.asarray(
        returns
    )

    initial_distances = np.asarray(
        initial_distances
    )

    final_distances = np.asarray(
        final_distances
    )

    minimum_distances = np.asarray(
        minimum_distances
    )

    episode_steps = np.asarray(
        episode_steps
    )


    # ========================================================
    # Statistics
    # ========================================================

    success_rate = (
        np.mean(successes)
        * 100
    )


    # ========================================================
    # Final report
    # ========================================================

    print(
        "\n" + "=" * 72
    )

    print(
        "EVALUATION RESULTS"
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
        f"({int(successes.sum())}/{N_EPISODES})"
    )


    print(
        f"Mean initial error:        "
        f"{initial_distances.mean() * 100:.2f} cm"
    )


    print(
        f"Mean final error:          "
        f"{final_distances.mean() * 100:.2f} cm"
    )


    print(
        f"Median final error:        "
        f"{np.median(final_distances) * 100:.2f} cm"
    )


    print(
        f"Mean minimum error:        "
        f"{minimum_distances.mean() * 100:.2f} cm"
    )


    print(
        f"Median minimum error:      "
        f"{np.median(minimum_distances) * 100:.2f} cm"
    )


    print(
        f"Best minimum error:        "
        f"{minimum_distances.min() * 100:.2f} cm"
    )


    print(
        f"Std final error:           "
        f"{final_distances.std() * 100:.2f} cm"
    )


    print(
        f"Mean episode return:       "
        f"{returns.mean():.3f}"
    )


    print(
        f"Mean episode steps:        "
        f"{episode_steps.mean():.1f}"
    )


    print(
        f"Median episode steps:      "
        f"{np.median(episode_steps):.1f}"
    )


    print(
        "=" * 72
    )


    print(
        f"\nResults CSV:\n{RESULT_CSV}"
    )


    if RECORD_VIDEO:

        print(
            f"\nVideos saved to:\n{VIDEO_DIR}"
        )


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":


    RUN = os.path.join(
        ROOT,
        "runs",
        "PandaReach-v3__ppo_continuous_action__1__1789070756",
    )


    model_path = os.path.join(
        RUN,
        "ppo_continuous_action.cleanrl_model",
    )


    obs_rms_path = os.path.join(
        RUN,
        "ppo_continuous_action.obs_rms.npz",
    )


    print(
        f"Using run: {RUN}"
    )


    if not os.path.exists(
        model_path
    ):

        raise FileNotFoundError(
            f"Model not found:\n"
            f"{model_path}"
        )


    if not os.path.exists(
        obs_rms_path
    ):

        raise FileNotFoundError(
            f"obs_rms not found:\n"
            f"{obs_rms_path}"
        )


    evaluate(
        model_path,
        obs_rms_path,
    )