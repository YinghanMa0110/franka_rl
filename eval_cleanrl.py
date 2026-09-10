import os
import sys
import time

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
    sys.path.insert(
        0,
        CLEANRL_DIR,
    )


# ============================================================
# Import training definitions
# ============================================================

from ppo_cont_2 import (  # noqa: E402
    Agent,
    Args,
    make_env,
)


# ============================================================
# Configuration
# ============================================================

ENV_ID = "PandaReach-v3"

REWARD_TYPE = "dense"

CONTROL_TYPE = "joints"

DISTANCE_THRESHOLD = 0.02  # 2 cm


N_EPISODES = 20

DETERMINISTIC = True

SEED = 0

RENDER = True

STEP_SLEEP = 0.05

PAUSE_BETWEEN_EPISODES = False

DEBUG = True


# ============================================================
# Wrapper utility
# ============================================================

def find_wrapper(env, cls):
    """
    Search through Gymnasium wrappers and return
    the first wrapper matching cls.
    """

    w = env

    while isinstance(w, gym.Wrapper):

        if isinstance(w, cls):
            return w

        w = w.env

    return None


# ============================================================
# Render environment
# ============================================================

def make_render_env(
    env_id,
):
    """
    Reconstruct the same environment used during training,
    but with human rendering enabled.

    IMPORTANT:
    Wrapper order must match training.
    """

    env = gym.make(
        env_id,
        reward_type=REWARD_TYPE,
        control_type=CONTROL_TYPE,
        render_mode="human",
    )

    # --------------------------------------------------------
    # Use the SAME success threshold as training
    # --------------------------------------------------------

    env.unwrapped.task.distance_threshold = (
        DISTANCE_THRESHOLD
    )

    print(
        "Evaluation distance threshold:",
        env.unwrapped.task.distance_threshold,
    )

    # --------------------------------------------------------
    # Same wrappers as training
    # --------------------------------------------------------

    env = gym.wrappers.FlattenObservation(
        env
    )

    env = gym.wrappers.RecordEpisodeStatistics(
        env
    )

    env = gym.wrappers.ClipAction(
        env
    )

    env = gym.wrappers.NormalizeObservation(
        env
    )

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
# Load observation normalization
# ============================================================

def load_and_freeze_obs_rms(
    env,
    obs_rms_path,
):
    """
    Load the observation running mean/variance from training.

    This is essential because the policy was trained with
    NormalizeObservation.
    """

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

    # --------------------------------------------------------
    # Freeze running statistics during evaluation
    # --------------------------------------------------------

    if hasattr(
        norm_w,
        "update_running_mean",
    ):

        norm_w.update_running_mean = False

    else:

        # Fallback for versions where the wrapper
        # has no explicit freeze flag.
        norm_w.obs_rms.update = (
            lambda x: None
        )

    print(
        "Loaded obs_rms:"
    )

    print(
        "  mean =",
        np.round(
            data["mean"],
            4,
        ),
    )

    print(
        "  var  =",
        np.round(
            data["var"],
            4,
        ),
    )

    print(
        "  count =",
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
    env_id=ENV_ID,
    n_episodes=N_EPISODES,
    deterministic=DETERMINISTIC,
    seed=SEED,
    render=RENDER,
    sleep=STEP_SLEEP,
    pause_between_episodes=PAUSE_BETWEEN_EPISODES,
    debug=DEBUG,
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
        f"Environment:         {env_id}"
    )

    print(
        f"Reward type:         {REWARD_TYPE}"
    )

    print(
        f"Control type:        {CONTROL_TYPE}"
    )

    print(
        f"Success threshold:   "
        f"{DISTANCE_THRESHOLD * 100:.1f} cm"
    )

    print(
        f"Model:               {model_path}"
    )

    print(
        f"Observation stats:   {obs_rms_path}"
    )

    print(
        f"Episodes:            {n_episodes}"
    )

    print(
        f"Deterministic:       {deterministic}"
    )

    print(
        "=" * 72
    )


    # ========================================================
    # Create dummy env to reconstruct Agent dimensions
    # ========================================================

    args = Args()

    args.env_id = env_id


    dummy = gym.vector.SyncVectorEnv(
        [
            make_env(
                env_id,
                0,
                False,
                "eval_dummy",
                args.gamma,
            )
        ]
    )


    print(
        "\nNetwork dimensions:"
    )

    print(
        "  observation space:",
        dummy.single_observation_space,
    )

    print(
        "  action space:",
        dummy.single_action_space,
    )


    # --------------------------------------------------------
    # Joint control should give 7D action space
    # --------------------------------------------------------

    action_shape = (
        dummy.single_action_space.shape
    )

    if action_shape != (7,):

        print(
            "\nWARNING:"
        )

        print(
            f"Expected joint-control action shape (7,), "
            f"but got {action_shape}."
        )

        print(
            "Check that make_env() uses "
            "control_type='joints'."
        )


    # ========================================================
    # Reconstruct neural network
    # ========================================================

    agent = Agent(
        dummy
    )


    # ========================================================
    # Load trained policy
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
        "\nModel loaded successfully."
    )


    # ========================================================
    # Create evaluation environment
    # ========================================================

    if render:

        env = make_render_env(
            env_id
        )

    else:

        env = make_env(
            env_id,
            0,
            False,
            "eval",
            args.gamma,
        )()


        # Very important:
        # make_env must also have threshold=0.02.
        #
        # This line makes evaluation robust even if you
        # accidentally forgot it inside make_env().
        env.unwrapped.task.distance_threshold = (
            DISTANCE_THRESHOLD
        )


    # ========================================================
    # Restore normalization statistics
    # ========================================================

    load_and_freeze_obs_rms(
        env,
        obs_rms_path,
    )


    # ========================================================
    # Debug task / robot
    # ========================================================

    if debug:

        base = env.unwrapped

        print(
            "\n--- Environment check ---"
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
            "goal range low =",
            getattr(
                base.task,
                "goal_range_low",
                None,
            ),
        )

        print(
            "goal range high =",
            getattr(
                base.task,
                "goal_range_high",
                None,
            ),
        )

        print()


    # ========================================================
    # Storage
    # ========================================================

    successes = []

    returns = []

    final_distances = []

    episode_steps = []


    # ========================================================
    # Evaluation loop
    # ========================================================

    for ep in range(
        n_episodes
    ):

        obs, info = env.reset(
            seed=seed + ep
        )

        done = False

        ep_ret = 0.0

        ever_success = False

        n_step = 0


        # ----------------------------------------------------
        # Initial state
        # ----------------------------------------------------

        base = env.unwrapped

        initial_goal = np.asarray(
            base.task.get_goal(),
            dtype=np.float32,
        )

        initial_ee = np.asarray(
            base.task.get_achieved_goal(),
            dtype=np.float32,
        )

        initial_distance = float(
            np.linalg.norm(
                initial_ee
                - initial_goal
            )
        )


        print(
            "\n" + "-" * 72
        )

        print(
            f"Episode {ep + 1}/{n_episodes}"
        )

        print(
            f"Initial EE:    "
            f"{np.round(initial_ee, 3)}"
        )

        print(
            f"Goal:          "
            f"{np.round(initial_goal, 3)}"
        )

        print(
            f"Initial error: "
            f"{initial_distance * 100:.2f} cm"
        )


        # ====================================================
        # Episode interaction loop
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


                if deterministic:

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
            # Debug joint action
            # ------------------------------------------------

            if (
                debug
                and n_step % 10 == 0
            ):

                clipped_action = np.clip(
                    raw_action,
                    -1.0,
                    1.0,
                )

                print(
                    f"\nAction step {n_step:3d}"
                )

                print(
                    "  raw joint action:",
                    np.round(
                        raw_action,
                        3,
                    ),
                )

                print(
                    "  clipped action:",
                    np.round(
                        clipped_action,
                        3,
                    ),
                )


            # ------------------------------------------------
            # Environment step
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


            # ------------------------------------------------
            # Success
            # ------------------------------------------------

            if bool(
                info.get(
                    "is_success",
                    False,
                )
            ):

                ever_success = True


            # ------------------------------------------------
            # Position debugging
            # ------------------------------------------------

            if (
                debug
                and n_step % 10 == 0
            ):

                base = env.unwrapped


                goal = np.asarray(
                    base.task.get_goal(),
                    dtype=np.float32,
                )

                achieved = np.asarray(
                    base.task.get_achieved_goal(),
                    dtype=np.float32,
                )

                distance = float(
                    np.linalg.norm(
                        achieved
                        - goal
                    )
                )


                try:

                    ee = np.asarray(
                        base.robot.get_ee_position(),
                        dtype=np.float32,
                    )

                except Exception:

                    ee = achieved


                print(
                    f"  step {n_step:3d}"
                )

                print(
                    f"  goal:     "
                    f"{np.round(goal, 3)}"
                )

                print(
                    f"  achieved: "
                    f"{np.round(achieved, 3)}"
                )

                print(
                    f"  EE:       "
                    f"{np.round(ee, 3)}"
                )

                print(
                    f"  distance: "
                    f"{distance * 100:.2f} cm"
                )


            # ------------------------------------------------
            # Render delay
            # ------------------------------------------------

            if (
                render
                and sleep > 0
            ):

                time.sleep(
                    sleep
                )


            done = bool(
                terminated
                or truncated
            )


        # ====================================================
        # End-of-episode metrics
        # ====================================================

        base = env.unwrapped


        goal = np.asarray(
            base.task.get_goal(),
            dtype=np.float32,
        )

        achieved = np.asarray(
            base.task.get_achieved_goal(),
            dtype=np.float32,
        )


        try:

            ee = np.asarray(
                base.robot.get_ee_position(),
                dtype=np.float32,
            )

        except Exception:

            ee = achieved


        final_distance = float(
            np.linalg.norm(
                achieved
                - goal
            )
        )


        # ----------------------------------------------------
        # Save metrics
        # ----------------------------------------------------

        successes.append(
            float(
                ever_success
            )
        )

        returns.append(
            float(
                ep_ret
            )
        )

        final_distances.append(
            final_distance
        )

        episode_steps.append(
            n_step
        )


        # ----------------------------------------------------
        # Episode output
        # ----------------------------------------------------

        print(
            "\nEpisode result:"
        )

        print(
            f"  success:     "
            f"{ever_success}"
        )

        print(
            f"  return:      "
            f"{ep_ret:.3f}"
        )

        print(
            f"  steps:       "
            f"{n_step}"
        )

        print(
            f"  goal:        "
            f"{np.round(goal, 3)}"
        )

        print(
            f"  final EE:    "
            f"{np.round(ee, 3)}"
        )

        print(
            f"  final error: "
            f"{final_distance * 100:.2f} cm"
        )


        if pause_between_episodes:

            input(
                "\nPress Enter for next episode..."
            )


    # ========================================================
    # Close env
    # ========================================================

    env.close()


    # ========================================================
    # Summary statistics
    # ========================================================

    success_rate = float(
        np.mean(
            successes
        )
        * 100.0
    )

    mean_return = float(
        np.mean(
            returns
        )
    )

    mean_distance = float(
        np.mean(
            final_distances
        )
    )

    median_distance = float(
        np.median(
            final_distances
        )
    )

    std_distance = float(
        np.std(
            final_distances
        )
    )

    mean_steps = float(
        np.mean(
            episode_steps
        )
    )

    median_steps = float(
        np.median(
            episode_steps
        )
    )


    # ========================================================
    # Final output
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
        f"Control type:             "
        f"{CONTROL_TYPE}"
    )

    print(
        f"Success threshold:        "
        f"{DISTANCE_THRESHOLD * 100:.1f} cm"
    )

    print(
        f"Success rate:             "
        f"{success_rate:.1f}% "
        f"({int(sum(successes))}/{n_episodes})"
    )

    print(
        f"Mean episode return:      "
        f"{mean_return:.3f}"
    )

    print(
        f"Mean final EE-goal error: "
        f"{mean_distance * 100:.2f} cm"
    )

    print(
        f"Median final error:       "
        f"{median_distance * 100:.2f} cm"
    )

    print(
        f"Std final error:          "
        f"{std_distance * 100:.2f} cm"
    )

    print(
        f"Mean episode steps:       "
        f"{mean_steps:.1f}"
    )

    print(
        f"Median episode steps:     "
        f"{median_steps:.1f}"
    )

    print(
        "=" * 72
    )


    return {
        "success_rate": success_rate,
        "mean_return": mean_return,
        "mean_final_distance_cm": (
            mean_distance * 100
        ),
        "median_final_distance_cm": (
            median_distance * 100
        ),
        "std_final_distance_cm": (
            std_distance * 100
        ),
        "mean_steps": mean_steps,
        "median_steps": median_steps,
    }


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # CHANGE THIS after your new 2 cm joint-control run
    # --------------------------------------------------------

    RUN = os.path.join(
        ROOT,
        "training",
        "cleanrl",
        "runs",
        "PandaReach-v3__ppo_cont_2__1__XXXXXXXXXX",
    )


    model_path = os.path.join(
        RUN,
        "ppo_cont_2.cleanrl_model",
    )


    obs_rms_path = os.path.join(
        RUN,
        "ppo_cont_2.obs_rms.npz",
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
        model_path=model_path,
        obs_rms_path=obs_rms_path,
        env_id=ENV_ID,
        n_episodes=N_EPISODES,
        deterministic=DETERMINISTIC,
        render=RENDER,
        sleep=STEP_SLEEP,
        pause_between_episodes=PAUSE_BETWEEN_EPISODES,
        debug=DEBUG,
    )
