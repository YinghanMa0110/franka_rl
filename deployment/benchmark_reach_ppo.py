"""
Franka RL Deployment Benchmark - Cartesian Impedance Controller

Deploys a PandaReach PPO policy on the real Franka using panda-py's
CartesianImpedance controller.

Automatically records:
- Success / failure
- Final distance error
- Episode steps
- Episode runtime
- EE trajectory length
- Action smoothness
- Step-level trajectory data
- Panda raw log

Designed for fair real-robot algorithm comparison:
PPO / SAC / TD3 / custom algorithms should use the same
goals, controller parameters, success threshold, and evaluation settings.
"""

import os
import csv
import time
import logging
from datetime import datetime

import numpy as np
import panda_py
from panda_py import controllers

from stable_baselines3 import PPO


logging.basicConfig(level=logging.INFO)


# ============================================================
# Configuration
# ============================================================

HOSTNAME = "192.168.1.8"

CHECKPOINT_PATH = "checkpoints/panda_reach_ppo_40000_steps"


# panda-gym base offset for coordinate conversion
BASE_OFFSET = np.array([
    -0.6,
    0.0,
    0.0
])


# ============================================================
# Deployment Parameters
# ============================================================

ACTION_SCALE = 0.05

MAX_STEP = 0.03

CONTROL_FREQ = 20

MAX_RUNTIME = 30

GOAL_THRESHOLD = 0.05


# Set True for policy inference without robot movement
DRY_RUN = False


# ============================================================
# Logging
# ============================================================

RESULT_DIR = "results"

RUN_DIR = os.path.join(
    RESULT_DIR,
    "runs"
)

EPISODE_SUMMARY_FILE = os.path.join(
    RESULT_DIR,
    "episodes.csv"
)


# ============================================================
# Fixed Benchmark Goals
#
# IMPORTANT:
# Keep these EXACTLY the same when comparing:
# PPO / SAC / TD3 / custom algorithms
#
# Units: metres
# Relative to move_to_start() EE position
# ============================================================

GOAL_OFFSETS = np.array([

    # ---------- X axis ----------
    [ 0.05,  0.00,  0.00],
    [ 0.08,  0.00,  0.00],
    [-0.05,  0.00,  0.00],

    # ---------- Y axis ----------
    [ 0.00,  0.05,  0.00],
    [ 0.00, -0.05,  0.00],

    # ---------- Z axis ----------
    [ 0.00,  0.00,  0.05],
    [ 0.00,  0.00, -0.05],

    # ---------- XY ----------
    [ 0.05,  0.05,  0.00],
    [ 0.05, -0.05,  0.00],
    [-0.05,  0.05,  0.00],
    [-0.05, -0.05,  0.00],

    # ---------- XZ ----------
    [ 0.05,  0.00,  0.05],
    [ 0.05,  0.00, -0.05],
    [-0.05,  0.00,  0.05],
    [-0.05,  0.00, -0.05],

    # ---------- XYZ ----------
    [ 0.05,  0.05,  0.05],
    [ 0.05,  0.05, -0.05],
    [ 0.05, -0.05,  0.05],
    [ 0.05, -0.05, -0.05],

    # ---------- Larger diagonal ----------
    [ 0.08,  0.08, -0.08],

], dtype=np.float64)


# ============================================================
# Coordinate Conversion
# ============================================================

def real_to_sim(real_pos):

    """
    Convert real Franka EE coordinates
    to panda-gym coordinates.
    """

    return real_pos + BASE_OFFSET


# ============================================================
# Observation Builder
# ============================================================

def build_obs(
    current_pos_real,
    current_vel,
    target_real
):

    current_sim = real_to_sim(
        current_pos_real
    )

    target_sim = real_to_sim(
        target_real
    )

    return {

        "observation": np.concatenate([
            current_sim,
            current_vel
        ]).astype(np.float32),

        "achieved_goal":
            current_sim.astype(np.float32),

        "desired_goal":
            target_sim.astype(np.float32),

    }


# ============================================================
# Save Episode Summary
# ============================================================

def save_episode_summary(result):

    os.makedirs(
        RESULT_DIR,
        exist_ok=True
    )

    file_exists = os.path.exists(
        EPISODE_SUMMARY_FILE
    )

    with open(
        EPISODE_SUMMARY_FILE,
        "a",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=result.keys()
        )

        if not file_exists:

            writer.writeheader()

        writer.writerow(
            result
        )


# ============================================================
# Save Step Data
# ============================================================

def save_step_log(
    step_records,
    filename
):

    if len(step_records) == 0:

        return

    os.makedirs(
        RUN_DIR,
        exist_ok=True
    )

    with open(
        filename,
        "w",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=step_records[0].keys()
        )

        writer.writeheader()

        writer.writerows(
            step_records
        )


# ============================================================
# Run One Episode
# ============================================================

def run_episode(
    panda,
    model,
    goal_offset,
    episode_id
):

    run_id = (
        datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )
        + f"_ep{episode_id:02d}"
    )


    print("\n")
    print("=" * 70)

    print(
        f"EPISODE {episode_id}"
    )

    print("=" * 70)


    # ========================================================
    # Reset Robot
    # ========================================================

    print(
        "Moving robot to start..."
    )

    panda.move_to_start()

    time.sleep(1.0)


    x0 = panda.get_position().copy()

    q0 = panda.get_orientation().copy()


    target_real = (
        x0
        + goal_offset
    )


    initial_distance = np.linalg.norm(
        target_real - x0
    )


    print(
        f"Start EE position: "
        f"{x0.round(4)}"
    )

    print(
        f"Goal offset:       "
        f"{goal_offset.round(4)}"
    )

    print(
        f"Target position:   "
        f"{target_real.round(4)}"
    )

    print(
        f"Initial distance:  "
        f"{initial_distance * 100:.2f} cm"
    )


    # ========================================================
    # PPO Sanity Check
    # ========================================================

    test_obs = build_obs(
        x0,
        np.zeros(3),
        target_real
    )

    test_action, _ = model.predict(
        test_obs,
        deterministic=True
    )


    print(
        f"Initial PPO action: "
        f"{test_action.round(4)}"
    )


    if DRY_RUN:

        print(
            "\nDRY_RUN enabled."
        )

        print(
            "Robot will not receive "
            "movement commands."
        )


    input(
        "\nCheck robot workspace and E-stop. "
        "Press ENTER to start episode "
        "(Ctrl+C to stop benchmark): "
    )


    # ========================================================
    # Controller
    # ========================================================

    ctrl = controllers.CartesianImpedance()


    if not DRY_RUN:

        panda.start_controller(
            ctrl
        )


    # ========================================================
    # Panda Internal Logging
    # ========================================================

    panda.enable_logging(
        int(
            CONTROL_FREQ
            * MAX_RUNTIME
        ) + 100
    )


    # ========================================================
    # Metrics
    # ========================================================

    step_count = 0

    prev_pos = x0.copy()

    prev_action = None

    trajectory_length = 0.0

    action_changes = []

    step_records = []


    success = False

    termination_reason = (
        "unknown"
    )


    final_distance = (
        initial_distance
    )


    start_time = (
        time.perf_counter()
    )


    # ========================================================
    # Control Loop
    # ========================================================

    try:

        with panda.create_context(

            frequency=CONTROL_FREQ,

            max_runtime=MAX_RUNTIME

        ) as ctx:


            while ctx.ok():


                # =================================================
                # Robot State
                # =================================================

                current_ee = (
                    panda.get_position().copy()
                )


                # =================================================
                # Actual EE Movement
                # =================================================

                step_displacement = (
                    np.linalg.norm(
                        current_ee
                        - prev_pos
                    )
                )


                trajectory_length += (
                    step_displacement
                )


                # =================================================
                # Velocity Estimate
                # =================================================

                current_vel = (

                    current_ee
                    - prev_pos

                ) * CONTROL_FREQ


                prev_pos = (
                    current_ee.copy()
                )


                # =================================================
                # Goal Error
                # =================================================

                error = (
                    target_real
                    - current_ee
                )


                distance = (
                    np.linalg.norm(
                        error
                    )
                )


                final_distance = (
                    distance
                )


                # =================================================
                # PPO Observation
                # =================================================

                obs = build_obs(

                    current_ee,

                    current_vel,

                    target_real

                )


                # =================================================
                # PPO Inference
                # =================================================

                raw_action, _ = (
                    model.predict(

                        obs,

                        deterministic=True

                    )
                )


                raw_action = np.asarray(
                    raw_action,
                    dtype=np.float64
                )


                # =================================================
                # Action Scaling
                # =================================================

                action = (

                    raw_action.copy()

                    * ACTION_SCALE

                )


                # =================================================
                # 3D Safety Norm Cap
                #
                # MAX_STEP represents total Cartesian displacement,
                # not independent per-axis clipping.
                # =================================================

                action_norm = (
                    np.linalg.norm(
                        action
                    )
                )


                if (
                    action_norm
                    > MAX_STEP
                ):

                    action = (

                        action
                        / action_norm
                        * MAX_STEP

                    )


                # =================================================
                # Action Smoothness
                # =================================================

                action_change = np.nan


                if (
                    prev_action
                    is not None
                ):

                    action_change = (
                        np.linalg.norm(

                            action
                            - prev_action

                        )
                    )


                    action_changes.append(
                        action_change
                    )


                prev_action = (
                    action.copy()
                )


                # =================================================
                # Cartesian Controller Target
                # =================================================

                target_position = (

                    current_ee
                    + action

                )


                if not DRY_RUN:

                    ctrl.set_control(

                        target_position,

                        q0

                    )


                step_count += 1


                elapsed_time = (

                    time.perf_counter()

                    - start_time

                )


                # =================================================
                # Save Step Data
                # =================================================

                step_records.append({

                    "run_id":
                        run_id,

                    "episode":
                        episode_id,

                    "step":
                        step_count,

                    "time_sec":
                        elapsed_time,


                    # Current EE

                    "ee_x":
                        current_ee[0],

                    "ee_y":
                        current_ee[1],

                    "ee_z":
                        current_ee[2],


                    # Goal offset

                    "goal_offset_x":
                        goal_offset[0],

                    "goal_offset_y":
                        goal_offset[1],

                    "goal_offset_z":
                        goal_offset[2],


                    # Absolute goal

                    "goal_x":
                        target_real[0],

                    "goal_y":
                        target_real[1],

                    "goal_z":
                        target_real[2],


                    # Error

                    "error_x":
                        error[0],

                    "error_y":
                        error[1],

                    "error_z":
                        error[2],

                    "distance_m":
                        distance,


                    # Velocity

                    "velocity_x":
                        current_vel[0],

                    "velocity_y":
                        current_vel[1],

                    "velocity_z":
                        current_vel[2],


                    # Raw PPO action

                    "raw_action_x":
                        raw_action[0],

                    "raw_action_y":
                        raw_action[1],

                    "raw_action_z":
                        raw_action[2],


                    # Scaled action

                    "action_x":
                        action[0],

                    "action_y":
                        action[1],

                    "action_z":
                        action[2],


                    # Action metrics

                    "action_norm":
                        np.linalg.norm(
                            action
                        ),

                    "action_change":
                        action_change,


                    # Controller target

                    "target_x":
                        target_position[0],

                    "target_y":
                        target_position[1],

                    "target_z":
                        target_position[2],


                    # Trajectory metrics

                    "step_displacement_m":
                        step_displacement,

                    "trajectory_length_m":
                        trajectory_length,

                })


                # =================================================
                # Console Output
                # =================================================

                if (
                    step_count
                    % CONTROL_FREQ
                    == 0
                ):

                    print(
                        "\n"
                        + "-"
                        * 50
                    )

                    print(
                        f"Step "
                        f"{step_count}"
                    )

                    print(
                        f"  EE position:    "
                        f"{current_ee.round(4)}"
                    )

                    print(
                        f"  Goal:           "
                        f"{target_real.round(4)}"
                    )

                    print(
                        f"  Error xyz:      "
                        f"{error.round(4)}"
                    )

                    print(
                        f"  Distance:       "
                        f"{distance * 100:.2f} cm"
                    )

                    print(
                        f"  Velocity:       "
                        f"{current_vel.round(4)}"
                    )

                    print(
                        f"  Raw PPO action: "
                        f"{raw_action.round(4)}"
                    )

                    print(
                        f"  Command delta:  "
                        f"{action.round(4)}"
                    )

                    print(
                        f"  Ctrl target:    "
                        f"{target_position.round(4)}"
                    )


                # =================================================
                # Success Condition
                # =================================================

                if (
                    distance
                    < GOAL_THRESHOLD
                ):

                    success = True

                    termination_reason = (
                        "goal_reached"
                    )


                    print(
                        "\nGoal reached!"
                    )

                    print(
                        f"Final distance: "
                        f"{distance * 100:.3f} cm"
                    )

                    break


        # ====================================================
        # Context Finished Naturally
        # ====================================================

        if not success:

            termination_reason = (
                "max_runtime"
            )


    # ========================================================
    # User Interrupt
    # ========================================================

    except KeyboardInterrupt:

        termination_reason = (
            "user_interrupt"
        )


        print(
            "\n\nInterrupted by user"
        )


    # ========================================================
    # Unexpected Error
    # ========================================================

    except Exception as e:

        termination_reason = (
            f"error:{type(e).__name__}"
        )


        print(
            f"\n\nError: {e}"
        )


    # ========================================================
    # Cleanup + Save Results
    # ========================================================

    finally:


        elapsed_time = (

            time.perf_counter()

            - start_time

        )


        # ====================================================
        # Smoothness Metrics
        # ====================================================

        if (
            len(action_changes)
            > 0
        ):

            mean_action_change = (
                float(
                    np.mean(
                        action_changes
                    )
                )
            )


            max_action_change = (
                float(
                    np.max(
                        action_changes
                    )
                )
            )


        else:

            mean_action_change = (
                np.nan
            )

            max_action_change = (
                np.nan
            )


        # ====================================================
        # Panda Log
        # ====================================================

        try:

            panda.disable_logging()

            log = (
                panda.get_log()
            )

        except Exception:

            log = None


        # ====================================================
        # Stop Controller
        # ====================================================

        if not DRY_RUN:

            try:

                panda.stop_controller()

            except Exception:

                pass


        # ====================================================
        # Save Step CSV
        # ====================================================

        step_log_file = os.path.join(

            RUN_DIR,

            f"run_{run_id}.csv"

        )


        save_step_log(

            step_records,

            step_log_file

        )


        print(
            f"\nStep log saved: "
            f"{step_log_file}"
        )


        # ====================================================
        # Episode Summary
        # ====================================================

        result = {

            "run_id":
                run_id,

            "episode":
                episode_id,

            "checkpoint":
                CHECKPOINT_PATH,

            "success":
                success,

            "termination_reason":
                termination_reason,


            # Goal offset

            "offset_x":
                goal_offset[0],

            "offset_y":
                goal_offset[1],

            "offset_z":
                goal_offset[2],


            # Start

            "start_x":
                x0[0],

            "start_y":
                x0[1],

            "start_z":
                x0[2],


            # Goal

            "goal_x":
                target_real[0],

            "goal_y":
                target_real[1],

            "goal_z":
                target_real[2],


            # Performance

            "initial_distance_cm":
                initial_distance * 100,

            "final_distance_cm":
                final_distance * 100,

            "steps":
                step_count,

            "time_sec":
                elapsed_time,

            "trajectory_length_cm":
                trajectory_length * 100,


            # Smoothness

            "mean_action_change":
                mean_action_change,

            "max_action_change":
                max_action_change,


            # Evaluation configuration

            "control_freq_hz":
                CONTROL_FREQ,

            "action_scale":
                ACTION_SCALE,

            "max_step_m":
                MAX_STEP,

            "goal_threshold_cm":
                GOAL_THRESHOLD * 100,

            "dry_run":
                DRY_RUN,

        }


        save_episode_summary(
            result
        )


        print(
            f"Episode summary appended: "
            f"{EPISODE_SUMMARY_FILE}"
        )


        # ====================================================
        # Save Panda Raw Log
        # ====================================================

        if (
            log is not None
            and len(log) > 0
        ):

            panda_log_file = os.path.join(

                RUN_DIR,

                f"run_{run_id}_panda.npy"

            )


            np.save(
                panda_log_file,
                log
            )


            print(
                f"Panda log saved: "
                f"{panda_log_file}"
            )


        # ====================================================
        # Print Episode Result
        # ====================================================

        print("\n")
        print("=" * 60)

        print(
            "EPISODE RESULT"
        )

        print("=" * 60)


        print(
            f"Success:              "
            f"{success}"
        )

        print(
            f"Termination:          "
            f"{termination_reason}"
        )

        print(
            f"Initial distance:     "
            f"{initial_distance * 100:.3f} cm"
        )

        print(
            f"Final distance:       "
            f"{final_distance * 100:.3f} cm"
        )

        print(
            f"Episode steps:        "
            f"{step_count}"
        )

        print(
            f"Episode time:         "
            f"{elapsed_time:.3f} s"
        )

        print(
            f"Trajectory length:    "
            f"{trajectory_length * 100:.3f} cm"
        )

        print(
            f"Mean action change:   "
            f"{mean_action_change:.6f}"
        )

        print(
            f"Max action change:    "
            f"{max_action_change:.6f}"
        )

        print("=" * 60)


    return result


# ============================================================
# Main Benchmark
# ============================================================

def main():


    # ========================================================
    # Create Result Directories
    # ========================================================

    os.makedirs(
        RUN_DIR,
        exist_ok=True
    )


    print("=" * 70)

    print(
        "PPO REAL ROBOT REACH BENCHMARK"
    )

    print("=" * 70)


    # ========================================================
    # Connect Robot
    # ========================================================

    print(
        f"Connecting to robot: "
        f"{HOSTNAME}"
    )


    panda = panda_py.Panda(
        HOSTNAME
    )


    print(
        "Robot connected."
    )


    # ========================================================
    # Load PPO
    # ========================================================

    print(
        f"Loading policy: "
        f"{CHECKPOINT_PATH}"
    )


    model = PPO.load(
        CHECKPOINT_PATH
    )


    print(
        "Policy loaded."
    )


    # ========================================================
    # Show Benchmark Settings
    # ========================================================

    print("\nBenchmark configuration:")


    print(
        f"  Episodes:       "
        f"{len(GOAL_OFFSETS)}"
    )

    print(
        f"  Control freq:   "
        f"{CONTROL_FREQ} Hz"
    )

    print(
        f"  Max runtime:    "
        f"{MAX_RUNTIME} s"
    )

    print(
        f"  Action scale:   "
        f"{ACTION_SCALE}"
    )

    print(
        f"  Max step:       "
        f"{MAX_STEP * 100:.1f} cm"
    )

    print(
        f"  Goal threshold: "
        f"{GOAL_THRESHOLD * 100:.1f} cm"
    )

    print(
        f"  DRY_RUN:        "
        f"{DRY_RUN}"
    )


    # ========================================================
    # Benchmark Results
    # ========================================================

    results = []


    # ========================================================
    # Run All Fixed Goals
    # ========================================================

    for (
        episode_id,
        goal_offset
    ) in enumerate(
        GOAL_OFFSETS,
        start=1
    ):


        result = run_episode(

            panda,

            model,

            goal_offset,

            episode_id

        )


        results.append(
            result
        )


        print(
            f"\nBenchmark progress: "
            f"{episode_id}/"
            f"{len(GOAL_OFFSETS)}"
        )


        # Stop entire benchmark
        # if Ctrl+C was used

        if (
            result[
                "termination_reason"
            ]
            == "user_interrupt"
        ):

            print(
                "\nBenchmark interrupted."
            )

            break


    # ========================================================
    # No Results
    # ========================================================

    if len(results) == 0:

        print(
            "No benchmark results."
        )

        return


    # ========================================================
    # Convert Results
    # ========================================================

    successes = np.array([

        int(
            r["success"]
        )

        for r in results

    ])


    errors = np.array([

        r["final_distance_cm"]

        for r in results

    ])


    runtimes = np.array([

        r["time_sec"]

        for r in results

    ])


    trajectory_lengths = np.array([

        r["trajectory_length_cm"]

        for r in results

    ])


    action_smoothness = np.array([

        r["mean_action_change"]

        for r in results

    ])


    # ========================================================
    # Final Benchmark Summary
    # ========================================================

    print("\n")
    print("=" * 70)

    print(
        "BENCHMARK SUMMARY"
    )

    print("=" * 70)


    print(
        f"Episodes completed:      "
        f"{len(results)}"
    )


    print(
        f"Successes:               "
        f"{successes.sum()}/"
        f"{len(successes)}"
    )


    print(
        f"Success rate:            "
        f"{successes.mean() * 100:.1f}%"
    )


    print(
        f"Mean final error:        "
        f"{errors.mean():.3f} cm"
    )


    print(
        f"Median final error:      "
        f"{np.median(errors):.3f} cm"
    )


    print(
        f"Std final error:         "
        f"{errors.std():.3f} cm"
    )


    print(
        f"Minimum final error:     "
        f"{errors.min():.3f} cm"
    )


    print(
        f"Maximum final error:     "
        f"{errors.max():.3f} cm"
    )


    print(
        f"Mean runtime:            "
        f"{runtimes.mean():.3f} s"
    )


    print(
        f"Mean trajectory length:  "
        f"{trajectory_lengths.mean():.3f} cm"
    )


    valid_smoothness = (
        action_smoothness[
            ~np.isnan(
                action_smoothness
            )
        ]
    )


    if (
        len(valid_smoothness)
        > 0
    ):

        print(
            f"Mean action smoothness:  "
            f"{valid_smoothness.mean():.6f}"
        )


    print("=" * 70)


    print(
        f"\nEpisode summary file:\n"
        f"{EPISODE_SUMMARY_FILE}"
    )


    print(
        f"\nDetailed run files:\n"
        f"{RUN_DIR}"
    )


    print("\nBenchmark finished.")


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":

    main()
