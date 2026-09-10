"""
Franka RL Deployment - Cartesian Impedance Controller
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
"""

import panda_py
from panda_py import controllers
import numpy as np
import logging

import os
import csv
import time
from datetime import datetime

from stable_baselines3 import PPO


logging.basicConfig(level=logging.INFO)


# ============ Configuration ============

HOSTNAME = '192.168.1.8'

CHECKPOINT_PATH = 'checkpoints/panda_reach_ppo_40000_steps'


# panda-gym base offset for coordinate conversion
BASE_OFFSET = np.array([-0.6, 0.0, 0.0])


# Action scaling
ACTION_SCALE = 0.05


# Safety
MAX_STEP = 0.03
CONTROL_FREQ = 20
MAX_RUNTIME = 30

# Success threshold
GOAL_THRESHOLD = 0.05


# Set True for inference without robot movement
DRY_RUN = False


# Logging
RESULT_DIR = "results"
RUN_DIR = os.path.join(RESULT_DIR, "runs")
EPISODE_SUMMARY_FILE = os.path.join(
    RESULT_DIR,
    "episodes.csv"
)


# ============ Coordinate Conversion ============

def real_to_sim(real_pos):

    """Convert real robot coordinates to panda-gym coordinates."""

    return real_pos + BASE_OFFSET


# ============ Observation Builder ============

def build_obs(
    current_pos_real,
    current_vel,
    target_real
):

    current_sim = real_to_sim(current_pos_real)
    target_sim = real_to_sim(target_real)

    return {

        'observation': np.concatenate([
            current_sim,
            current_vel,
        ]).astype(np.float32),

        'achieved_goal':
            current_sim.astype(np.float32),

        'desired_goal':
            target_sim.astype(np.float32),

    }


# ============ Save Episode Summary ============

def save_episode_summary(result):

    os.makedirs(RESULT_DIR, exist_ok=True)

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

        writer.writerow(result)


# ============ Save Step Data ============

def save_step_log(step_records, filename):

    if len(step_records) == 0:
        return

    os.makedirs(RUN_DIR, exist_ok=True)

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

        writer.writerows(step_records)


# ============ Main ============

def main():

    os.makedirs(RUN_DIR, exist_ok=True)

    run_id = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    print("=" * 50)
    print("Connecting to robot...")

    panda = panda_py.Panda(HOSTNAME)

    panda.move_to_start()

    print("Robot at start position")


    # ---------- Load PPO ----------

    print(
        f"Loading policy: {CHECKPOINT_PATH}"
    )

    model = PPO.load(CHECKPOINT_PATH)

    print("Policy loaded")


    # ---------- Initial robot pose ----------

    x0 = panda.get_position()

    q0 = panda.get_orientation()

    print(
        f"Start EE position (real): "
        f"{x0.round(3)}"
    )


    # ---------- Target ----------

    target_real = (
        x0
        + np.array([
            0.1,
            0.1,
            -0.1
        ])
    )

    print(
        f"Target position (real):   "
        f"{target_real.round(3)}"
    )


    # ---------- PPO sanity check ----------

    print("-" * 50)

    print(
        "Running sanity check "
        "(one inference pass)..."
    )

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
        f"Test observation: "
        f"{test_obs['observation'].round(3)}"
    )

    print(
        f"Test action: "
        f"{test_action.round(3)}"
    )

    print(
        "Sanity check passed - shapes OK"
    )


    if DRY_RUN:

        print("=" * 50)

        print(
            "DRY_RUN enabled — "
            "robot will NOT move."
        )

        print("=" * 50)


    # ---------- Parameters ----------

    print("\nControl parameters:")

    print(
        f"  Frequency:      "
        f"{CONTROL_FREQ} Hz"
    )

    print(
        f"  Max runtime:    "
        f"{MAX_RUNTIME} s"
    )

    print(
        f"  Max step:       "
        f"{MAX_STEP} m"
    )

    print(
        f"  Action scale:   "
        f"{ACTION_SCALE}"
    )

    print(
        f"  Goal threshold: "
        f"{GOAL_THRESHOLD * 100:.1f} cm"
    )

    print(
        f"  DRY_RUN:        "
        f"{DRY_RUN}"
    )


    input(
        ">>> Press Enter to start "
        "(Ctrl+C to abort) <<<"
    )


    # ---------- Controller ----------

    ctrl = controllers.CartesianImpedance()

    panda.start_controller(ctrl)


    # ---------- Panda internal logging ----------

    panda.enable_logging(
        int(
            CONTROL_FREQ
            * MAX_RUNTIME
        ) + 100
    )


    # ============ Metrics ============

    step_count = 0

    prev_pos = x0.copy()

    prev_action = None

    trajectory_length = 0.0

    action_changes = []

    step_records = []

    success = False

    termination_reason = "unknown"

    final_distance = np.linalg.norm(
        target_real - x0
    )

    start_time = time.perf_counter()


    try:

        with panda.create_context(

            frequency=CONTROL_FREQ,

            max_runtime=MAX_RUNTIME

        ) as ctx:

            while ctx.ok():

                # =========================
                # Robot State
                # =========================

                current_ee = (
                    panda.get_position()
                )


                # Actual EE displacement
                step_displacement = (
                    np.linalg.norm(
                        current_ee - prev_pos
                    )
                )


                trajectory_length += (
                    step_displacement
                )


                # Velocity
                current_vel = (

                    current_ee - prev_pos

                ) * CONTROL_FREQ


                prev_pos = (
                    current_ee.copy()
                )


                # =========================
                # Goal Error
                # =========================

                error = (
                    target_real
                    - current_ee
                )


                distance = np.linalg.norm(
                    error
                )


                final_distance = distance


                # =========================
                # PPO Observation
                # =========================

                obs = build_obs(

                    current_ee,

                    current_vel,

                    target_real

                )


                # =========================
                # PPO Inference
                # =========================

                raw_action, _ = (
                    model.predict(

                        obs,

                        deterministic=True

                    )
                )


                # Scale
                action = (

                    raw_action.copy()

                    * ACTION_SCALE

                )


                # Safety clipping
                action = np.clip(

                    action,

                    -MAX_STEP,

                    MAX_STEP

                )


                # =========================
                # Action Smoothness
                # =========================

                action_change = np.nan


                if prev_action is not None:

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


                # =========================
                # Controller Target
                # =========================

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


                # =========================
                # Save Every Step
                # =========================

                step_records.append({

                    "run_id":
                        run_id,

                    "step":
                        step_count,

                    "time_sec":
                        elapsed_time,

                    "ee_x":
                        current_ee[0],

                    "ee_y":
                        current_ee[1],

                    "ee_z":
                        current_ee[2],

                    "goal_x":
                        target_real[0],

                    "goal_y":
                        target_real[1],

                    "goal_z":
                        target_real[2],

                    "error_x":
                        error[0],

                    "error_y":
                        error[1],

                    "error_z":
                        error[2],

                    "distance_m":
                        distance,

                    "velocity_x":
                        current_vel[0],

                    "velocity_y":
                        current_vel[1],

                    "velocity_z":
                        current_vel[2],

                    "raw_action_x":
                        raw_action[0],

                    "raw_action_y":
                        raw_action[1],

                    "raw_action_z":
                        raw_action[2],

                    "action_x":
                        action[0],

                    "action_y":
                        action[1],

                    "action_z":
                        action[2],

                    "action_change":
                        action_change,

                    "target_x":
                        target_position[0],

                    "target_y":
                        target_position[1],

                    "target_z":
                        target_position[2],

                    "step_displacement_m":
                        step_displacement,

                    "trajectory_length_m":
                        trajectory_length,

                })


                # =========================
                # Console output
                # =========================

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


                # =========================
                # Success Condition
                # =========================

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


            # ctx finished naturally
            if not success:

                termination_reason = (
                    "max_runtime"
                )


    except KeyboardInterrupt:

        termination_reason = (
            "user_interrupt"
        )

        print(
            "\n\nInterrupted by user"
        )


    except Exception as e:

        termination_reason = (
            f"error:{type(e).__name__}"
        )

        print(
            f"\n\nError: {e}"
        )


    finally:

        # =========================
        # Final Metrics
        # =========================

        elapsed_time = (

            time.perf_counter()

            - start_time

        )


        if len(action_changes) > 0:

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


        # =========================
        # Panda log
        # =========================

        panda.disable_logging()

        log = panda.get_log()


        # =========================
        # Stop controller
        # =========================

        try:

            panda.stop_controller()

        except Exception:

            pass


        # =========================
        # Print Result
        # =========================

        print("\n")
        print("=" * 60)

        print("EPISODE RESULT")

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


        # =========================
        # Save Step CSV
        # =========================

        step_log_file = os.path.join(

            RUN_DIR,

            f"run_{run_id}.csv"

        )


        save_step_log(

            step_records,

            step_log_file

        )


        print(
            f"Step log saved: "
            f"{step_log_file}"
        )


        # =========================
        # Save Episode Summary
        # =========================

        result = {

            "run_id":
                run_id,

            "checkpoint":
                CHECKPOINT_PATH,

            "success":
                success,

            "termination_reason":
                termination_reason,

            "start_x":
                x0[0],

            "start_y":
                x0[1],

            "start_z":
                x0[2],

            "goal_x":
                target_real[0],

            "goal_y":
                target_real[1],

            "goal_z":
                target_real[2],

            "initial_distance_cm":
                np.linalg.norm(
                    target_real - x0
                ) * 100,

            "final_distance_cm":
                final_distance * 100,

            "steps":
                step_count,

            "time_sec":
                elapsed_time,

            "trajectory_length_cm":
                trajectory_length * 100,

            "mean_action_change":
                mean_action_change,

            "max_action_change":
                max_action_change,

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


        # =========================
        # Save Panda Raw Log
        # =========================

        if (
            log is not None
            and len(log) > 0
        ):

            panda_log_file = (
                os.path.join(

                    RUN_DIR,

                    f"run_{run_id}_panda.npy"

                )
            )

            np.save(
                panda_log_file,
                log
            )

            print(
                f"Panda log saved: "
                f"{panda_log_file}"
            )


        # =========================
        # Return to Start
        # =========================

        print(
            "Returning to start..."
        )

        try:

            panda.move_to_start()

        except Exception:

            print(
                "move_to_start failed, "
                "attempting recover..."
            )

            panda.recover()

            panda.move_to_start()


        print("Done.")


if __name__ == '__main__':
    main()
