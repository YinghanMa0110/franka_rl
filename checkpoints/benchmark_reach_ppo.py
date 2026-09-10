"""
Real Robot Reach Benchmark - PPO

Purpose:
Evaluate a trained PandaReach PPO policy on the real Franka Panda
using a fixed set of target positions.

Metrics:
- Success rate
- Final error
- Time to goal
- Trajectory length

All algorithms should later use exactly the same:
- goals
- start pose
- controller parameters
- success threshold
- runtime
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


# ============================================================
# Configuration
# ============================================================

HOSTNAME = "192.168.1.8"

CHECKPOINT_PATH = "checkpoints/panda_reach_ppo_40000_steps"

BASE_OFFSET = np.array([-0.6, 0.0, 0.0])

CONTROL_FREQ = 20.0
DT = 1.0 / CONTROL_FREQ

MAX_RUNTIME = 30.0

# Same definition as PandaReach
GOAL_THRESHOLD = 0.05

# Keep these FIXED for algorithm comparison
ACTION_SCALE = 0.03
MAX_STEP = 0.02

TRANSLATIONAL_STIFFNESS = 900.0
ROTATIONAL_STIFFNESS = 30.0

DRY_RUN = False


# ============================================================
# Fixed benchmark goals
#
# These are offsets relative to the robot start position.
# IMPORTANT:
# Use exactly the same list for PPO / SAC / TD3 / their algorithm.
# ============================================================

GOAL_OFFSETS = np.array([

    # x only
    [ 0.05,  0.00,  0.00],
    [ 0.08,  0.00,  0.00],
    [-0.05,  0.00,  0.00],

    # y only
    [ 0.00,  0.05,  0.00],
    [ 0.00, -0.05,  0.00],

    # z only
    [ 0.00,  0.00,  0.05],
    [ 0.00,  0.00, -0.05],

    # xy
    [ 0.05,  0.05,  0.00],
    [ 0.05, -0.05,  0.00],
    [-0.05,  0.05,  0.00],
    [-0.05, -0.05,  0.00],

    # xz
    [ 0.05,  0.00,  0.05],
    [ 0.05,  0.00, -0.05],
    [-0.05,  0.00,  0.05],
    [-0.05,  0.00, -0.05],

    # xyz
    [ 0.05,  0.05,  0.05],
    [ 0.05,  0.05, -0.05],
    [ 0.05, -0.05,  0.05],
    [ 0.05, -0.05, -0.05],

    # Previous-style larger target
    [ 0.08,  0.08, -0.08],

], dtype=np.float64)


# ============================================================
# Coordinate conversion
# ============================================================

def real_to_sim(real_pos):
    return real_pos + BASE_OFFSET


def build_obs(current_pos_real, current_vel, target_real):

    current_sim = real_to_sim(current_pos_real)
    target_sim = real_to_sim(target_real)

    obs = {
        "observation": np.concatenate(
            [current_sim, current_vel]
        ).astype(np.float32),

        "achieved_goal":
            current_sim.astype(np.float32),

        "desired_goal":
            target_sim.astype(np.float32),
    }

    return obs


# ============================================================
# Run one episode
# ============================================================

def run_episode(
    panda,
    model,
    episode_id,
    goal_offset,
):

    print("\n" + "=" * 70)
    print(f"EPISODE {episode_id}")
    print("=" * 70)

    # --------------------------------------------------------
    # Return to same start position
    # --------------------------------------------------------

    print("Moving robot to start position...")

    panda.move_to_start()

    time.sleep(1.0)

    start_pos = panda.get_position().copy()
    start_orientation = panda.get_orientation().copy()

    target_real = start_pos + goal_offset

    print(f"Start position : {start_pos}")
    print(f"Goal offset    : {goal_offset}")
    print(f"Target position: {target_real}")

    initial_distance = np.linalg.norm(
        target_real - start_pos
    )

    print(
        f"Initial distance: "
        f"{initial_distance * 100:.2f} cm"
    )

    input(
        "\nCheck workspace + E-stop. "
        "Press ENTER to start episode..."
    )

    if DRY_RUN:
        print("DRY RUN - skipping movement.")
        return None

    # --------------------------------------------------------
    # Controller
    # --------------------------------------------------------

    ctrl = controllers.CartesianImpedance()

    ctrl.set_impedance(
        np.diag([
            TRANSLATIONAL_STIFFNESS,
            TRANSLATIONAL_STIFFNESS,
            TRANSLATIONAL_STIFFNESS,
            ROTATIONAL_STIFFNESS,
            ROTATIONAL_STIFFNESS,
            ROTATIONAL_STIFFNESS,
        ])
    )

    panda.start_controller(ctrl)

    time.sleep(0.2)

    # --------------------------------------------------------
    # Episode variables
    # --------------------------------------------------------

    start_time = time.time()

    prev_pos = panda.get_position().copy()

    trajectory_length = 0.0
    num_steps = 0

    success = False

    # --------------------------------------------------------
    # Control loop
    # --------------------------------------------------------

    try:

        while True:

            loop_start = time.time()

            current_pos = panda.get_position().copy()

            error = target_real - current_pos

            distance = np.linalg.norm(error)

            elapsed = time.time() - start_time

            # --------------------------------------------
            # Success
            # --------------------------------------------

            if distance < GOAL_THRESHOLD:

                success = True

                print(
                    f"\nSUCCESS: "
                    f"{distance * 100:.2f} cm"
                )

                break

            # --------------------------------------------
            # Timeout
            # --------------------------------------------

            if elapsed > MAX_RUNTIME:

                print(
                    f"\nTIMEOUT: "
                    f"{distance * 100:.2f} cm"
                )

                break

            # --------------------------------------------
            # Velocity estimate
            # --------------------------------------------

            current_vel = (
                current_pos - prev_pos
            ) * CONTROL_FREQ

            # trajectory length
            trajectory_length += np.linalg.norm(
                current_pos - prev_pos
            )

            prev_pos = current_pos.copy()

            # --------------------------------------------
            # Build PPO observation
            # --------------------------------------------

            obs = build_obs(
                current_pos,
                current_vel,
                target_real,
            )

            raw_action, _ = model.predict(
                obs,
                deterministic=True,
            )

            raw_action = np.asarray(
                raw_action,
                dtype=np.float64,
            )

            # --------------------------------------------
            # Scale action
            # --------------------------------------------

            action = raw_action * ACTION_SCALE

            norm = np.linalg.norm(action)

            if norm > MAX_STEP:
                action = (
                    action
                    / norm
                    * MAX_STEP
                )

            # Panda-gym style relative action
            ctrl_target = (
                current_pos + action
            )

            ctrl.set_control(
                ctrl_target,
                start_orientation,
            )

            num_steps += 1

            # --------------------------------------------
            # Debug print every 1 second
            # --------------------------------------------

            if num_steps % int(CONTROL_FREQ) == 0:

                print(
                    f"step={num_steps:4d} | "
                    f"error={distance*100:5.2f} cm | "
                    f"action={np.round(raw_action, 3)}"
                )

            # --------------------------------------------
            # Maintain 20 Hz
            # --------------------------------------------

            loop_elapsed = time.time() - loop_start

            sleep_time = DT - loop_elapsed

            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:

        print("\nEpisode interrupted by user.")

    finally:

        panda.stop_controller()

    # --------------------------------------------------------
    # Final metrics
    # --------------------------------------------------------

    final_pos = panda.get_position().copy()

    final_error = np.linalg.norm(
        target_real - final_pos
    )

    episode_time = time.time() - start_time

    print("\nEpisode result")
    print("-" * 40)

    print(
        f"Success           : {success}"
    )

    print(
        f"Final error       : "
        f"{final_error * 100:.2f} cm"
    )

    print(
        f"Time              : "
        f"{episode_time:.2f} s"
    )

    print(
        f"Trajectory length : "
        f"{trajectory_length * 100:.2f} cm"
    )

    print(
        f"Steps             : "
        f"{num_steps}"
    )

    return {

        "episode": episode_id,

        "offset_x_m": goal_offset[0],
        "offset_y_m": goal_offset[1],
        "offset_z_m": goal_offset[2],

        "target_x_m": target_real[0],
        "target_y_m": target_real[1],
        "target_z_m": target_real[2],

        "initial_distance_cm":
            initial_distance * 100,

        "success":
            int(success),

        "final_error_cm":
            final_error * 100,

        "time_s":
            episode_time,

        "trajectory_length_cm":
            trajectory_length * 100,

        "steps":
            num_steps,
    }


# ============================================================
# Main
# ============================================================

def main():

    logging.basicConfig(
        level=logging.INFO
    )

    print("=" * 70)
    print("REAL ROBOT PPO REACH BENCHMARK")
    print("=" * 70)

    # --------------------------------------------------------
    # Load PPO
    # --------------------------------------------------------

    print(
        f"Loading checkpoint: "
        f"{CHECKPOINT_PATH}"
    )

    model = PPO.load(
        CHECKPOINT_PATH
    )

    print("Checkpoint loaded.")

    # --------------------------------------------------------
    # Connect Franka
    # --------------------------------------------------------

    print(
        f"Connecting to Franka: "
        f"{HOSTNAME}"
    )

    panda = panda_py.Panda(
        HOSTNAME
    )

    print("Robot connected.")

    # --------------------------------------------------------
    # CSV
    # --------------------------------------------------------

    os.makedirs(
        "results",
        exist_ok=True,
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    csv_path = (
        f"results/"
        f"ppo_reach_real_{timestamp}.csv"
    )

    results = []

    # --------------------------------------------------------
    # Run benchmark
    # --------------------------------------------------------

    for i, offset in enumerate(
        GOAL_OFFSETS,
        start=1,
    ):

        result = run_episode(
            panda,
            model,
            i,
            offset,
        )

        if result is not None:

            results.append(
                result
            )

            # Save after EVERY episode
            # so results survive interruption

            with open(
                csv_path,
                "w",
                newline="",
            ) as f:

                writer = csv.DictWriter(
                    f,
                    fieldnames=result.keys(),
                )

                writer.writeheader()

                writer.writerows(
                    results
                )

        print(
            f"\nProgress: "
            f"{i}/{len(GOAL_OFFSETS)}"
        )

    # ========================================================
    # Summary
    # ========================================================

    if len(results) == 0:
        return

    successes = np.array([
        r["success"]
        for r in results
    ])

    final_errors = np.array([
        r["final_error_cm"]
        for r in results
    ])

    times = np.array([
        r["time_s"]
        for r in results
    ])

    paths = np.array([
        r["trajectory_length_cm"]
        for r in results
    ])

    print("\n")
    print("=" * 70)
    print("BENCHMARK RESULTS")
    print("=" * 70)

    print(
        f"Episodes: "
        f"{len(results)}"
    )

    print(
        f"Success rate: "
        f"{successes.mean()*100:.1f}%"
    )

    print(
        f"Mean final error: "
        f"{final_errors.mean():.2f} cm"
    )

    print(
        f"Median final error: "
        f"{np.median(final_errors):.2f} cm"
    )

    print(
        f"Mean time: "
        f"{times.mean():.2f} s"
    )

    print(
        f"Mean trajectory length: "
        f"{paths.mean():.2f} cm"
    )

    print(
        f"\nResults saved to:\n"
        f"{csv_path}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()
