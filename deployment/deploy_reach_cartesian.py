"""
Franka RL Deployment - Cartesian Impedance Controller
Deploys a PandaReach PPO policy on the real Franka using panda-py's
CartesianImpedance controller (real-time, smooth, with built-in joint walls).
"""

import panda_py
from panda_py import controllers
import numpy as np
import logging
from stable_baselines3 import PPO

logging.basicConfig(level=logging.INFO)

# ============ Configuration ============
HOSTNAME = '192.168.1.8'
CHECKPOINT_PATH = 'checkpoints/panda_reach_ppo_40000_steps'

# panda-gym base offset for coordinate conversion
# panda-gym places the robot base at [-0.6, 0, 0] in world frame
BASE_OFFSET = np.array([-0.6, 0.0, 0.0])

# Action scaling: policy outputs are in [-1, 1], panda-gym scales by 0.05
# (max EE displacement per step is 5cm in sim)
ACTION_SCALE = 0.05

# Safety
MAX_STEP = 0.03        # max EE displacement per control step (m) - conservative
CONTROL_FREQ = 20      # Hz
MAX_RUNTIME = 15       # seconds
GOAL_THRESHOLD = 0.05  # distance to consider goal reached (m)

# Set to True to run inference WITHOUT sending commands (safe dry run)
DRY_RUN = True


# ============ Coordinate Conversion ============
def real_to_sim(real_pos):
    """Convert real robot coordinates to panda-gym sim coordinates."""
    return real_pos + BASE_OFFSET


# ============ Observation Builder ============
def build_obs(current_pos_real, current_vel, target_real):
    """
    Build observation matching panda-gym PandaReach format:
      observation: [ee_x, ee_y, ee_z, ee_vx, ee_vy, ee_vz]  (6D)
      achieved_goal: [ee_x, ee_y, ee_z]  (3D)
      desired_goal: [target_x, target_y, target_z]  (3D)
    All positions converted to sim coordinate frame.
    """
    current_sim = real_to_sim(current_pos_real)
    target_sim = real_to_sim(target_real)

    return {
        'observation': np.concatenate([
            current_sim,
            current_vel,
        ]).astype(np.float32),
        'achieved_goal': current_sim.astype(np.float32),
        'desired_goal': target_sim.astype(np.float32),
    }


# ============ Main ============
def main():
    # --- Connect ---
    print("=" * 50)
    print("Connecting to robot...")
    panda = panda_py.Panda(HOSTNAME)
    panda.move_to_start()
    print("Robot at start position")

    # --- Load policy ---
    print(f"Loading policy: {CHECKPOINT_PATH}")
    model = PPO.load(CHECKPOINT_PATH)
    print("Policy loaded")

    # --- Get start position and orientation ---
    x0 = panda.get_position()      # current EE position (xyz)
    q0 = panda.get_orientation()   # current EE orientation (quaternion) - keep fixed
    print(f"Start EE position (real): {x0.round(3)}")

    # --- Set target ---
    target_real = x0 + np.array([0.1, 0.1, -0.1])  # move 10cm in x,y and down 10cm
    print(f"Target position (real):   {target_real.round(3)}")

    # --- Sanity check: one inference pass before touching hardware ---
    print("-" * 50)
    print("Running sanity check (one inference pass)...")
    test_obs = build_obs(x0, np.zeros(3), target_real)
    test_action, _ = model.predict(test_obs, deterministic=True)
    print(f"Test observation: {test_obs['observation'].round(3)}")
    print(f"Test action:      {test_action.round(3)}")
    print("Sanity check passed - shapes OK")

    if DRY_RUN:
        print("=" * 50)
        print("DRY_RUN is True - running inference loop WITHOUT moving robot.")
        print("Set DRY_RUN = False to actually control the robot.")
        print("=" * 50)

    # --- Confirmation ---
    print(f"\nControl parameters:")
    print(f"  Frequency:     {CONTROL_FREQ} Hz")
    print(f"  Max runtime:   {MAX_RUNTIME} s")
    print(f"  Max step:      {MAX_STEP} m")
    print(f"  Action scale:  {ACTION_SCALE}")
    print(f"  DRY_RUN:       {DRY_RUN}")
    input(">>> Press Enter to start (Ctrl+C to abort) <<<")

    # --- Start controller ---
    ctrl = controllers.CartesianImpedance()
    panda.start_controller(ctrl)

    # --- Enable logging for analysis ---
    panda.enable_logging(int(CONTROL_FREQ * MAX_RUNTIME) + 100)

    step_count = 0
    prev_pos = x0.copy()

    try:
        with panda.create_context(
            frequency=CONTROL_FREQ,
            max_runtime=MAX_RUNTIME
        ) as ctx:
            while ctx.ok():
                # Read current position
                current_pos = panda.get_position()

                # Estimate velocity (finite difference)
                current_vel = (current_pos - prev_pos) * CONTROL_FREQ
                prev_pos = current_pos.copy()

                # Build observation
                obs = build_obs(current_pos, current_vel, target_real)

                # Policy inference (outputs displacement in [-1, 1])
                raw_action, _ = model.predict(obs, deterministic=True)

                # Scale action to real displacement
                displacement = raw_action * ACTION_SCALE

                # Safety: limit step size
                displacement = np.clip(displacement, -MAX_STEP, MAX_STEP)

                # Compute new target position
                x_d = current_pos + displacement

                # Send control (unless dry run)
                if not DRY_RUN:
                    ctrl.set_control(x_d, q0)

                # Check goal
                dist = np.linalg.norm(current_pos - target_real)
                step_count += 1

                if step_count % CONTROL_FREQ == 0:  # print once per second
                    print(f"Step {step_count}: pos={current_pos.round(3)} "
                          f"action={raw_action.round(3)} dist={dist:.3f}")

                if dist < GOAL_THRESHOLD:
                    print(f"\nGoal reached! Final distance: {dist:.3f}")
                    break

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")

    except Exception as e:
        print(f"\n\nError: {e}")

    finally:
        # --- Safe shutdown ---
        panda.disable_logging()
        log = panda.get_log()

        print("=" * 50)
        print(f"Total steps: {step_count}")

        # Stop controller before moving
        try:
            panda.stop_controller()
        except Exception:
            pass

        print("Returning to start...")
        try:
            panda.move_to_start()
        except Exception:
            print("move_to_start failed, attempting recover...")
            panda.recover()
            panda.move_to_start()

        print("Done.")

        # Save log for analysis
        if log is not None and len(log) > 0:
            np.save('experiment_log.npy', log)
            print("Log saved to experiment_log.npy")


if __name__ == '__main__':
    main()
