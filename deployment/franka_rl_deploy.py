"""
Franka RL Deployment Script (fixed)
Loads a trained RL checkpoint and deploys it on the real Franka Emika Panda.

Fixes vs original:
1. EE position via panda.get_position() (raw O_T_EE is column-major, reshape was wrong)
2. ACTION_MODE config — policy output can be delta or absolute (VERIFY against training env!)
3. Observation builder must match training env (VERIFY! panda-gym Reach uses EE pos+vel, not q+dq)
4. prev_action initialized to current q (Layer 2 active from step 1)
5. finally: stop_controller + recover before move_to_start
6. Safety layer order: smoothing/deviation first, hard joint-limit clip LAST
7. Optional VecNormalize loading
8. DRY_RUN mode — print actions without sending them
"""

import panda_py
from panda_py import controllers
from stable_baselines3 import PPO
import numpy as np
import logging
import os

# ============ Configuration ============
logging.basicConfig(level=logging.INFO)

HOSTNAME = os.environ.get('PANDA_HOST', '192.168.3.100')
USERNAME = os.environ.get('PANDA_USER', 'user')
PASSWORD = os.environ.get('PANDA_PASS', 'password')

# >>> CHANGE THIS LINE TO DEPLOY A DIFFERENT ALGORITHM <<<
CHECKPOINT_PATH = '../checkpoints/panda_reach_ppo_40000_steps'

# Path to VecNormalize stats saved during training (set to None if not used).
# If training used VecNormalize and you skip this, the policy will silently misbehave.
VECNORM_PATH = None  # e.g. '../checkpoints/vecnormalize.pkl'

# !!! MUST MATCH TRAINING ENV !!!
#   'delta'    : policy outputs joint increments in [-1, 1], scaled by ACTION_SCALE
#                (panda-gym joints control mode style)
#   'absolute' : policy outputs absolute joint positions in radians
ACTION_MODE = 'delta'
ACTION_SCALE = 0.05  # rad per unit action, panda-gym default for joints mode

# Set True for first run: prints actions but does NOT send them to the robot
DRY_RUN = True

# ============ Safety Limits ============

HARDWARE_LIMITS = np.array([
    [-2.8973,  2.8973],  # A1
    [-1.7628,  1.7628],  # A2
    [-2.8973,  2.8973],  # A3
    [-3.0718, -0.0698],  # A4
    [-2.8973,  2.8973],  # A5
    [-0.0175,  3.7525],  # A6
    [-2.8973,  2.8973],  # A7
])

SAFE_MARGIN = 0.2  # radians
SOFT_LIMITS = HARDWARE_LIMITS.copy()
SOFT_LIMITS[:, 0] += SAFE_MARGIN
SOFT_LIMITS[:, 1] -= SAFE_MARGIN

MAX_DELTA = 0.01       # max change per step (radians)
MAX_DEVIATION = 0.2    # max deviation from current position (radians)
CONTROL_FREQ = 10      # Hz
MAX_RUNTIME = 5        # seconds


# ============ Observation Builder ============
def build_observation(panda, state, target_pos):
    """
    !!! MUST MATCH TRAINING ENV OBS FORMAT !!!

    Below is the panda-gym PandaReach (dense/sparse) format:
      observation   = [ee_position (3), ee_velocity (3)]
      achieved_goal = ee_position (3)
      desired_goal  = target (3)

    If your env used q + dq instead, swap the 'observation' line accordingly.
    SB3 will raise on shape mismatch — test with one model.predict() BEFORE
    touching the robot.
    """
    # Correct EE position: use the library, not raw O_T_EE reshape
    ee_pos = np.asarray(panda.get_position())

    # EE linear velocity: J(q) @ dq, take translational part
    jac = np.array(panda.get_model().zero_jacobian(state)).reshape(6, 7, order='F')
    ee_vel = jac[:3] @ np.array(state.dq)

    return {
        'observation': np.concatenate([ee_pos, ee_vel]).astype(np.float32),
        'achieved_goal': ee_pos.astype(np.float32),
        'desired_goal': target_pos.astype(np.float32),
    }


# ============ Action Conversion ============
def to_target_q(raw_action, current_q):
    """Convert policy output to an absolute joint-position target."""
    raw_action = np.asarray(raw_action, dtype=np.float64).flatten()

    if raw_action.shape[0] != 7:
        raise ValueError(
            f"Policy output has shape {raw_action.shape}, expected 7. "
            f"Your policy was likely trained with end-effector control "
            f"(3-dim actions) — this script only supports joint control. "
            f"Retrain with control_type='joints' or add an IK layer."
        )

    if ACTION_MODE == 'delta':
        return current_q + np.clip(raw_action, -1.0, 1.0) * ACTION_SCALE
    elif ACTION_MODE == 'absolute':
        return raw_action
    else:
        raise ValueError(f"Unknown ACTION_MODE: {ACTION_MODE}")


# ============ Safety Filter ============
def safe_action(target_q, current_q, prev_action):
    """
    Layer 1: smoothing  — limit change per step
    Layer 2: deviation  — limit distance from current position
    Layer 3: hard clip  — soft joint limits LAST so nothing can undo it
    """
    action = target_q.copy()

    action = np.clip(action, prev_action - MAX_DELTA, prev_action + MAX_DELTA)
    action = np.clip(action, current_q - MAX_DEVIATION, current_q + MAX_DEVIATION)
    action = np.clip(action, SOFT_LIMITS[:, 0], SOFT_LIMITS[:, 1])

    return action


# ============ Debug Helper ============
def debug_action(target_q, safe_act, step):
    if not np.allclose(target_q, safe_act):
        print(f"  [Step {step}] WARNING: Action was clipped!")

    for i, (act, limits) in enumerate(zip(safe_act, HARDWARE_LIMITS)):
        margin = min(act - limits[0], limits[1] - act)
        if margin < 0.1:
            print(f"  [Step {step}] WARNING: Joint A{i+1} near limit! "
                  f"margin={margin:.3f}")


# ============ Main ============
def main():
    # --- Step 1: Connect to Desk ---
    print("=" * 50)
    print("Step 1: Connecting to Desk...")
    desk = panda_py.Desk(HOSTNAME, USERNAME, PASSWORD)
    try:
        desk.take_control()  # required on newer firmware; harmless otherwise
    except Exception:
        pass
    desk.unlock()
    desk.activate_fci()
    print("Desk connected, brakes unlocked, FCI activated.")

    # --- Step 2: Connect to robot ---
    print("=" * 50)
    print("Step 2: Connecting to robot...")
    panda = panda_py.Panda(HOSTNAME)
    print("Robot connected.")

    # --- Step 3: Load policy ---
    print("=" * 50)
    print(f"Step 3: Loading policy from {CHECKPOINT_PATH}...")
    model = PPO.load(CHECKPOINT_PATH)

    vec_norm = None
    if VECNORM_PATH is not None:
        from stable_baselines3.common.vec_env import VecNormalize
        import pickle
        with open(VECNORM_PATH, 'rb') as f:
            vec_norm = pickle.load(f)
        vec_norm.training = False
        print("VecNormalize stats loaded.")
    print("Policy loaded.")

    # --- Step 4: Move to start position ---
    print("=" * 50)
    print("Step 4: Moving to start position...")
    panda.move_to_start()
    print("At start position.")
    print(f"Current joints: {np.array(panda.q).round(3)}")

    # --- Step 5: Set target ---
    target_pos = np.array([0.5, 0.0, 0.5])
    print(f"Target position: {target_pos}")

    # --- Step 5.5: Offline sanity check (no robot motion) ---
    print("=" * 50)
    print("Sanity check: one inference pass with real robot state...")
    state = panda.get_state()
    obs = build_observation(panda, state, target_pos)
    if vec_norm is not None:
        obs = vec_norm.normalize_obs(obs)
    raw_action, _ = model.predict(obs, deterministic=True)
    print(f"  raw_action = {np.asarray(raw_action).round(3)}")
    target_q = to_target_q(raw_action, np.array(state.q))
    print(f"  target_q   = {target_q.round(3)}")
    print(f"  current_q  = {np.array(state.q).round(3)}")
    print("Sanity check passed (shapes OK). Verify the numbers look reasonable!")

    # --- Step 6: User confirmation ---
    print("=" * 50)
    print(f"Control parameters:")
    print(f"  DRY_RUN:       {DRY_RUN}")
    print(f"  Action mode:   {ACTION_MODE} (scale={ACTION_SCALE})")
    print(f"  Frequency:     {CONTROL_FREQ} Hz")
    print(f"  Max runtime:   {MAX_RUNTIME} s")
    print(f"  Max delta:     {MAX_DELTA} rad/step")
    print(f"  Max deviation: {MAX_DEVIATION} rad")
    print("=" * 50)
    input(">>> Press Enter to start RL control (Ctrl+C to abort) <<<")

    # --- Step 7: Start controller ---
    ctrl = controllers.JointPosition()
    if not DRY_RUN:
        panda.start_controller(ctrl)

    # --- Step 8: Control loop ---
    prev_action = np.array(panda.get_state().q)  # Layer 1 active from step 1
    step_count = 0

    try:
        with panda.create_context(
            frequency=CONTROL_FREQ,
            max_runtime=MAX_RUNTIME
        ) as ctx:
            while ctx.ok():
                state = panda.get_state()
                current_q = np.array(state.q)

                obs = build_observation(panda, state, target_pos)
                if vec_norm is not None:
                    obs = vec_norm.normalize_obs(obs)

                raw_action, _ = model.predict(obs, deterministic=True)
                target_q = to_target_q(raw_action, current_q)
                action = safe_action(target_q, current_q, prev_action)

                debug_action(target_q, action, step_count)

                if DRY_RUN:
                    print(f"[DRY] step {step_count}: "
                          f"would send {action.round(3)}")
                else:
                    ctrl.set_control(action)

                prev_action = action
                step_count += 1

                if step_count % 10 == 0:
                    print(f"Step {step_count}: "
                          f"q={current_q.round(3)} "
                          f"action={action.round(3)}")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user (Ctrl+C)")

    finally:
        # --- Step 9: Safe shutdown ---
        print("=" * 50)
        print(f"Total steps executed: {step_count}")
        if not DRY_RUN:
            try:
                panda.stop_controller()
            except Exception as e:
                print(f"stop_controller: {e}")
        print("Returning to start position...")
        try:
            panda.move_to_start()
        except Exception:
            print("move_to_start failed, recovering from error state...")
            panda.recover()
            panda.move_to_start()
        print("Returned to start. Done.")


if __name__ == '__main__':
    main() 
