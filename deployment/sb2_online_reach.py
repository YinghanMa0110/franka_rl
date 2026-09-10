import os
import csv
import time
import logging
from datetime import datetime

import numpy as np
import gymnasium as gym
from gymnasium import spaces

import panda_py
from panda_py import controllers

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

logging.basicConfig(level=logging.INFO)


# ============================================================
# Configuration（复用 deploy 脚本）
# ============================================================
HOSTNAME        = "192.168.1.8"
CHECKPOINT_PATH = "checkpoints/panda_reach_ppo_40000_steps"   # 真机验证过的 SB3 checkpoint

BASE_OFFSET = np.array([-1, 0.0, 0.0])   # panda-gym world <- real base，仅 obs 用

ACTION_SCALE   = 0.05
MAX_STEP       = 0.02    # 单步 3D 位移上限 (m)
CONTROL_FREQ   = 20       # Hz
GOAL_THRESHOLD = 0.05     # 成功阈值 (m)

TRANSLATIONAL_STIFFNESS = 600.0
ROTATIONAL_STIFFNESS    = 30.0

# ---- 安全：workspace 硬边界（deploy 脚本没有，训练探索必须加！按 lab 实际范围收紧）----
WORKSPACE_LOW  = np.array([0.25, -0.25, 0])
WORKSPACE_HIGH = np.array([0.65,  0.25, 0.60])

# ---- 训练 goal 采样范围（真机 base frame，相对 start EE，匹配 benchmark 的 ±0.08）----
GOAL_LOW  = np.array([-0.20, -0.20, -0.20])
GOAL_HIGH = np.array([ 0.20,  0.20,  0.20])

# ---- fine-tune 超参（比 sim 保守）----
TOTAL_TIMESTEPS = 3000    # 第一次务必先设 300 试跑，确认闭环再放大到几千
LEARNING_RATE   = 1e-4
N_STEPS         = 200     # 一次 update 前收集步数（4 episodes）
BATCH_SIZE      = 50
N_EPOCHS        = 5
CLIP_RANGE      = 0.1
MAX_EP_STEPS    = 50      # 每 episode 步数（50 @ 20Hz = 2.5s）

SAVE_DIR    = "checkpoints"
SAVE_PREFIX = "reach_online"
RESULT_DIR  = "results"
TRAIN_CURVE = os.path.join(RESULT_DIR, "online_train_curve.csv")


# ============================================================
# Coordinate Conversion + Observation（复用 deploy 脚本）
# ============================================================
def real_to_sim(real_pos):
    return real_pos + BASE_OFFSET


def build_obs(current_pos_real, current_vel, target_real):
    current_sim = real_to_sim(current_pos_real)
    target_sim  = real_to_sim(target_real)
    return {
        "observation":   np.concatenate([current_sim, current_vel]).astype(np.float32),
        "achieved_goal": current_sim.astype(np.float32),
        "desired_goal":  target_sim.astype(np.float32),
    }


# ============================================================
# Online Reach Env
# ============================================================
class FrankaReachOnlineEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, panda):
        super().__init__()
        self.panda = panda
        self.dt = 1.0 / CONTROL_FREQ
        self.ctrl = None
        self.controller_running = False

        self.action_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Dict({
            "observation":   spaces.Box(-10.0, 10.0, shape=(6,), dtype=np.float32),
            "achieved_goal": spaces.Box(-10.0, 10.0, shape=(3,), dtype=np.float32),
            "desired_goal":  spaces.Box(-10.0, 10.0, shape=(3,), dtype=np.float32),
        })

        self.q0 = None
        self.target_real = None
        self.prev_pos = None
        self.step_count = 0
        self.global_step = 0
        self.episode_idx = 0
        self.ep_reward = 0.0

        os.makedirs(RESULT_DIR, exist_ok=True)
        self._curve_header_written = os.path.exists(TRAIN_CURVE)

    def _stop_controller(self):
        if self.controller_running:
            try:
                self.panda.stop_controller()
            except Exception:
                pass
            self.controller_running = False

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self._stop_controller()
        try:
            self.panda.move_to_start()
        except Exception:
            print("[reset] move_to_start failed -> recover()")
            self.panda.recover()
            self.panda.move_to_start()
        time.sleep(0.5)

        x0 = self.panda.get_position().copy()
        self.q0 = self.panda.get_orientation().copy()
        self.prev_pos = x0.copy()

        # 采样 goal（真机 frame），clip 到 workspace
        offset = self.np_random.uniform(low=GOAL_LOW, high=GOAL_HIGH)
        self.target_real = np.clip(x0 + offset, WORKSPACE_LOW, WORKSPACE_HIGH)

        # 起控制器（stiffness 沿用 deploy 的 900）
        self.ctrl = controllers.CartesianImpedance()
        self.ctrl.set_impedance(np.diag([
            TRANSLATIONAL_STIFFNESS, TRANSLATIONAL_STIFFNESS, TRANSLATIONAL_STIFFNESS,
            ROTATIONAL_STIFFNESS, ROTATIONAL_STIFFNESS, ROTATIONAL_STIFFNESS,
        ]))
        self.panda.start_controller(self.ctrl)
        self.controller_running = True

        self.step_count = 0
        self.ep_reward = 0.0
        return build_obs(x0, np.zeros(3), self.target_real), {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float64).flatten()

        current = self.panda.get_position().copy()
        # action -> 位移，3D norm 裁剪（安全层 1，沿用 deploy 逻辑）
        disp = action * ACTION_SCALE
        n = np.linalg.norm(disp)
        if n > MAX_STEP:
            disp = disp / n * MAX_STEP
        # workspace 硬裁剪（安全层 2，训练必须）
        x_d = np.clip(current + disp, WORKSPACE_LOW, WORKSPACE_HIGH)

        terminated = False
        truncated = False
        try:
            self.ctrl.set_control(x_d, self.q0)
            time.sleep(self.dt)
        except Exception as e:
            print(f"[step] controller error: {e} -> truncate")
            truncated = True

        new_pos = self.panda.get_position().copy()
        vel = (new_pos - self.prev_pos) * CONTROL_FREQ
        self.prev_pos = new_pos.copy()

        # dense reward，真机 frame（offset 相消，与 BASE_OFFSET 精度无关）
        d = np.linalg.norm(new_pos - self.target_real)
        reward = -float(d)

        self.step_count += 1
        self.global_step += 1
        self.ep_reward += reward

        if d < GOAL_THRESHOLD:
            terminated = True
        if self.step_count >= MAX_EP_STEPS:
            truncated = True

        if terminated or truncated:
            self._log_episode(d, bool(terminated))

        info = {"distance": d, "is_success": float(d < GOAL_THRESHOLD)}
        return build_obs(new_pos, vel, self.target_real), reward, terminated, truncated, info

    def _log_episode(self, final_d, success):
        self.episode_idx += 1
        row = {
            "wall_time": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "global_step": self.global_step,
            "episode": self.episode_idx,
            "ep_steps": self.step_count,
            "final_distance_cm": round(final_d * 100, 3),
            "success": int(success),
            "ep_reward": round(self.ep_reward, 4),
        }
        with open(TRAIN_CURVE, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=row.keys())
            if not self._curve_header_written:
                w.writeheader()
                self._curve_header_written = True
            w.writerow(row)
        print(f"[train] ep {self.episode_idx:3d}  step {self.global_step:5d}  "
              f"dist {final_d*100:5.2f}cm  success {int(success)}  R {self.ep_reward:.2f}")

    def close(self):
        self._stop_controller()
        try:
            self.panda.move_to_start()
        except Exception:
            try:
                self.panda.recover()
                self.panda.move_to_start()
            except Exception:
                pass


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 70)
    print("PPO REAL ROBOT ONLINE FINE-TUNING")
    print("=" * 70)

    print(f"Connecting to robot: {HOSTNAME}")
    panda = panda_py.Panda(HOSTNAME)
    print("Robot connected.")

    env = FrankaReachOnlineEnv(panda)

    print(f"Warm-starting from: {CHECKPOINT_PATH}")
    # 从 sim checkpoint 接着训（不从零）。若 n_steps 报 buffer 错，改用：
    #   m = PPO("MultiInputPolicy", env, n_steps=N_STEPS, batch_size=BATCH_SIZE,
    #           learning_rate=LEARNING_RATE, clip_range=CLIP_RANGE, n_epochs=N_EPOCHS)
    #   m.set_parameters(CHECKPOINT_PATH)
    model = PPO.load(
        CHECKPOINT_PATH,
        env=env,
        custom_objects={
            "learning_rate": LEARNING_RATE,
            "n_steps": N_STEPS,
            "batch_size": BATCH_SIZE,
            "clip_range": CLIP_RANGE,
            "n_epochs": N_EPOCHS,
        },
    )
    print("Policy loaded.")
    print(f"  total_timesteps={TOTAL_TIMESTEPS}  ep_len={MAX_EP_STEPS}  "
          f"freq={CONTROL_FREQ}Hz  K={TRANSLATIONAL_STIFFNESS}")
    print(f"  workspace={WORKSPACE_LOW} .. {WORKSPACE_HIGH}")
    input("\n手放急停，检查 workspace，Enter 开始 fine-tune (Ctrl+C 中止): ")

    ckpt_cb = CheckpointCallback(
        save_freq=N_STEPS, save_path=SAVE_DIR, name_prefix=SAVE_PREFIX
    )

    try:
        model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=ckpt_cb)
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        final_path = os.path.join(SAVE_DIR, f"{SAVE_PREFIX}_final")
        model.save(final_path)
        print(f"Saved: {final_path}")
        print(f"Train curve: {TRAIN_CURVE}")
        env.close()
        print("Robot returned to start. Done.")


if __name__ == "__main__":
    main()
