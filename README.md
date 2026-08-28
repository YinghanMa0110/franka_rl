# Franka RL Deployment Framework

A reusable framework for deploying **Reinforcement Learning algorithms** on the **Franka Emika Panda** robot.

This project provides a complete pipeline from simulation training to real-world robot deployment:

1. Training RL policies in simulation (`panda-gym`, SB3 or CleanRL)
2. Validating trained policies
3. Deploying policies on the real Franka Panda using `panda-py`

**Key feature:** Switching between different trained policies only requires changing the checkpoint path.

---

# Current Progress

## Real Robot Setup (Completed)

The real Franka Panda deployment environment has been successfully configured on an NVIDIA Jetson platform.

* ✅ Franka Panda communication via FCI
* ✅ `panda-py` installation and setup
* ✅ Robot connection test
* ✅ Move-to-start motion
* ✅ Cartesian impedance control
* ✅ Inverse kinematics testing
* ✅ Gripper control
* ✅ Robot state acquisition

## RL Policy Deployment (Completed)

Successfully deployed a trained PPO policy on the real robot:

* ✅ Loaded SB3 checkpoint (PandaReach-v3, dense reward, Cartesian action) and ran real-time inference on hardware
* ✅ `CartesianImpedance` controller driving the policy's output at 20 Hz
* ✅ Real robot moved end-effector toward target: final distance **0.066 m** (sim threshold: 0.05 m)
* ✅ Full safety pipeline verified: joint limit clipping, action smoothing, deviation limiting, dry-run mode
* ✅ Coordinate conversion between real robot frame and panda-gym's training frame (`BASE_OFFSET`)

See `logs/manchester_visit_20250827-28.md` for the full deployment log.

## CleanRL Investigation (In Progress)

Per guidance to move beyond SB3's "black box" for algorithm-level research (e.g. modifying PPO's update rule), switched to CleanRL's single-file, fully transparent implementation.

* ✅ Trained baseline PPO (`ppo_continuous_action.py`) on `PandaReach-v3`, 200k steps
* ✅ Fixed 3 bugs found by inspecting internals:
  - gymnasium API change (`TransformObservation` requires explicit `observation_space`)
  - episode-end logging broken by another gymnasium API change (`infos["episode"]` replacing `infos["final_info"]`)
  - `reward_type` silently defaulting to **sparse** instead of dense — reproduced the same sparse-vs-dense gap seen earlier with SB3
* ✅ After fixes, training curve is healthy: `episodic_return` improves from ~-50 (sparse) to -0.1 to -0.35 (dense, near end of training)
* ⚠️ **Known issue**: evaluation success rate is unstable (10–50% across runs) because `NormalizeObservation`'s running statistics are reset on every fresh environment instead of being saved from training and reloaded at eval/deployment time. Root-caused, not yet fixed.

See `logs/manchester_visit_20250827-28.md` for the full debugging log.

---

# Repository Structure

```
franka_rl/
├── README.md
├── requirements.txt
├── .gitignore
│
├── training/
│   ├── ppo_train.py              # Train PPO policy in simulation (SB3)
│   ├── test_policy.py            # Validate SB3-trained policy in simulation
│   └── cleanrl/
│       └── ppo_continuous_action.py  # Single-file, transparent PPO (CleanRL)
│
├── deployment/
│   ├── franka_rl_deploy.py       # Joint-space deployment template (for joint-action policies)
│   ├── deploy_reach_cartesian.py # Cartesian-space deployment (used for the successful real-robot test)
│   └── test/
│       ├── move_to_start.py      # Test: basic motion
│       ├── state_test.py         # Test: read robot state
│       ├── cart_test.py          # Test: Cartesian impedance control
│       ├── ik_test.py            # Test: inverse kinematics motion
│       └── gripper_test.py       # Test: gripper control
│
├── logs/
│   ├── manchester_visit_first_session.md   # Visit 1: panda-py basics verification
│   ├── manchester_visit_20250827-28.md     # Visit 2: RL deployment + CleanRL debugging
│   └── data/
│       └── reach_deployment_20250827.npy   # Logged robot state from the successful deployment
│
├── config/
│   └── joint_limits.yaml         # Joint safety limits
│
├── checkpoints/
│   ├── panda_reach_ppo_40000_steps.zip     # SB3 checkpoint (Cartesian control, 90% sim success)
│   └── ppo_continuous_action.cleanrl_model # CleanRL checkpoint (dense reward, 200k steps)
│
└── docs/
    ├── user_guide.md             # Step-by-step usage guide
    └── cheatsheet.md             # panda-py API quick reference
```

---

# Hardware Setup

## Robot

* **Robot:** Franka Emika Panda (7-DOF)
* **Interface:** FCI + panda-py
* **Computer:** NVIDIA Jetson
* **Camera:** Intel RealSense D435i

The Jetson environment has already been configured with all required dependencies.

---

# Quick Start

## 1. Connect to Franka Panda

Make sure:

* Robot is powered on
* FCI is enabled (unlock brakes + activate FCI via the browser-based Desk interface at the robot's IP)
* Jetson and robot are connected to the same network

```bash
ping 192.168.1.8
```

Update the robot IP in scripts if required (default: `192.168.1.8`).

---

# Real Robot Tests

## Move Robot to Start Position
```bash
python3 deployment/test/move_to_start.py
```

## Cartesian Impedance Control
```bash
python3 deployment/test/cart_test.py
```
Starts a `CartesianImpedance` controller, reads current EE pose, generates a target, and runs the control loop at up to 1000 Hz.

## Inverse Kinematics
```bash
python3 deployment/test/ik_test.py
```
Pipeline: Cartesian pose → IK → joint position command → executed on the robot.

## Gripper Control
```bash
python3 deployment/test/gripper_test.py
```

## Robot State
```bash
python3 deployment/test/state_test.py
```
Retrieves joint positions/velocities/torques and end-effector pose — these are the raw signals used to build RL observations.

---

# RL Training Pipeline

## Option A: Stable Baselines3

```bash
python3 training/ppo_train.py       # train
python3 training/test_policy.py     # validate in sim
```

## Option B: CleanRL (algorithm-transparent, recommended for research/modification)

```bash
python3 training/cleanrl/ppo_continuous_action.py --exp_name my_run
```

> **Important:** `PandaReach-v3` defaults to **sparse** reward. Confirm `make_env()` passes `reward_type='dense'` to `gym.make()` — this single flag is the difference between the policy learning almost nothing (`episodic_return≈-50`) and learning successfully (`episodic_return≈-0.1`).

## Deploy on Real Franka

```bash
python3 deployment/deploy_reach_cartesian.py
```

Pipeline: `Simulation → Trained Policy Checkpoint → Real Franka Panda → panda-py CartesianImpedance controller`

Always run with `DRY_RUN = True` first to verify the policy produces sane actions before touching the hardware.

---

# Deploy a Different Algorithm

Swapping algorithms/checkpoints does **not** require touching the safety/control loop. Just:

1. Place the new checkpoint in `checkpoints/`
2. Update the checkpoint path in the deployment script
3. If the new policy uses a different observation format or action semantics (e.g. joint-space instead of Cartesian, or a different obs vector), update only the `build_observation()` / action-conversion function accordingly

---

# Safety

The deployment framework includes multiple safety layers:

1. **Joint Limit Clipping** — conservative software limits (with margin) prevent unsafe joint configurations
2. **Action Smoothing** — limits sudden changes between consecutive actions
3. **Deviation Limit** — prevents large jumps from the current robot state

`panda-py` additionally provides built-in virtual joint walls at the controller level as a further safeguard.

---

# Tech Stack

## Simulation / Training
* Stable Baselines3 (initial baseline, "black box")
* CleanRL (single-file PPO, used for algorithm-level debugging and future modification e.g. pi-PG)
* Gymnasium
* panda-gym

## Real Robot Deployment
* panda-py
* libfranka
* Franka FCI

## Hardware
* Franka Emika Panda
* NVIDIA Jetson
* Intel RealSense D435i

---

# Known Issues / Next Steps

1. **CleanRL evaluation instability** — `NormalizeObservation` statistics must be saved at the end of training and reloaded during evaluation/deployment for consistent, accurate success-rate measurement. Currently statistics reset on every fresh environment instance.
2. **Sim-to-real precision gap** — best real-robot result (0.066 m) is just outside the simulation success threshold (0.05 m). Candidates: tune `ACTION_SCALE`/`MAX_RUNTIME`, verify `BASE_OFFSET` coordinate conversion precision, or apply domain randomization during training.
3. **Real-robot fine-tuning** — exploring continuing training directly on hardware to close the sim-to-real gap. The reward function needed for this has been confirmed directly from panda-gym source (`panda_gym/envs/tasks/reach.py`):
   ```python
   reward = -distance(achieved_goal, desired_goal)          # dense
   success = distance(achieved_goal, desired_goal) < 0.05
   ```
   Still need to design a safe online-learning protocol before attempting this on hardware.
4. **PPO algorithm modification** — next discussion with Hossein: which part of CleanRL's PPO to modify first (clipped surrogate objective, GAE advantage estimation, or a pi-PG-style curvature-aware update).

---

# References

* panda-py Documentation — https://jeanelsner.github.io/panda-py/
* Franka Documentation — https://frankaemika.github.io/docs/
* Stable Baselines3 Documentation — https://stable-baselines3.readthedocs.io/
* CleanRL Documentation — https://docs.cleanrl.dev/
* Lobbezoo & Kwon (2023). "Simulated and Real Robotic Reach, Grasp, and Pick-and-Place Using Combined RL and Traditional Controls." *Robotics*.

---

# Author

Yinghan Ma
MEng Robotics & AI, UCL

In collaboration with Dr. Mingfei Sun's group, University of Manchester
