# Franka RL Deployment Framework

A reusable framework for deploying **Reinforcement Learning algorithms** on the **Franka Emika Panda** robot.

This project provides a complete pipeline from simulation training to real-world robot deployment:

1. Training RL policies in simulation (`panda-gym`)
2. Validating trained policies
3. Deploying policies on the real Franka Panda using `panda-py`

**Key feature:** Switching between different trained policies only requires changing the checkpoint path.

---

# Current Progress

## Real Robot Setup (Completed)

The real Franka Panda deployment environment has been successfully configured on an NVIDIA Jetson platform.

Completed:

* ✅ Franka Panda communication via FCI
* ✅ `panda-py` installation and setup
* ✅ Robot connection test
* ✅ Move-to-start motion
* ✅ Cartesian impedance control
* ✅ Inverse kinematics testing
* ✅ Gripper control
* ✅ Robot state acquisition

The current system can send Cartesian and joint-space commands to the physical robot and retrieve real-time robot states.

---

# Repository Structure

```
franka_rl/
├── README.md
├── requirements.txt
├── .gitignore
│
├── training/
│   ├── ppo_train.py          # Train PPO policy in simulation
│   └── test_policy.py        # Validate trained policy in simulation
│
├── deployment/
│   ├── franka_rl_deploy.py   # Deploy RL policy on real Franka (main script)
│   ├── state_test.py         # Test: read robot state
│   ├── cart_test.py          # Test: Cartesian impedance control
│   ├── ik_test.py            # Test: inverse kinematics motion
│   └── gripper_test.py       # Test: gripper control
│
├── config/
│   └── joint_limits.yaml     # Joint safety limits
│
├── checkpoints/
│   └── .gitkeep              # Placeholder (trained weights go here)
│
└── docs/
    ├── user_guide.md         # Step-by-step usage guide
    └── cheatsheet.md         # panda-py API quick reference
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
* FCI is enabled
* Jetson and robot are connected to the same network
* access Franka desk by robot IP: 192.168.1.8

Update the robot IP if required:

```python
robot = panda_py.Panda("192.168.1.8")
```

Test connection:

```bash
ping 192.168.1.8
```

---

# Real Robot Tests

## Move Robot to Start Position

```bash
python3 deployment/move_to_start.py
```

Moves the robot to the predefined initial configuration.

---

## Cartesian Impedance Control

```bash
python3 deployment/cart_test.py
```

Features:

* Starts Cartesian impedance controller
* Reads current end-effector pose
* Generates Cartesian target position
* Executes control loop at 1000 Hz

---

## Inverse Kinematics

```bash
python3 deployment/ik_test.py
```

Pipeline:

```
Cartesian pose
      ↓
Inverse Kinematics
      ↓
Joint position command
      ↓
Franka Panda
```

---

## Gripper Control

```bash
python3 deployment/gripper_test.py
```

Tests:

* Gripper grasp
* Gripper open/close motion

---

## Robot State

```bash
python3 deployment/state_test.py
```

Retrieves:

* Joint positions
* Joint velocities
* Joint torques
* End-effector pose
* End-effector Cartesian position

These states will be used as observations for reinforcement learning.

---

# RL Training Pipeline

## 1. Train Policy in Simulation

```bash
python training/ppo_train.py
```

Current simulation environment:

* panda-gym
* Stable Baselines3 PPO

---

## 2. Validate Policy

```bash
python training/test_policy.py
```

---

## 3. Deploy on Real Franka

```bash
python deployment/franka_rl_deploy.py
```

Deployment pipeline:

```
Simulation
    |
    ↓
Trained RL Policy
    |
    ↓
Checkpoint
    |
    ↓
Real Franka Panda
    |
    ↓
panda-py Controller
```

---

# Deploy Different Algorithms

Changing algorithms does not require modifying the deployment framework.

Simply update:

```python
CHECKPOINT_PATH = "checkpoints/your_new_checkpoint"
```

The same deployment interface can be reused for different policies.

---

# Safety

The deployment framework includes multiple safety layers:

## 1. Joint Limit Clipping

Conservative software limits prevent the robot from reaching unsafe joint configurations.

## 2. Action Smoothing

Limits sudden changes between consecutive actions.

## 3. Deviation Limit

Prevents large jumps from the current robot state.

Additionally, `panda-py` provides built-in safety mechanisms including virtual joint walls.

---

# Tech Stack

## Simulation

* Stable Baselines3
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

# References

* panda-py Documentation
  https://jeanelsner.github.io/panda-py/

* Franka Documentation
  https://frankaemika.github.io/docs/

* Stable Baselines3 Documentation
  https://stable-baselines3.readthedocs.io/

* Lobbezoo & Kwon (2023).
  "Simulated and Real Robotic Reach, Grasp, and Pick-and-Place Using Combined RL and Traditional Controls."

---

# Author

Yinghan Ma
MEng Robotics & AI, UCL

In collaboration with Dr. Mingfei Sun's group, University of Manchester

