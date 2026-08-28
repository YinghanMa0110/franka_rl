# Development Log — First Manchester Lab Visit (panda-py Basics)

## Overview

First hands-on session with the real Franka Emika Panda robot. Goal: verify basic panda-py control functionality works on real hardware before attempting RL policy deployment. This session focused purely on direct robot control primitives — no policy inference yet.

---

## What Was Tested

### 1. Connection and Basic State

```python
import panda_py
from panda_py import libfranka

panda = panda_py.Panda('192.168.1.8')
gripper = libfranka.Gripper('192.168.1.8')
```

- Connected successfully to robot at `192.168.1.8`
- Confirmed both arm (`panda_py.Panda`) and gripper (`libfranka.Gripper`) need separate connections

### 2. Move to Start + Inverse Kinematics

```python
panda.move_to_start()
pose = panda.get_pose()
pose[2, 3] -= 0.1          # lower z by 10cm
q = panda_py.ik(pose)
panda.move_to_joint_position(q)
```

- `move_to_start()` successfully returns robot to the standard neutral pose
- Confirmed `get_pose()` returns a 4×4 homogeneous transform; index `[2,3]` is the z-coordinate
- IK (`panda_py.ik`) correctly converted the modified Cartesian pose into joint targets, robot moved as expected

### 3. Gripper Control

```python
gripper.grasp(0, 0.2, 10, 0.04, 0.04)   # close
gripper.move(0.08, 0.2)                  # open
```

- Grasp and release both worked correctly on real hardware

### 4. Reading Full Robot State

```python
state = panda.get_state()
```

- Confirmed access to: `state.q` (joint positions), `state.dq` (joint velocities), `state.tau_J` (joint torques), `state.O_T_EE` (end-effector pose, 16 floats)

**Early finding (later confirmed critical):** naively reshaping `O_T_EE` into a 4×4 matrix and extracting `[:3, 3]` for position returned `[0, 0, 0]`. Root cause not yet diagnosed at this stage — flagged for follow-up. (Resolved in the later RL deployment session: `O_T_EE` is column-major; `panda.get_position()` is the correct API to use instead of manual reshaping.)

### 5. Real-Time Control: Cartesian Impedance

```python
from panda_py import controllers

ctrl = controllers.CartesianImpedance(filter_coeff=1.0)
panda.start_controller(ctrl)

pos = panda.get_position()
q = panda.get_orientation()
target = pos.copy()
target[2] += 0.1   # move +10cm in z

with panda.create_context(frequency=1000, max_runtime=10) as ctx:
    while ctx.ok():
        ctrl.set_control(target, q)
```

- Successfully ran a 1kHz real-time control loop for 10 seconds
- Robot moved to the fixed target (+10cm in z) and held position via impedance control
- Confirmed the general pattern for later RL deployment: `start_controller()` must be called before `set_control()` has any effect — `set_control()` alone does nothing without an active controller running in the background thread

---

## Key Takeaways From This Session

1. **Basic connectivity and motion control work reliably** — `move_to_start`, IK-based joint motion, gripper control, and state reading are all confirmed functional on the real robot.
2. **`O_T_EE` extraction bug identified but not yet root-caused** — this became the first thing fixed in the subsequent RL deployment session.
3. **Real-time controller pattern understood**: create controller → `start_controller()` → loop calling `set_control()` inside a `create_context()` block → `stop_controller()` on exit.
4. This session validated the control primitives needed before attempting policy-driven control; the next visit built directly on this to deploy an actual trained RL checkpoint.

---

## What This Enabled Next

This session's confirmed working primitives (`get_position()`, `CartesianImpedance` controller, `create_context()` loop pattern) became the foundation for the deployment script (`deploy_reach_cartesian.py`) used successfully in the following visit to run a trained PPO policy on the real robot. See `logs/manchester_visit_20250827-28.md` for that session.
