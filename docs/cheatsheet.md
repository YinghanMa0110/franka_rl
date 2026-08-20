# panda-py Cheatsheet

## Connection
```python
import panda_py
from panda_py import libfranka, controllers, constants

desk = panda_py.Desk(hostname, username, password)
desk.unlock()
desk.activate_fci()

panda = panda_py.Panda(hostname)
gripper = libfranka.Gripper(hostname)
```

## Read State
```python
state = panda.get_state()    # full robot state
state.q                       # 7 joint positions (rad)
state.dq                      # 7 joint velocities (rad/s)
state.tau_J                   # 7 joint torques (Nm)
state.O_T_EE                  # 4x4 end-effector pose matrix

pose = panda.get_pose()       # 4x4 end-effector pose
position = panda.get_position()  # xyz position
orientation = panda.get_orientation()  # quaternion
```

## Motion
```python
panda.move_to_start()                # go to standard start position
panda.move_to_joint_position(q)      # move joints to target angles
panda.move_to_pose(pose)             # move end-effector to target pose
q = panda_py.ik(pose)                # inverse kinematics: pose → joints
T = panda_py.fk(q)                   # forward kinematics: joints → pose
```

## Gripper
```python
gripper.grasp(width, speed, force, epsilon_inner, epsilon_outer)
gripper.move(width, speed)
```

## Real-Time Control (RL Deployment)
```python
ctrl = controllers.JointPosition()        # or CartesianImpedance()
panda.start_controller(ctrl)

with panda.create_context(frequency=20, max_runtime=10) as ctx:
    while ctx.ok():
        state = panda.get_state()
        action = compute_action(state)
        ctrl.set_control(action)
```

## Logging
```python
panda.enable_logging(2000)     # buffer size in steps (2000 = 2s at 1kHz)
panda.move_to_pose(target)
panda.disable_logging()
log = panda.get_log()          # dict with all logged state data
```

## Constants
```python
from panda_py import constants
constants.JOINT_POSITION_START  # standard start position
```

## Stable Baselines3
```python
from stable_baselines3 import PPO

# Train
model = PPO('MultiInputPolicy', env, verbose=1)
model.learn(total_timesteps=3_000_000)
model.save('checkpoint')

# Load and predict
model = PPO.load('checkpoint')
action, _ = model.predict(obs, deterministic=True)
```
