# User Guide: Franka RL Deployment

## 1. Environment Setup

### Install Dependencies
```bash
pip install panda-python stable-baselines3 gymnasium panda-gym numpy
```

### Optional (for camera)
```bash
pip install pyrealsense2 opencv-python
```

---

## 2. Training a Policy (Simulation)

### Train PPO on PandaReach
```bash
cd training
python ppo_train.py
```

This will:
- Train for 3M timesteps with dense reward
- Auto-save checkpoints every 10k steps to `checkpoints/`
- Save final model when complete

### Validate the Policy
```bash
python test_policy.py
```

Expected output: 100% success rate (10/10 episodes).

### Training Tips
- Always use `reward_type='dense'` (sparse reward fails to converge)
- PandaReach: ~100k steps to converge
- PandaPickAndPlace: ~8M steps to converge

---

## 3. Connecting to the Real Franka

### Prerequisites
- Franka Emika Panda robot powered on
- Workstation connected to robot via Ethernet
- Robot IP address, Desk username and password (ask lab manager)

### Connection Steps

**Step 1:** Open Desk in browser at `https://<robot_ip>` and verify robot status.

**Step 2:** Run the deployment script:
```bash
cd deployment
python franka_rl_deploy.py
```

The script will:
1. Connect to Desk and unlock brakes
2. Activate FCI
3. Connect to the robot
4. Load the checkpoint
5. Move to start position
6. Wait for your confirmation (Press Enter)
7. Run the control loop
8. Return to start when done

### First-Time Testing Checklist
- [ ] Know where the emergency stop button is
- [ ] Keep your hand near the emergency stop
- [ ] Have someone supervise
- [ ] Use conservative parameters (already set as default)
- [ ] Start with `move_to_start()` only before running the full loop

---

## 4. Deploying a New Algorithm

### Step 1: Place checkpoint in `checkpoints/`

### Step 2: Change one line in `deployment/franka_rl_deploy.py`:
```python
CHECKPOINT_PATH = 'checkpoints/your_new_checkpoint'
```

### Step 3: Run:
```bash
python deployment/franka_rl_deploy.py
```

### Important
If the new algorithm uses a different observation format, update `build_observation()` in the deployment script to match.

---

## 5. Safety

### Three Layers of Protection

| Layer | Protection | Default Value |
|-------|-----------|---------------|
| 1 | Joint limit clipping | Hardware limits - 0.2 rad margin |
| 2 | Action smoothing | Max 0.01 rad/step |
| 3 | Deviation limit | Max 0.2 rad from current position |

### Adjusting Parameters

After successful initial tests, gradually increase:
```
First test:  MAX_DELTA=0.01, CONTROL_FREQ=10,  MAX_RUNTIME=5
Second test: MAX_DELTA=0.02, CONTROL_FREQ=20,  MAX_RUNTIME=10
Third test:  MAX_DELTA=0.05, CONTROL_FREQ=50,  MAX_RUNTIME=30
```

**Change only one parameter at a time.**

### Emergency Stop
- Press the physical kill switch immediately if anything goes wrong
- Press Ctrl+C in the terminal — the script will safely return to start position

---

## 6. Troubleshooting

### "Desk connection failed"
- Check the robot IP address
- Check username/password
- Make sure robot is powered on

### "FCI activation failed"
- Someone else may be using the robot
- Try unlocking brakes in the Desk browser interface first

### "Action was clipped" warnings
- Normal during first tests
- If excessive, the observation format may not match training

### Robot moves erratically
- Immediately press emergency stop
- Reduce MAX_DELTA and CONTROL_FREQ
- Check that `build_observation()` matches training format

### Robot doesn't move
- Check that the controller is started (`panda.start_controller(ctrl)`)
- Check that FCI is activated
- Check terminal for error messages

---

## 7. Hardware Reference

### Franka Emika Panda Specs
| Parameter | Value |
|-----------|-------|
| Degrees of freedom | 7 |
| Payload | 3 kg |
| Reach | 855 mm |
| Pose repeatability | ±0.1 mm |
| Control frequency | 1 kHz (FCI) |
| Collision detection | <2 ms |

### Software Stack
```
Your Python code
    ↓
panda-py (Python interface)
    ↓
libfranka (C++ library)
    ↓
FCI (1 kHz communication)
    ↓
Real Franka
```
