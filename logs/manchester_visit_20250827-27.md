# Development Log — Manchester Lab Visit (2 days)

## Overview

First hands-on session with the real Franka Emika Panda robot at the University of Manchester lab. Goals: (1) verify the deployment framework works end-to-end on real hardware, (2) begin comparing SB3 vs CleanRL for algorithm-level debugging as suggested by Hossein.

---

## Day 1: Real Robot Deployment

### Environment Setup
- Connected to Jetson (tegra-ubuntu) via VS Code Remote
- Cloned `franka_rl` repo, installed dependencies (panda-py, stable-baselines3, gymnasium, panda-gym)
- Robot IP: `192.168.1.8`
- Used browser-based Desk interface to unlock brakes and activate FCI manually (skipped in-code Desk connection)

### Bug Fixes During Deployment

**1. `O_T_EE` position extraction returning `[0, 0, 0]`**
- Root cause: `O_T_EE` is stored column-major; direct `.reshape(4,4)` scrambles the data
- Fix: use `panda.get_position()` (library handles the correct transform) instead of manually parsing `O_T_EE`

**2. `zero_jacobian()` TypeError — missing argument**
- Root cause: newer panda-py API requires a `Frame` parameter
- Fix: `panda.get_model().zero_jacobian(libfranka.Frame.kEndEffector, state)`

**3. Action shape mismatch (7 expected, got 3)**
- Root cause: the SB3 checkpoint (`panda_reach_ppo_40000_steps`) was trained with **Cartesian end-effector control** (3D action = xyz displacement), not joint control (7D)
- Fix: switched from `franka_rl_deploy.py` (joint-position script) to `deploy_reach_cartesian.py`, using panda-py's `CartesianImpedance` controller and `panda.get_position()` directly

**4. VS Code save issues**
- File edits weren't persisting to disk (editor showed unsaved changes / diff view confusion)
- Resolved by editing directly in terminal via `nano` when VS Code save was unreliable

### Result: First Successful Real-Robot RL Deployment

```
Checkpoint: SB3 PPO, PandaReach-v3, dense reward, 40,000 timesteps
Controller: CartesianImpedance
Control frequency: 20 Hz
Start EE position: [0.307, -0.000, 0.487]
Target position:   [0.407,  0.100, 0.387]
Final distance:    0.066 m (goal threshold: 0.05 m)
```

The policy successfully moved the real robot's end-effector toward the target over 300 control steps, confirming the full pipeline works: connect → load checkpoint → build observation → policy inference → safety filtering → execute on hardware → safe shutdown.

**Sim-to-real gap observed:** simulation success rate was 90-100%; real robot reached 0.066 m vs the 0.05 m threshold — close but not within tolerance. Consistent with the sim-to-real degradation reported in Lobbezoo & Kwon (2023).

### Coordinate System Note
panda-gym's PandaReach places the simulated robot base at world-frame offset `[-0.6, 0, 0]`. A `BASE_OFFSET` correction was applied when converting real robot coordinates into the frame the policy was trained on.

---

## Day 2: CleanRL Investigation (per Hossein's guidance)

**Context:** Hossein noted that SB3 is a "black box" — unsuitable for algorithm-level research (e.g. modifying PPO's update rule for the pi-PG approach), since its internals aren't easily inspectable or modifiable. CleanRL was recommended instead, as each algorithm is a single, fully transparent file.

### Setup
- Installed CleanRL's `ppo_continuous_action.py` (single-file PPO implementation for continuous action spaces)
- Added `import panda_gym` and set `env_id = "PandaReach-v3"` as default

### Bug Chain (in order of discovery)

**1. `TransformObservation` API change**
- Newer gymnasium requires an explicit `observation_space` argument
- Fix: `gym.wrappers.TransformObservation(env, fn, env.observation_space)`

**2. `episodic_return` never printed during training**
- Root cause: CleanRL's original code checks `if "final_info" in infos`, but current gymnasium's `SyncVectorEnv` returns episode stats directly under an `"episode"` key (with a `"_episode"` boolean mask), not nested under `"final_info"`
- Fix: rewrote the check to `if "episode" in infos: ... infos["_episode"]`
- Verified via isolated test: single-env `RecordEpisodeStatistics` does include `info["episode"]`; vectorized env exposes `infos["episode"]["r"]` as an array with a `_episode` mask

**3. First training run: `episodic_return` stuck at `-50.0`, success rate 10% after 200k steps**
- Root cause: `make_env()` called `gym.make(env_id)` **without** `reward_type='dense'`, silently defaulting to **sparse** reward
- This exactly reproduced the earlier finding from SB3 experiments (sparse reward on FetchReach: 0% success after 100k steps; dense reward: 100% success after 112k steps)
- Fix: added `reward_type='dense'` to both `gym.make()` call sites (normal and video-capture branches)

**4. Retrained with dense reward (200k steps)**
- `episodic_return` improved from `~-50` (sparse, essentially always failing) to `-0.1` to `-0.35` by the end of training — a strong, healthy learning curve, comparable in shape to the SB3 training curve
- Checkpoint saved successfully: `ppo_continuous_action.cleanrl_model`

**5. Evaluation instability: success rate ranged 10%–50% across repeated test runs**
- Diagnosis: the training wrapper chain includes `NormalizeObservation`, which maintains **running mean/variance statistics** accumulated over the full 200k-step training run
- The evaluation script only replicated `FlattenObservation` + `ClipAction` + `NormalizeObservation`, but the normalization statistics are **reset from scratch** on every fresh environment instance — they are never saved from training and reloaded for evaluation
- This means the policy receives observations normalized against a different (and effectively random, freshly-initializing) distribution than the one it was trained on, explaining the inconsistent results (10%, 20%, 50%, 12% across different runs/episode counts)

**Status: unresolved.** Next step is to persist `obs_rms` (mean/var) from the training `NormalizeObservation` wrapper alongside the model checkpoint, and load those exact statistics at evaluation/deployment time.

### Comparison Summary

| | SB3 (black box) | CleanRL (transparent) |
|---|---|---|
| Reach success rate (sim) | 90% (10/10 test) | 50% best case / unstable (10–50%) pending normalization fix |
| Training steps used | 40,000 | 200,000 |
| Root causes surfaced | — | 3 distinct bugs found and fixed by inspecting internals |

This is the concrete value Hossein pointed to: SB3 likely handles wrapper/normalization bookkeeping internally in ways that are invisible (and un-debuggable) to the user. CleanRL exposed exactly where that bookkeeping was going wrong.

---

## Open Items for Next Visit

1. **Fix normalization mismatch**: save `obs_rms` statistics at end of CleanRL training; load them in the evaluation/deployment script for an accurate, stable success-rate measurement.
2. **Algorithm modification discussion with Hossein**: confirm which part of PPO to modify first — the clipped surrogate objective (`pg_loss1`/`pg_loss2`), the GAE advantage estimation, or something specific to the pi-PG curvature approach.
3. **Real-robot fine-tuning**: Hossein raised the idea of continuing training directly on the real robot (not just inference) to close the sim-to-real gap. Reward function confirmed from panda-gym source (`panda_gym/envs/tasks/reach.py`):
   ```python
   reward = -distance(achieved_goal, desired_goal)   # dense
   success = distance(achieved_goal, desired_goal) < 0.05
   ```
   This is directly reusable on the real robot (just needs real EE position and target position). Still need to design a safe online-learning protocol before attempting this on hardware.
4. Improve real-robot precision: current best result (0.066 m) is just outside the 0.05 m success threshold — worth tuning `ACTION_SCALE` / `MAX_RUNTIME` further.
