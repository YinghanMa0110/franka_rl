import time

from panda_py import controllers

import gymnasium as gym
import numpy as np
from gymnasium import spaces


# ============================================================
# Default configuration
# ============================================================

CONTROL_FREQ = 20

MAX_EP_STEPS = 50

GOAL_THRESHOLD = 0.02       # 2 cm

# panda-gym joint-control semantics:
#
# action ∈ [-1, 1]^7
# delta_q = action * 0.05 rad
# q_target = q_current + delta_q
#
JOINT_ACTION_SCALE = 0.05


# ============================================================
# Orientation safety
#
# The original PandaReach task only optimizes EE position.
# Orientation is NOT part of the learned observation/reward.
#
# For the real robot we monitor orientation separately and
# terminate the episode if the end-effector rotates too far
# away from its reset orientation.
# ============================================================

MAX_ORIENTATION_DEVIATION_DEG = 20.0


# ============================================================
# Franka joint limits
# ============================================================

JOINT_LOW = np.array([
    -2.8973,
    -1.7628,
    -2.8973,
    -3.0718,
    -2.8973,
    -0.0175,
    -2.8973,
], dtype=np.float64)


JOINT_HIGH = np.array([
    2.8973,
    1.7628,
    2.8973,
    -0.0698,
    2.8973,
    3.7525,
    2.8973,
], dtype=np.float64)


JOINT_MARGIN = 0.10


SAFE_JOINT_LOW = (
    JOINT_LOW
    + JOINT_MARGIN
)


SAFE_JOINT_HIGH = (
    JOINT_HIGH
    - JOINT_MARGIN
)


# ============================================================
# Workspace safety
# ============================================================

WORKSPACE_LOW = np.array([
    0.25,
    -0.25,
    0.00,
], dtype=np.float64)


WORKSPACE_HIGH = np.array([
    0.65,
    0.25,
    0.60,
], dtype=np.float64)


# ============================================================
# Coordinate conversion
# ============================================================

BASE_OFFSET = np.array([
    -1.0,
    0.0,
    0.0,
], dtype=np.float64)


class FrankaReachJointEnv(gym.Env):
    """
    Shared real-robot Reach environment for CleanRL PPO and SAC.

    Raw observation:
        12D flattened PandaReach-style observation

        [
            achieved_goal (3),
            desired_goal  (3),
            observation   (6)
        ]

    where:

        observation =
            [EE position, EE velocity]

    Action:
        7D normalized joint action

        action ∈ [-1, 1]^7

        delta_q =
            action * 0.05 rad

        q_target =
            q_current + delta_q

    Reward:
        -Euclidean EE-to-goal distance

    Success:
        distance < 0.02 m

    Real-robot safety:
        - joint limits
        - Cartesian workspace limits
        - EE orientation deviation limit

    IMPORTANT:
        Orientation monitoring is ONLY a safety layer.

        It is not added to the 12D policy observation and does
        not change the simulation-trained reward.
    """

    metadata = {
        "render_modes": []
    }


    def __init__(
        self,
        panda,
        *,
        goal_low=None,
        goal_high=None,
        goal_threshold=GOAL_THRESHOLD,
        max_ep_steps=MAX_EP_STEPS,
        control_freq=CONTROL_FREQ,
        joint_action_scale=JOINT_ACTION_SCALE,
        workspace_low=WORKSPACE_LOW,
        workspace_high=WORKSPACE_HIGH,
        safe_joint_low=SAFE_JOINT_LOW,
        safe_joint_high=SAFE_JOINT_HIGH,
        base_offset=BASE_OFFSET,
        max_orientation_deviation_deg=
            MAX_ORIENTATION_DEVIATION_DEG,
    ):

        super().__init__()


        self.panda = panda


        # ====================================================
        # Task
        # ====================================================

        self.goal_threshold = float(
            goal_threshold
        )


        self.max_ep_steps = int(
            max_ep_steps
        )


        self.control_freq = float(
            control_freq
        )


        self.dt = (
            1.0
            / self.control_freq
        )


        self.joint_action_scale = float(
            joint_action_scale
        )


        # ====================================================
        # Goal distribution
        # ====================================================

        if goal_low is None:

            goal_low = np.array([
                -0.08,
                -0.08,
                -0.08,
            ])


        if goal_high is None:

            goal_high = np.array([
                0.08,
                0.08,
                0.08,
            ])


        self.goal_low = np.asarray(
            goal_low,
            dtype=np.float64,
        )


        self.goal_high = np.asarray(
            goal_high,
            dtype=np.float64,
        )


        # ====================================================
        # Safety
        # ====================================================

        self.workspace_low = np.asarray(
            workspace_low,
            dtype=np.float64,
        )


        self.workspace_high = np.asarray(
            workspace_high,
            dtype=np.float64,
        )


        self.safe_joint_low = np.asarray(
            safe_joint_low,
            dtype=np.float64,
        )


        self.safe_joint_high = np.asarray(
            safe_joint_high,
            dtype=np.float64,
        )


        self.base_offset = np.asarray(
            base_offset,
            dtype=np.float64,
        )


        self.max_orientation_deviation_deg = float(
            max_orientation_deviation_deg
        )


        self.max_orientation_deviation_rad = (
            np.deg2rad(
                self.max_orientation_deviation_deg
            )
        )


        # ====================================================
        # Gym spaces
        # ====================================================

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(7,),
            dtype=np.float32,
        )


        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(12,),
            dtype=np.float32,
        )


        # ====================================================
        # Episode state
        # ====================================================

        self.target_real = None

        self.prev_pos = None

        self.step_count = 0

        self.ep_return = 0.0

        self.minimum_distance = np.inf

        self.initial_distance = None


        # Orientation at reset
        self.initial_rotation = None


        self.controller_running = False

        self.ctrl = None


    # ========================================================
    # Coordinate conversion
    # ========================================================

    def real_to_sim(
        self,
        real_pos,
    ):

        return (
            np.asarray(
                real_pos,
                dtype=np.float64,
            )
            + self.base_offset
        )


    # ========================================================
    # Robot state
    # ========================================================

    def get_position(
        self,
    ):

        return np.asarray(
            self.panda.get_position(),
            dtype=np.float64,
        )


    def get_q(
        self,
    ):

        state = (
            self.panda.get_state()
        )


        q = np.asarray(
            state.q,
            dtype=np.float64,
        )


        return q[:7].copy()


    def get_ee_transform(
        self,
    ):
        """
        Read Franka O_T_EE and convert it into a 4x4 matrix.

        Franka stores the transform in column-major ordering.
        """

        state = (
            self.panda.get_state()
        )


        T = np.asarray(
            state.O_T_EE,
            dtype=np.float64,
        ).reshape(
            4,
            4,
            order="F",
        )


        return T


    def get_rotation(
        self,
    ):

        T = (
            self.get_ee_transform()
        )


        return T[
            :3,
            :3
        ].copy()


    # ========================================================
    # Orientation
    # ========================================================

    @staticmethod
    def rotation_error_rad(
        reference_rotation,
        current_rotation,
    ):
        """
        Compute geodesic rotation difference between R_ref and R.

        angle =
            acos(
                (trace(R_ref^T R) - 1) / 2
            )

        Returns radians in [0, pi].
        """

        R_ref = np.asarray(
            reference_rotation,
            dtype=np.float64,
        )


        R = np.asarray(
            current_rotation,
            dtype=np.float64,
        )


        R_error = (
            R_ref.T
            @ R
        )


        cos_angle = (
            np.trace(
                R_error
            )
            - 1.0
        ) / 2.0


        # Numerical protection for acos.
        cos_angle = np.clip(
            cos_angle,
            -1.0,
            1.0,
        )


        angle = np.arccos(
            cos_angle
        )


        return float(
            angle
        )


    def get_orientation_deviation(
        self,
    ):

        if self.initial_rotation is None:

            return 0.0


        current_rotation = (
            self.get_rotation()
        )


        return self.rotation_error_rad(
            self.initial_rotation,
            current_rotation,
        )


    # ========================================================
    # Observation
    # ========================================================

    def build_observation(
        self,
        current_pos_real,
        current_vel_real,
    ):

        current_sim = (
            self.real_to_sim(
                current_pos_real
            )
        )


        target_sim = (
            self.real_to_sim(
                self.target_real
            )
        )


        achieved_goal = (
            current_sim.astype(
                np.float32
            )
        )


        desired_goal = (
            target_sim.astype(
                np.float32
            )
        )


        observation = np.concatenate([
            current_sim,
            current_vel_real,
        ]).astype(
            np.float32
        )


        flat_obs = np.concatenate([
            achieved_goal,
            desired_goal,
            observation,
        ]).astype(
            np.float32
        )


        if flat_obs.shape != (12,):

            raise RuntimeError(
                f"Expected 12D observation, "
                f"got {flat_obs.shape}"
            )


        return flat_obs


    # ========================================================
    # Controller
    # ========================================================

    def _start_joint_controller(
        self,
    ):

        self.ctrl = (
            controllers.JointPosition(
                filter_coeff=1.0,
            )
        )


        self.panda.start_controller(
            self.ctrl
        )


        self.controller_running = True


    def _send_joint_target(
        self,
        q_target,
    ):

        q_target = np.asarray(
            q_target,
            dtype=np.float64,
        )


        if q_target.shape != (7,):

            raise ValueError(
                "Expected q_target shape (7,), "
                f"got {q_target.shape}"
            )


        if not self.controller_running:

            raise RuntimeError(
                "Joint controller is not running."
            )


        self.ctrl.set_control(
            q_target
        )


    def _stop_controller(
        self,
    ):

        if not self.controller_running:

            return


        try:

            self.panda.stop_controller()


        except Exception as e:

            print(
                f"[controller] stop warning: {e}"
            )


        finally:

            self.controller_running = False

            self.ctrl = None


    # ========================================================
    # Goal sampling
    # ========================================================

    def sample_goal(
        self,
        start_pos,
    ):

        offset = (
            self.np_random.uniform(
                low=self.goal_low,
                high=self.goal_high,
            )
        )


        target = (
            start_pos
            + offset
        )


        target = np.clip(
            target,
            self.workspace_low,
            self.workspace_high,
        )


        return target


    # ========================================================
    # Safety
    # ========================================================

    def _check_workspace(
        self,
        position,
    ):

        position = np.asarray(
            position,
            dtype=np.float64,
        )


        return bool(
            np.all(
                position
                >= self.workspace_low
            )
            and
            np.all(
                position
                <= self.workspace_high
            )
        )


    def _safe_joint_target(
        self,
        q_target,
    ):

        q_target = np.asarray(
            q_target,
            dtype=np.float64,
        )


        return np.clip(
            q_target,
            self.safe_joint_low,
            self.safe_joint_high,
        )


    # ========================================================
    # Reset
    # ========================================================

    def reset(
        self,
        *,
        seed=None,
        options=None,
    ):

        super().reset(
            seed=seed
        )


        self._stop_controller()


        # ====================================================
        # Return to known robot start
        # ====================================================

        try:

            self.panda.move_to_start()


        except Exception:

            print(
                "[reset] move_to_start failed "
                "-> recover()"
            )


            self.panda.recover()


            self.panda.move_to_start()


        time.sleep(
            0.5
        )


        # ====================================================
        # Start state
        # ====================================================

        start_pos = (
            self.get_position()
        )


        if not self._check_workspace(
            start_pos
        ):

            raise RuntimeError(
                "Start EE position outside workspace: "
                f"{start_pos}"
            )


        # ====================================================
        # Record orientation BEFORE policy starts moving
        # ====================================================

        self.initial_rotation = (
            self.get_rotation()
        )


        # ====================================================
        # Goal
        # ====================================================

        if (
            options is not None
            and "target_real" in options
        ):

            target = np.asarray(
                options[
                    "target_real"
                ],
                dtype=np.float64,
            )


            target = np.clip(
                target,
                self.workspace_low,
                self.workspace_high,
            )


            self.target_real = (
                target
            )


        elif (
            options is not None
            and "goal_offset" in options
        ):

            offset = np.asarray(
                options[
                    "goal_offset"
                ],
                dtype=np.float64,
            )


            target = (
                start_pos
                + offset
            )


            self.target_real = (
                np.clip(
                    target,
                    self.workspace_low,
                    self.workspace_high,
                )
            )


        else:

            self.target_real = (
                self.sample_goal(
                    start_pos
                )
            )


        # ====================================================
        # Episode state
        # ====================================================

        self.prev_pos = (
            start_pos.copy()
        )


        self.step_count = 0


        self.ep_return = 0.0


        self.initial_distance = float(
            np.linalg.norm(
                start_pos
                - self.target_real
            )
        )


        self.minimum_distance = (
            self.initial_distance
        )


        # ====================================================
        # Controller
        # ====================================================

        self._start_joint_controller()


        # ====================================================
        # Observation
        # ====================================================

        obs = self.build_observation(
            start_pos,
            np.zeros(
                3,
                dtype=np.float64,
            ),
        )


        info = {

            "distance":
                self.initial_distance,

            "distance_cm":
                self.initial_distance
                * 100.0,

            "is_success":
                float(
                    self.initial_distance
                    < self.goal_threshold
                ),

            "target_real":
                self.target_real.copy(),

            "initial_distance":
                self.initial_distance,

            "orientation_deviation_rad":
                0.0,

            "orientation_deviation_deg":
                0.0,

            "max_orientation_deviation_deg":
                self.max_orientation_deviation_deg,

            "safety_reason":
                None,
        }


        return (
            obs,
            info,
        )


    # ========================================================
    # Step
    # ========================================================

    def step(
        self,
        action,
    ):

        action = np.asarray(
            action,
            dtype=np.float64,
        ).flatten()


        if action.shape != (7,):

            raise ValueError(
                "Expected 7D action, "
                f"got shape {action.shape}"
            )


        # ====================================================
        # Same action clipping as panda-gym
        # ====================================================

        action = np.clip(
            action,
            self.action_space.low,
            self.action_space.high,
        )


        # ====================================================
        # Current joint state
        # ====================================================

        q_current = (
            self.get_q()
        )


        # ====================================================
        # panda-gym joint mapping
        # ====================================================

        delta_q = (
            action
            * self.joint_action_scale
        )


        q_target = (
            q_current
            + delta_q
        )


        # ====================================================
        # Joint safety
        # ====================================================

        unclipped_q_target = (
            q_target.copy()
        )


        q_target = (
            self._safe_joint_target(
                q_target
            )
        )


        joint_target_clipped = bool(
            not np.allclose(
                q_target,
                unclipped_q_target,
            )
        )


        truncated = False

        safety_reason = None


        # ====================================================
        # Send command
        # ====================================================

        try:

            self._send_joint_target(
                q_target
            )


            time.sleep(
                self.dt
            )


        except Exception as e:

            print(
                f"[step] controller error: {e}"
            )


            truncated = True


            safety_reason = (
                "controller_error"
            )


        # ====================================================
        # New robot state
        # ====================================================

        new_pos = (
            self.get_position()
        )


        # ====================================================
        # Workspace safety
        # ====================================================

        if not self._check_workspace(
            new_pos
        ):

            truncated = True


            safety_reason = (
                "workspace_violation"
            )


        # ====================================================
        # Orientation monitoring
        # ====================================================

        orientation_deviation_rad = (
            self.get_orientation_deviation()
        )


        orientation_deviation_deg = (
            np.rad2deg(
                orientation_deviation_rad
            )
        )


        if (
            orientation_deviation_rad
            > self.max_orientation_deviation_rad
        ):

            truncated = True


            safety_reason = (
                "orientation_violation"
            )


            print(
                "[SAFETY] orientation deviation "
                f"{orientation_deviation_deg:.2f} deg "
                f"> "
                f"{self.max_orientation_deviation_deg:.2f} deg"
            )


        # ====================================================
        # Velocity
        # ====================================================

        velocity = (
            new_pos
            - self.prev_pos
        ) * self.control_freq


        self.prev_pos = (
            new_pos.copy()
        )


        # ====================================================
        # Distance / reward
        # ====================================================

        distance = float(
            np.linalg.norm(
                new_pos
                - self.target_real
            )
        )


        reward = (
            -distance
        )


        self.minimum_distance = min(
            self.minimum_distance,
            distance,
        )


        self.ep_return += (
            reward
        )


        self.step_count += 1


        # ====================================================
        # Success
        # ====================================================

        terminated = bool(
            distance
            < self.goal_threshold
        )


        # ====================================================
        # Time limit
        # ====================================================

        if (
            self.step_count
            >= self.max_ep_steps
        ):

            truncated = True


            if safety_reason is None:

                safety_reason = (
                    "time_limit"
                )


        # ====================================================
        # Observation
        # ====================================================

        obs = self.build_observation(
            new_pos,
            velocity,
        )


        # ====================================================
        # Info
        # ====================================================

        info = {

            "distance":
                distance,

            "distance_cm":
                distance
                * 100.0,

            "minimum_distance":
                self.minimum_distance,

            "minimum_distance_cm":
                self.minimum_distance
                * 100.0,

            "is_success":
                float(
                    terminated
                ),

            "step_count":
                self.step_count,

            "episode_return":
                self.ep_return,

            "q_current":
                q_current.copy(),

            "q_target":
                q_target.copy(),

            "joint_target_clipped":
                joint_target_clipped,

            "action":
                action.astype(
                    np.float32
                ).copy(),

            "target_real":
                self.target_real.copy(),

            "orientation_deviation_rad":
                float(
                    orientation_deviation_rad
                ),

            "orientation_deviation_deg":
                float(
                    orientation_deviation_deg
                ),

            "max_orientation_deviation_deg":
                self.max_orientation_deviation_deg,

            "safety_reason":
                safety_reason,
        }


        return (
            obs,
            float(
                reward
            ),
            terminated,
            truncated,
            info,
        )


    # ========================================================
    # Close
    # ========================================================

    def close(
        self,
    ):

        self._stop_controller()


        try:

            self.panda.move_to_start()


        except Exception:

            try:

                self.panda.recover()


                self.panda.move_to_start()


            except Exception as e:

                print(
                    f"[close] warning: {e}"
                )
