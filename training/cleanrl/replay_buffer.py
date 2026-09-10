from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch


@dataclass
class ReplayBufferSamples:
    observations: torch.Tensor
    actions: torch.Tensor
    next_observations: torch.Tensor
    dones: torch.Tensor
    rewards: torch.Tensor


class ReplayBuffer:
    def __init__(
        self,
        buffer_size,
        observation_space,
        action_space,
        device,
        n_envs=1,
        handle_timeout_termination=False,
    ):
        assert isinstance(
            observation_space,
            gym.spaces.Box,
        ), "Only Box observation space is supported."

        assert isinstance(
            action_space,
            gym.spaces.Box,
        ), "Only Box action space is supported."

        self.buffer_size = int(buffer_size)
        self.device = torch.device(device)
        self.n_envs = int(n_envs)

        self.obs_shape = observation_space.shape
        self.action_dim = int(
            np.prod(action_space.shape)
        )

        self.observations = np.zeros(
            (
                self.buffer_size,
                self.n_envs,
                *self.obs_shape,
            ),
            dtype=np.float32,
        )

        self.next_observations = np.zeros(
            (
                self.buffer_size,
                self.n_envs,
                *self.obs_shape,
            ),
            dtype=np.float32,
        )

        self.actions = np.zeros(
            (
                self.buffer_size,
                self.n_envs,
                self.action_dim,
            ),
            dtype=np.float32,
        )

        self.rewards = np.zeros(
            (
                self.buffer_size,
                self.n_envs,
            ),
            dtype=np.float32,
        )

        self.dones = np.zeros(
            (
                self.buffer_size,
                self.n_envs,
            ),
            dtype=np.float32,
        )

        self.pos = 0
        self.full = False

    def add(
        self,
        obs,
        next_obs,
        action,
        reward,
        done,
        infos=None,
    ):
        self.observations[self.pos] = np.asarray(
            obs,
            dtype=np.float32,
        )

        self.next_observations[self.pos] = np.asarray(
            next_obs,
            dtype=np.float32,
        )

        self.actions[self.pos] = np.asarray(
            action,
            dtype=np.float32,
        ).reshape(
            self.n_envs,
            self.action_dim,
        )

        self.rewards[self.pos] = np.asarray(
            reward,
            dtype=np.float32,
        )

        self.dones[self.pos] = np.asarray(
            done,
            dtype=np.float32,
        )

        self.pos += 1

        if self.pos >= self.buffer_size:
            self.full = True
            self.pos = 0

    def sample(self, batch_size):
        upper_bound = (
            self.buffer_size
            if self.full
            else self.pos
        )

        if upper_bound == 0:
            raise RuntimeError(
                "Cannot sample from an empty replay buffer."
            )

        batch_indices = np.random.randint(
            0,
            upper_bound,
            size=batch_size,
        )

        env_indices = np.random.randint(
            0,
            self.n_envs,
            size=batch_size,
        )

        obs = self.observations[
            batch_indices,
            env_indices,
        ]

        next_obs = self.next_observations[
            batch_indices,
            env_indices,
        ]

        actions = self.actions[
            batch_indices,
            env_indices,
        ]

        rewards = self.rewards[
            batch_indices,
            env_indices,
        ].reshape(-1, 1)

        dones = self.dones[
            batch_indices,
            env_indices,
        ].reshape(-1, 1)

        return ReplayBufferSamples(
            observations=torch.as_tensor(
                obs,
                dtype=torch.float32,
                device=self.device,
            ),
            actions=torch.as_tensor(
                actions,
                dtype=torch.float32,
                device=self.device,
            ),
            next_observations=torch.as_tensor(
                next_obs,
                dtype=torch.float32,
                device=self.device,
            ),
            dones=torch.as_tensor(
                dones,
                dtype=torch.float32,
                device=self.device,
            ),
            rewards=torch.as_tensor(
                rewards,
                dtype=torch.float32,
                device=self.device,
            ),
        )