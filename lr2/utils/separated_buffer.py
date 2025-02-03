from abc import ABC
from gymnasium import spaces
from typing import Dict, Tuple, Union
import torch
from stable_baselines3.common.preprocessing import get_obs_shape, get_action_dim
from typing import NamedTuple
import numpy as np
from stable_baselines3.common.utils import get_device
from abc import ABC, abstractmethod
from typing import  Dict, Generator, Optional, Tuple, Union


class ReplayBufferSamples(NamedTuple):
    observations: torch.Tensor
    actions: torch.Tensor
    next_observations: torch.Tensor
    dones: torch.Tensor
    rewards: torch.Tensor


class RepuRolloutBufferSamples(NamedTuple):
    observations: torch.Tensor
    next_observations: torch.Tensor
    actions: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    rewards_list: torch.Tensor
    old_log_prob: torch.Tensor


class DilemmaRolloutBufferSamples(NamedTuple):
    observations: torch.Tensor
    actions: torch.Tensor
    old_values: torch.Tensor
    old_log_prob: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor


class BaseBuffer(ABC):
    """
    Base class that represent a buffer (rollout or replay)
    """

    observation_space: spaces.Space
    obs_shape: Tuple[int, ...]

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[torch.device, str] = "auto",
        n_envs: int = 1,
        episode_length: int = 1,
    ):
        super().__init__()
        self.buffer_size = buffer_size
        self.observation_space = observation_space
        self.action_space = action_space
        self.obs_shape = get_obs_shape(observation_space)
        self.action_dim = get_action_dim(action_space)
        self.n_envs = n_envs
        self.episode_length = episode_length

        self.device = get_device(device)
        self.step = 0
        self.full = False

    def insert(self, repu_obs, next_obs, reward, termination, truncation, action):
        """
        Insert a new transition in the buffer
        """
        raise NotImplementedError()

    def sample(self, batch_size: int):
        """
        :param batch_size: Number of element to sample
        """
        upper_bound = self.buffer_size if self.full else self.step
        batch_inds = np.random.randint(0, upper_bound, size=batch_size)
        return self._get_samples(batch_inds)

    @abstractmethod
    def _get_samples(self, batch_inds: np.ndarray) -> Union[ReplayBufferSamples]:
        """
        :param batch_inds:
        """
        raise NotImplementedError()

    @staticmethod
    def swap_and_flatten(arr: np.ndarray) -> np.ndarray:
        """
        Swap and then flatten axes 0 (buffer_size) and 1 (n_envs)
        """
        shape = arr.shape
        return arr.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])

    def reset(self) -> None:
        """
        Reset the buffer.
        """
        self.step = 0
        self.full = False

    def to_torch(
        self, array: Union[np.ndarray, Dict[str, np.ndarray]], copy: bool = True
    ) -> torch.Tensor:
        """
        Convert a numpy array to a PyTorch tensor.
        """
        # deal with obvservation
        if isinstance(array[0], dict):
            for _, obs in enumerate(array):
                array[_] = {
                    key: torch.as_tensor(_obs, device=self.device)
                    for (key, _obs) in obs.items()
                }
            return array
        elif isinstance(array[0],np.ndarray) and isinstance(array[0][0], dict):
            for obs_i, obs in enumerate(array):
                for sub_obs_i,sub_obs in enumerate(obs):
                    array[obs_i][sub_obs_i] = {
                    key: torch.as_tensor(_obs, device=self.device)
                    for (key, _obs) in sub_obs.items()
                }
            return array
        else:
            return torch.tensor(array, device=self.device)

class DilemmaRolloutBuffer(BaseBuffer):
    """
    Rollout buffer used in on-policy algorithms for each agent
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: torch.device | str = "auto",
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
    ):
        super().__init__(
            buffer_size, observation_space, action_space, device, n_envs=n_envs
        )
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.generator_ready = False
        self.reset()

    def reset(self) -> None:
        self.observations = np.array(
            [
                [
                    {
                        key: np.zeros(shape, dtype=np.float32)
                        for key, shape in self.obs_shape.items()
                    }
                    for _ in range(self.n_envs)
                ]
                for _ in range(self.buffer_size)
            ]
        )
        self.actions = np.zeros(
            (self.buffer_size, self.n_envs, 1),
            dtype=np.int32,
        )
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.returns = np.zeros(
            (self.buffer_size, self.n_envs), dtype=np.float32
        )  # the lambda-return (TD(lambda) estimate)
        self.episode_starts = np.zeros(
            (self.buffer_size, self.n_envs), dtype=np.float32
        )
        self.values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.log_probs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.advantages = np.zeros(
            (self.buffer_size, self.n_envs), dtype=np.float32
        )  # GAE(lambda) advantage
        self.generator_ready = False
        super().reset()

    def insert(
        self,
        obs: np.ndarray[Dict],
        action: np.ndarray,
        reward: np.ndarray,
        episode_start: np.ndarray,
        value: torch.Tensor,
        log_prob: torch.Tensor,
    ):
        """
        Insert a new transition in the buffer
        """
        self.observations[self.step] = np.array(obs)
        self.rewards[self.step] += np.array(reward)
        self.actions[self.step] = np.array(action).reshape(-1, 1)
        self.episode_starts[self.step] = np.array(episode_start)
        if isinstance(value, torch.Tensor):
            value = value.clone().cpu().numpy().flatten()
            log_prob = log_prob.clone().cpu().numpy().flatten()
        self.values[self.step] = np.array(value)
        self.log_probs[self.step] = np.array(log_prob)

        self.step += 1
        if self.step == self.buffer_size:
            self.full = True

    def sample(
        self, batch_size: Optional[int] = None
    ) -> Generator[DilemmaRolloutBufferSamples, None, None]:
        """
        get all the samples in the buffer
        """
        assert self.full, ""
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        # Prepare the data
        if not self.generator_ready:
            _tensor_names = [
                "observations",
                "actions",
                "values",
                "log_probs",
                "advantages",
                "returns",
            ]
            for tensor in _tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(self, batch_inds: np.ndarray) -> DilemmaRolloutBufferSamples:
        data = (
            self.observations[batch_inds],
            self.actions[batch_inds],
            self.values[batch_inds].flatten(),
            self.log_probs[batch_inds].flatten(),
            self.advantages[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
        )
        return DilemmaRolloutBufferSamples(*tuple(map(self.to_torch, data)))

    def compute_returns_and_advantage(
        self, last_values: np.ndarray, dones: np.ndarray
    ) -> None:
        """
        Post-processing step: compute the lambda-return (TD(lambda) estimate)
        and GAE(lambda) advantage.
        """
        last_gae_lam = 0
        # iterate over each sub env
        for step in reversed(range(self.buffer_size)):
            # if this is last rollout step
            if step == self.buffer_size - 1:
                # If we are in the last step of episode, we don't need to compute using the next value
                # 1 - not terminal state
                next_non_terminal = 1.0 - dones.astype(np.float32)
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_values = self.values[step + 1]
            delta = (
                self.rewards[step]
                + self.gamma * next_values * next_non_terminal
                - self.values[step]
            )
            last_gae_lam = (
                delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            )
            self.advantages[step] = last_gae_lam
        # TD(lambda) estimator
        self.returns = self.advantages + self.values


class RepuRolloutBuffer(BaseBuffer):
    """
    Rollout buffer used in on-policy algorithms for each agent
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: torch.device | str = "auto",
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
        neighbour_count: int = 4,
    ):
        super().__init__(
            buffer_size, observation_space, action_space, device, n_envs=n_envs
        )
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.generator_ready = False
        self.neighbour_count = neighbour_count
        self.reset()

    def reset(self) -> None:
        self.observations = np.array(
            [
                [
                    {
                        key: np.zeros(shape, dtype=np.float32)
                        for key, shape in self.obs_shape.items()
                    }
                    for _ in range(self.n_envs)
                ]
                for _ in range(self.buffer_size)
            ]
        )
        self.next_observations = np.array(
            [
                [
                    {
                        key: np.zeros(shape, dtype=np.float32)
                        for key, shape in self.obs_shape.items()
                    }
                    for _ in range(self.n_envs)
                ]
                for _ in range(self.buffer_size)
            ]
        )
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.returns = np.zeros((self.buffer_size, self.n_envs,self.neighbour_count), dtype=np.float32)
        self.values = np.zeros((self.buffer_size, self.n_envs,self.neighbour_count), dtype=np.float32)
        self.log_prob = np.zeros((self.buffer_size, self.n_envs,self.neighbour_count), dtype=np.float32)
        self.advantages = np.zeros((self.buffer_size, self.n_envs,self.neighbour_count), dtype=np.float32)
        self.episode_starts = np.zeros(
            (self.buffer_size, self.n_envs), dtype=np.float32
        )
        self.rewards_list = np.zeros(
            (self.buffer_size, self.n_envs, self.neighbour_count), dtype=np.float32
        )
        self.actions = np.zeros(
            (self.buffer_size, self.n_envs, self.neighbour_count),
            dtype=np.float32,
        )
        
        self.generator_ready = False
        super().reset()

    def insert(
        self,
        obs: np.ndarray[Dict],
        next_obs: np.ndarray[Dict],
        action: np.ndarray,
        episode_start: np.ndarray,
        value: np.ndarray,
        log_prob: np.ndarray,
    ):
        """
        Insert a new transition in the buffer
        """
        self.observations[self.step] = obs.copy()
        self.next_observations[self.step] = next_obs.copy()
        if action is not None:
            self.actions[self.step] = action.copy()

        self.episode_starts[self.step] = np.array(episode_start)

        if isinstance(value, torch.Tensor):
            value = value.clone().cpu().numpy().flatten()
        self.values[self.step] = np.array(value)
        self.log_prob[self.step] = np.array(log_prob)

        self.step += 1
        if self.step == self.buffer_size:
            self.full = True

    def sample(
        self, batch_size: Optional[int] = None
    ) -> Generator[RepuRolloutBufferSamples, None, None]:
        """
        get all the samples in the buffer
        """
        assert self.full, ""
        # indices = np.random.permutation(self.buffer_size * self.n_envs)
        indices = range(self.buffer_size * self.n_envs)
        # Prepare the data
        if not self.generator_ready:
            _tensor_names = [
                "observations",
                "next_observations",
                "actions",
                "advantages",
                "returns",
                "rewards_list",
                'log_prob'
            ]
            for tensor in _tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs  # actual train freq in hyperparams
        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(self, batch_inds: np.ndarray) -> ReplayBufferSamples:
        data = (
            self.observations[batch_inds],
            self.next_observations[batch_inds],
            self.actions[batch_inds],
            self.advantages[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
            self.rewards_list[batch_inds],
            self.log_prob[batch_inds].flatten()
        )
        return RepuRolloutBufferSamples(*tuple(map(self.to_torch, data)))

    def compute_returns_and_advantage(
        self, last_values: np.ndarray, dones: np.ndarray
    ) -> None:
        """
        Post-processing step: compute the lambda-return (TD(lambda) estimate)
        and GAE(lambda) advantage.
        """
        # self.rewards = self.rewards_list.sum(axis=2)
        last_gae_lam = 0
        # iterate over each sub env
        for step in reversed(range(self.buffer_size)):
            # if this is last rollout step
            if step == self.buffer_size - 1:
                # If we are in the last step of episode, we don't need to compute using the next value
                # 1 - not terminal state
                next_non_terminal = 1.0 - dones.astype(np.float32)
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_values = self.values[step + 1]
            delta = (
                self.rewards_list[step]
                + self.gamma * next_values * next_non_terminal.reshape(-1, 1)
                - self.values[step]
            )
            last_gae_lam = (
                delta + self.gamma * self.gae_lambda * next_non_terminal.reshape(-1, 1) * last_gae_lam
            )
            self.advantages[step] = last_gae_lam
        # TD(lambda) estimator
        self.returns = self.advantages + self.values


