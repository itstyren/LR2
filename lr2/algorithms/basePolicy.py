import numpy as np
from torch import nn
import torch
from gymnasium import spaces
from typing import Any, Dict, Optional, Tuple, Type, Union
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    FlattenExtractor,
)
from abc import ABC, abstractmethod
import lr2.utils.tools as utils
from stable_baselines3.common.type_aliases import PyTorchObs
from stable_baselines3.common.utils import get_device
from torch.nn import functional as F

class BaseModel(nn.Module):
    """
    The base model object: makes predictions in response to observations.
    """

    optimizer: torch.optim.Optimizer

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        features_extractor: Optional[BaseFeaturesExtractor] = None,
        optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()

        if optimizer_kwargs is None:
            optimizer_kwargs = {}

        if features_extractor_kwargs is None:
            features_extractor_kwargs = {}

        self.observation_space = observation_space
        self.action_space = action_space
        self.features_extractor = features_extractor

        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = optimizer_kwargs

        self.features_extractor_class = features_extractor_class
        self.features_extractor_kwargs = features_extractor_kwargs

    def _update_features_extractor(
        self,
        net_kwargs: Dict[str, Any],
        features_extractor: Optional[BaseFeaturesExtractor] = None,
    ) -> Dict[str, Any]:
        """
        Update the network keyword arguments and create a new features extractor object if needed.
        If a ``features_extractor`` object is passed, then it will be shared.

        :param net_kwargs: the base network keyword arguments, without the ones
            related to features extractor
        :param features_extractor: a features extractor object.
            If None, a new object will be created.
        :return: The updated keyword arguments
        """
        net_kwargs = net_kwargs.copy()
        if features_extractor is None:
            # The features extractor is not shared, create a new one
            features_extractor = self.make_features_extractor()
        net_kwargs.update(
            dict(
                features_extractor=features_extractor,
                features_dim=features_extractor.features_dim,
            )
        )
        return net_kwargs

    def make_features_extractor(self) -> BaseFeaturesExtractor:
        """
        Helper method to create a features extractor.
        """
        return self.features_extractor_class(
            self.observation_space, **self.features_extractor_kwargs
        )

    def set_training_mode(self, mode: bool) -> None:
        """
        Put the policy in either training or evaluation mode.
        """
        self.train(mode)

    def obs_to_tensor(
        self, observation: Union[np.ndarray, Dict[str, np.ndarray]]
    ) -> Tuple[torch.Tensor, bool]:
        """
        Convert an input observation to a PyTorch tensor that can be fed to a model.
        """
        for _, obs in enumerate(observation):
            observation[_] = {
                key: torch.as_tensor(_obs, device=self.device)
                for (key, _obs) in obs.items()
            }
        return observation

    def extract_features(
        self, obs: torch.Tensor, features_extractor: BaseFeaturesExtractor
    ) -> torch.Tensor:
        """
        Preprocess the observation if needed and extract features.

         :param obs: The observation
         :param features_extractor: The features extractor to use.
         :return: The extracted features
        """
        obs_seq = []
        for obs_ in obs:
            preprocessed_obs = utils.preprocess_obs(
                obs_, self.observation_space
            )
            obs_seq.append(torch.cat([preprocessed_obs[key].flatten() for key in preprocessed_obs], dim=0))
        obs_seq = torch.stack(obs_seq)
        return obs_seq

    @property
    def device(self) -> torch.device:
        """Infer which device this policy lives on by inspecting its parameters.
        If it has no parameters, the 'cpu' device is used as a fallback.

        :return:"""
        for param in self.parameters():
            return param.device
        return get_device("cpu")



class BasePolicy(BaseModel, ABC):
    """The base policy object.

    Parameters are mostly the same as `BaseModel`; additions are documented below.

    """

    features_extractor: BaseFeaturesExtractor

    def __init__(self, *args, squash_output: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self._squash_output = squash_output


    @property
    def squash_output(self) -> bool:
        """(bool) Getter for squash_output."""
        return self._squash_output

    @abstractmethod
    def _predict(self, observation: PyTorchObs, deterministic: bool = False) -> torch.Tensor:
        """
        Get the action according to the policy for a given observation.
        """

    @staticmethod
    def init_weights(module: nn.Module, gain: float = 1) -> None:
        """
        Orthogonal initialization (used in PPO and A2C)
        """
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.orthogonal_(module.weight, gain=gain)
            if module.bias is not None:
                module.bias.data.fill_(0.0)

    def predict(
        self,
        obs_seq: Union[np.ndarray, Dict[str, np.ndarray]],
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        trun obs to tensor and predict the action using policy
        """
        self.set_training_mode(False)

        # Check for common mistake that the user does not mix Gym/VecEnv API
        # Tuple obs are not supported by SB3, so we can safely do that check
        if isinstance(obs_seq, tuple) and len(obs_seq) == 2 and isinstance(obs_seq[1], dict):
            raise ValueError(
                "You have passed a tuple to the predict() function instead of a Numpy array or a Dict. "
                "You are probably mixing Gym API with SB3 VecEnv API: `obs, info = env.reset()` (Gym) "
                "vs `obs = vec_env.reset()` (SB3 VecEnv). "
                "See related issue https://github.com/DLR-RM/stable-baselines3/issues/1694 "
                "and documentation for more information: https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html#vecenv-api-vs-gym-api"
            )
        with torch.no_grad():
            obs_seq = self.obs_to_tensor(obs_seq)
            return self._predict(obs_seq,deterministic=deterministic)

