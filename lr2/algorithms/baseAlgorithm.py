from abc import ABC
from typing import List, Type, Union
from stable_baselines3.common.type_aliases import GymEnv, Schedule
from .basePolicy import BasePolicy
import torch
from stable_baselines3.common.utils import (
    get_schedule_fn,
    update_learning_rate,
)
from typing import Any, Dict, List, Optional, Type, Union


class BaseAlgorithm(ABC):
    """
    Base class for Reinforcement Learning algorithms.

    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    def __init__(
        self,
        all_args,
        logger,
        env: Union[GymEnv, str],
        policy_class: Union[str, Type[BasePolicy]],
        learning_rate: Union[float, Schedule],
        n_epochs: int,
        max_grad_norm: float,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        device=torch.device("cpu"),
        _init_setup_model: bool = True,
    ) -> None:
        self.all_args = all_args
        self.logger = logger
        self.policy_class = policy_class
        self.env = env

        self.max_grad_norm = max_grad_norm
        self.learning_rate = learning_rate

        self.n_epochs=n_epochs

        # Track the training progress remaining (from 1 to 0)
        # this is used to update the learning rate
        self._current_progress_remaining = 1.0        
        self.device = device

        # For logging (and TD3 delayed updates)
        self._n_updates = 0 

        self.policy_kwargs = {} if policy_kwargs is None else policy_kwargs

        if _init_setup_model:
            self._setup_model()


    def _setup_model(self,obs_type,action_type) -> None:
        """
        Set up the model
        """
        # setup the learning rate schedule
        self._setup_lr_schedule()
        self._setup_updates_schedule()

        optimizer_kwargs = {
            "weight_decay": self.all_args.weight_decay,
            "eps": self.all_args.opti_eps,
        }


        obs_space = self.env.observation_spaces["agent_0"][obs_type]
        act_space = self.env.action_spaces["agent_0"][action_type]
        
        self.policy = self.policy_class(
            self.all_args,
            obs_space,
            act_space,
            self.lr_schedule,
            optimizer_kwargs=optimizer_kwargs,
            **self.policy_kwargs
        )
        self.policy = self.policy.to(self.device)

    def _setup_lr_schedule(self) -> None:
        '''
        Transform to callable if needed.
        '''
        # return constant value of callable function based on input type of learning_rate
        self.lr_schedule = get_schedule_fn(self.learning_rate)

    def _setup_updates_schedule(self) -> None:
        '''
        Transform to callable if needed.
        '''
        # return constant value of callable function based on input type of learning_rate
        self.updates_schedule = get_schedule_fn(self.n_epochs)

    def _update_n_epochs(self) -> None:
        '''
        Update the number of updates
        '''
        n_epochs=int(self.updates_schedule(self._current_progress_remaining))
        return n_epochs
            

    def _update_learning_rate(
        self, optimizers: Union[List[torch.optim.Optimizer], torch.optim.Optimizer]
    ) -> float:
        """
        Update the optimizers learning rate using the current learning rate schedule
        and the current progress remaining (from 1 to 0).

        :param optimizers:
            An optimizer or a list of optimizers.
        """
        if not isinstance(optimizers, list):
            optimizers = [optimizers]

        lr =self.lr_schedule(self._current_progress_remaining)
        
        # update optimizer
        for optimizer in optimizers:
            update_learning_rate(
                optimizer, lr)
            
        return lr


    def _update_current_progress_remaining(
        self, num_timesteps: int, total_timesteps: int
    ) -> None:
        """
        Compute current progress remaining (starts from 1 and ends to 0)

        :param num_timesteps: current number of timesteps
        :param total_timesteps:
        """
        self._current_progress_remaining = 1.0 - float(num_timesteps) / float(
            total_timesteps
        )

    def train(self) -> None:
        """
        Consume current rollout data and update policy parameters.
        Implemented by individual algorithms.
        """
        raise NotImplementedError