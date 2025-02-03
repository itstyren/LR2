from typing import Any, Type
import warnings
from stable_baselines3.common.type_aliases import Schedule
from .policy import ActorCriticPolicy
from ..baseAlgorithm import BaseAlgorithm
from typing import Dict, Optional, Tuple, Type, Union
import numpy as np
import torch
from stable_baselines3.common.type_aliases import GymEnv, Schedule
from stable_baselines3.common.utils import get_schedule_fn
from gymnasium import spaces
from torch.nn import functional as F
from stable_baselines3.common.utils import get_linear_fn


class Strategy_PPO(BaseAlgorithm):
    """
    Proximal Policy Optimization algorithm (PPO) (clip version)
    """

    policy: ActorCriticPolicy

    def __init__(
        self,
        all_args,
        logger,
        env: Union[GymEnv, str],
        policy_class: Union[str, Type[ActorCriticPolicy]],
        policy_kwargs: Optional[Dict[str, Any]] = None,
        learning_rate: Union[float, Schedule] = 1e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        clip_range: Union[float, Schedule] = 0.2,
        clip_range_vf: Union[None, float, Schedule] = None,
        normalize_advantage: bool = True,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        device: Union[torch.device, str] = "auto",
        _init_setup_model: bool = True,
        ent_coef_fraction: float = 1,
        ent_coef_initial_eps: float = 1,
        ent_coef_final_eps: float = 0.01,        
    ) -> None:
        

        if env is not None:
            self.action_space = env.action_spaces["agent_0"]["d"]

        super().__init__(
            all_args,
            logger,
            env,
            policy_class,
            learning_rate,
            n_epochs,
            max_grad_norm=max_grad_norm,
            device=device,
            policy_kwargs=policy_kwargs,
            _init_setup_model=False,
        )

        self.clip_range = clip_range
        self.gamma = gamma
        self.normalize_advantage = normalize_advantage
        self.clip_range_vf=clip_range_vf

        self.vf_coef=vf_coef

        self.ent_coef_fraction=ent_coef_fraction
        self.ent_coef_initial_eps=ent_coef_initial_eps
        self.ent_coef_final_eps=ent_coef_final_eps

        self._sanity_check(n_steps,batch_size)

        if _init_setup_model:
            self._setup_model(obs_type="dilemma",action_type='d')

    def _setup_model(self,obs_type="dilemma",action_type='d') -> None:
        super()._setup_model(obs_type,action_type)

        # Initialize schedules for policy/value clipping
        self.clip_range = get_schedule_fn(self.clip_range)
        if self.clip_range_vf is not None:
            if isinstance(self.clip_range_vf, (float, int)):
                assert self.clip_range_vf > 0, "`clip_range_vf` must be positive, " "pass `None` to deactivate vf clipping"

            self.clip_range_vf = get_schedule_fn(self.clip_range_vf)
        self.ent_coef_schedule = get_linear_fn(
            self.ent_coef_initial_eps,
            self.ent_coef_final_eps,
            self.ent_coef_fraction,
        )


    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Get the dilemma action according to the policy for a given observation.
        """
        actions, values, log_prob,entropy= self.policy.predict(observation, deterministic=deterministic)
        return actions, values, log_prob,entropy

    def train(self, batch_size, rollout_buffer):
        """
        Update PPO policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        lr = self._update_learning_rate(self.policy.optimizer)
        n_epochs=self._update_n_epochs()
        # Update entropy coefficient
        ent_coef=self.ent_coef_schedule(self._current_progress_remaining)

        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []
        train_info = {}

        continue_training = True

        # train for n_epochs epochs
        for epoch in range(n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in rollout_buffer.sample(batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()
                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations, actions
                )
                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                # ratio between old and new policy, should be one at the first iteration
                ratio = torch.exp(log_prob - rollout_data.old_log_prob)
                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = torch.mean((torch.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the difference between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + torch.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -torch.mean(-log_prob)
                else:
                    entropy_loss = -torch.mean(entropy)
                    
                entropy_losses.append(entropy_loss.item())


                loss = policy_loss + ent_coef * entropy_loss + self.vf_coef * value_loss

                with torch.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = torch.mean((torch.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        
        train_info["dilemma_train/lr"] = lr
        train_info["dilemma_train/ent_coef"]= ent_coef
        train_info["dilemma_train/n_updates"] = self._n_updates
        train_info["dilemma_train/policy_gradient_loss"]=np.mean(pg_losses)
        train_info["dilemma_train/value_loss"]=np.mean(value_losses)
        train_info["dilemma_train/loss"]=loss.item()
        train_info["dilemma_train/approx_kl"]= np.mean(approx_kl_divs)
        train_info["dilemma_train/n_epochs"] = n_epochs

        return train_info

    def _sanity_check(self,n_steps,batch_size):
        '''
        Sanity check to avoid noisy gradient and NaN because of the advantage normalization
        '''
        if self.normalize_advantage:
            assert (
                batch_size > 1
            ), "`batch_size` must be greater than 1. See https://github.com/DLR-RM/stable-baselines3/issues/440" 
        
        if self.env is not None:
            # Check that `n_steps * n_envs > 1` to avoid NaN
            # when doing advantage normalization
            buffer_size = self.env.num_envs * n_steps
            assert buffer_size > 1 or (
                not self.normalize_advantage
            ), f"`n_steps * n_envs` must be greater than 1. Currently n_steps={self.n_steps} and n_envs={self.env.num_envs}"
            # Check that the rollout buffer size is a multiple of the mini-batch size
            untruncated_batches = buffer_size // batch_size
            if buffer_size % batch_size > 0:
                warnings.warn(
                    f"You have specified a mini-batch size of {batch_size},"
                    f" but because the `RolloutBuffer` is of size `n_steps * n_envs = {buffer_size}`,"
                    f" after every {untruncated_batches} untruncated mini-batches,"
                    f" there will be a truncated mini-batch of size {buffer_size % batch_size}\n"
                    f"We recommend using a `batch_size` that is a factor of `n_steps * n_envs`.\n"
                    f"Info: (n_steps={n_steps} and n_envs={self.env.num_envs})"
                )