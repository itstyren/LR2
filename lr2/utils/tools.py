import math
from typing import Callable
import torch    
from typing import Dict, Union
from gymnasium import spaces
from torch.nn import functional as F
import numpy as np
import os

def round_up(number: float, decimals: int = 2) -> float:
    """
    Round a number up to a specified number of decimal places.

    :param number: The number to be rounded.
    :param decimals: The number of decimal places to round to (default is 2).
    :return: The rounded number.
    """
    if not isinstance(decimals, int):
        raise TypeError("Decimal places must be an integer")
    if decimals < 0:
        raise ValueError("Decimal places must be 0 or more")

    if decimals == 0:
        return math.ceil(number)
    else:
        factor = 10**decimals
        return math.ceil(number * factor) / factor
    

def linear_schedule_to_0(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func


def linear_schedule_to_1_int(initial_value: int) -> Callable[[float], int]:
    """
    Linear schedule that decreases from initial_value to 1, returning integer values.

    :param initial_value: Initial value from which the schedule starts decreasing (must be > 1).
    :return: schedule that computes current value depending on remaining progress, rounded to integer.
    """

    def func(progress_remaining: float) -> int:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining: Fraction of progress remaining (1 at the start, 0 at the end).
        :return: current value decreasing linearly to 1, rounded to an integer.
        """
        return max(1, int(round(progress_remaining * (initial_value - 1) + 1)))

    return func


def linear_schedule_up(initial_value: int, final_value: int) -> Callable[[float], int]:
    """
    Linear schedule that increases from initial_value to final_value, returning integer values.

    :param initial_value: The starting value of the schedule.
    :param final_value: The target value to increase to.
    :return: A schedule that computes the current value depending on remaining progress, rounded to an integer.
    """

    def func(progress_remaining: float) -> int:
        """
        Progress will increase from 0 (beginning) to 1 (end).

        :param progress_remaining: Fraction of progress remaining (0 at the start, 1 at the end).
        :return: current value increasing linearly, rounded to an integer.
        """
        return int(round((1 - progress_remaining) * (final_value - initial_value) + initial_value))
    return func


def t2n(x):
    """
    Convert a tensor into a NumPy array.
    """
    return x.detach().cpu().numpy()

def collect_reputation_accessment(repu_all_envs):
    '''
    Collect elements from each corresponding env index across agent 
    Create new arrays store all agent for sing env

    :param repu_all_envs: list of reputation accessment for all agents in all envs [[agent][env]]
    '''
    # Create a repu accessment list for each environment for all agents
    repu_each_env_all = []

    # Loop over the index of the sub-arrays (since all arrays are assumed to have the same shape)
    for i in range(repu_all_envs[0].shape[0]):
        # Extract the i-th sub-array from each main array and stack them vertically
        stacked_sub_arrays = np.vstack([array[i] for array in repu_all_envs])
        repu_each_env_all.append(stacked_sub_arrays)
    
    return np.array(repu_each_env_all)


def preprocess_obs(
    obs: torch.Tensor,
    observation_space: spaces.Space,
) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
    if isinstance(observation_space, spaces.Box):
        return obs.float()
    if isinstance(observation_space, spaces.MultiDiscrete):
        # Tensor concatenation of one hot encodings of each Categorical sub-space
        return torch.cat(
            [
                F.one_hot(obs[_].long(), num_classes=dim).float()
                for _, dim in enumerate(observation_space.nvec)
            ],
            dim=-1,
        ).view(obs.shape[0], observation_space.nvec[0])
    elif isinstance(observation_space, spaces.Dict):
        # Do not modify by reference the original observation
        assert isinstance(obs, Dict), f"Expected dict, got {type(obs)}"
        preprocessed_obs = {}
        for key, _obs in obs.items():
            preprocessed_obs[key] = preprocess_obs(_obs, observation_space[key])
        return preprocessed_obs
    elif isinstance(observation_space, spaces.Discrete):
        # One hot encoding and convert to float to avoid errors
        return F.one_hot(obs.long(), num_classes=int(observation_space.n)).float()
    


def calculate_normalized_average_reputation(repu_all, dilemma_actions_all):
    
    def normalize_values(value1, value2):

        # Normalize both values to the range [0, 1]
        normalized_value1 = (value1 ) / (value1+ value2)
        normalized_value2 = (value2 ) / (value1+ value2)
        
        return normalized_value1, normalized_value2
    
    
    # Find all actions corresponding to 0 and 1
    action_0_mask = (dilemma_actions_all == 0)
    action_1_mask = (dilemma_actions_all == 1)
    
    # Calculate the total and the frequency of each action (0 or 1)
    total_action_0_repu = np.sum(repu_all[action_0_mask])
    total_action_1_repu = np.sum(repu_all[action_1_mask])
    
    count_action_0 = np.sum(action_0_mask)
    count_action_1 = np.sum(action_1_mask)
    
    # Calculate the frequency of each strategy (number of occurrences of action 0 and action 1)
    frequency_action_0 = count_action_0 / dilemma_actions_all.size
    frequency_action_1 = count_action_1 / dilemma_actions_all.size


    # Calculate the average reputation for action 0 and action 1
    avg_repu_action_0 = total_action_0_repu / count_action_0 if count_action_0 != 0 else 0
    avg_repu_action_1 = total_action_1_repu / count_action_1 if count_action_1 != 0 else 0


    avg_repu_action_0*=frequency_action_0
    avg_repu_action_1*=frequency_action_1

    # Normalize the average reputations relative to each other to the range [0, 1]
    normalized_avg_repu_action_0, normalized_avg_repu_action_1 = normalize_values(avg_repu_action_0, avg_repu_action_1)

    return normalized_avg_repu_action_0, normalized_avg_repu_action_1


def calculate_action_reward(reward_all,action_all):
    '''
    Calculate the reward for each action
    '''
    # Find all actions corresponding to 0 and 1
    coop_action_mask = (action_all == 0)
    defect_action_mask = (action_all == 1)
    
    # Calculate the total and the frequency of each action (0 or 1)
    total_coop_action_reward = np.sum(reward_all[coop_action_mask])
    total_defect_action_reward  = np.sum(reward_all[defect_action_mask])
    
    count_coop_action = np.sum(coop_action_mask)
    count_defect_action = np.sum(defect_action_mask)
    
    # Calculate the average reward for action 0 and action 1
    avg_reward_coop_action = total_coop_action_reward / count_coop_action if count_coop_action != 0 else 0
    avg_reward_defect_action = total_defect_action_reward / count_defect_action if count_defect_action != 0 else 0

    return [avg_reward_coop_action, avg_reward_defect_action]


def save_array(array, save_path, filename):
    # Convert the array to a NumPy array if it's not already
    if not isinstance(array, np.ndarray):
        array = np.array(array)

    # Construct the full file path
    full_path = os.path.join(save_path, filename)

    # Check if the file already exists
    if os.path.exists(full_path):
        # If the file exists, delete it
        os.remove(full_path)

    # Save the array using NumPy
    np.savez(full_path,array)
    print(f"Array saved successfully to {full_path}")