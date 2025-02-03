import os
import sys
import time
import wandb
from tensorboardX import SummaryWriter
import lr2.utils.tools as utils
from stable_baselines3.common.type_aliases import (
    TrainFreq,
    TrainFrequencyUnit,
)
import torch
import numpy as np
from typing import Optional
import re

class Runner(object):
    """
    Base Runner calss for RL training
    """

    def __init__(self, config):
        # pass the configuration
        self.all_args = config["all_args"]
        self.envs = config["envs"]
        self.device = config["device"]
        self.num_agents = config["num_agents"]

        # setting from cli arguments
        self.num_env_steps = self.all_args.num_env_steps  # the total train steps
        self.episode_length = self.all_args.episode_length
        self.use_wandb = self.all_args.use_wandb
        self.n_rollout_threads = self.all_args.n_rollout_threads
        self.algorithm_name = self.all_args.algorithm_name
        self.experiment_name = self.all_args.experiment_name
        self.env_name = self.all_args.env_name
        self.gamma = self.all_args.gamma
        self.repu_reward_coef = self.all_args.repu_reward_coef

        # Alternate training of upper and lower networks
        self.train_upper = False
        self.repu_loss = np.zeros(self.num_agents)
        self.repu_gradient_loss = np.zeros(self.num_agents)
        self.repu_value_loss = np.zeros(self.num_agents)
        self.repu_approx_kl = np.zeros(self.num_agents)

        self.network_type=self.all_args.network_type

        # Define search_key based on network_type
        if self.network_type == "lattice":
            self.neighbour_count = 4  # Number of neighbors for Lattice
        elif self.network_type == "honeycomb":
            self.neighbour_count = 3  # Number of neighbors for Honeycomb
        elif self.network_type == "moore":
            self.neighbour_count = 8  # Number of neighbors for Moore
        elif self.network_type == "mixed":
            self.neighbour_count = self.all_args.mixed_num  # Number of neighbors for Mixed
        else:
            raise ValueError(f"Unsupported network type: {self.network_type}")

        self.repu_ti = {
            "repu_train/lr": 0,
            "repu_train/n_updates": 0,
            "repu_train/ent_coef": 0,
            "repu_train/n_epochs": self.all_args.repu_n_epochs,
        }
        self.dilemma_ti = {
            "dilemma_train/lr": 0,
            "dilemma_train/ent_coef": 0,
            "dilemma_train/n_updates": 0,
            "dilemma_train/n_epochs": 1,
        }
        self.dilemma_gradient_loss = np.zeros(self.num_agents)
        self.dilemma_value_loss = np.zeros(self.num_agents)
        self.dilemma_loss = np.zeros(self.num_agents)
        self.dilemma_approx_kl = np.zeros(self.num_agents)

        self._last_repu_obs = None
        self._last_dilemma_obs = None
        self._last_episode_starts = None  # type: Optional[np.ndarray]

        # training setting
        # Save train freq parameter, will be converted later to TrainFreq object
        train_freq = utils.round_up(
            self.all_args.train_freq / self.n_rollout_threads, 0
        )
        self.train_freq = (train_freq, self.all_args.freq_type)
        # Convert train freq parameter to TrainFreq object
        self._convert_train_freq()

        # if using pre-defined norm, then skip the reputaion training
        norm_match = re.match(r'PPO_(\w+)', self.all_args.algorithm_name)
        self.skip_repu_train = True if (norm_match or self.all_args.disable_repu) else False

        # if decay learning rate for optimizer
        if self.all_args.use_linear_lr_decay:
            repu_lr = utils.linear_schedule_to_0(self.all_args.repu_lr)
            dilemma_lr = utils.linear_schedule_to_0(self.all_args.dilemma_lr)
        else:
            repu_lr = self.all_args.repu_lr
            dilemma_lr = self.all_args.dilemma_lr
        if self.all_args.use_linear_update:
            repu_n_epochs = utils.linear_schedule_to_1_int(self.all_args.repu_n_epochs)
            dilemma_n_epochs = utils.linear_schedule_up(
                1, self.all_args.dilemma_n_epochs
            )
        else:
            repu_n_epochs = self.all_args.repu_n_epochs
            dilemma_n_epochs = self.all_args.dilemma_n_epochs

        # set interval for logging
        self.log_interval = self.all_args.log_interval
        self.video_interval = self.all_args.video_interval

        # dir
        self.model_dir = self.all_args.model_dir

        # report environment configuration
        self.env_report()

        # set up the policy and strategy
        from lr2.algorithms.ppo.policy import ActorCriticPolicy as DilemmaPolicy


        from lr2.algorithms.a2c.a2c_trainer import Strategy_PPO as EvalTrainAlgo
        from lr2.algorithms.ppo.ppo_trainer import Strategy_PPO as DilemmaTrainAlgo
        from lr2.utils.separated_buffer import RepuRolloutBuffer, DilemmaRolloutBuffer

        self._setup_logging(config)

        self.repu_trainer = []
        self.dilemma_trainer = []
        self.dilemma_buffer = []
        self.repu_buffer = []

        dilemma_policy_kwargs = dict(
            net_arch=self.all_args.dilemma_net_arch,
        )
        repu_policy_kwargs = dict(net_arch=self.all_args.repu_net_arch)

        # assign the reputaion evalutaion trainer for each agent
        for _ in range(self.num_agents):
            eval_tr = EvalTrainAlgo(
                all_args=self.all_args,
                logger=self.logger,
                env=self.envs,
                policy_class=DilemmaPolicy,
                policy_kwargs=repu_policy_kwargs,
                learning_rate=repu_lr,
                n_steps=self.train_freq.frequency,
                batch_size=self.all_args.mini_batch,
                n_epochs=repu_n_epochs,
                gamma=self.gamma,
                device=self.device,
                ent_coef_fraction=self.all_args.repu_ent_coef_fraction,
                ent_coef_initial_eps=self.all_args.repu_ent_coef_initial_eps,
                ent_coef_final_eps=self.all_args.repu_ent_coef_final_eps,
            )
            self.repu_trainer.append(eval_tr)

        for _ in range(self.num_agents):
            dilemma_tr = DilemmaTrainAlgo(
                all_args=self.all_args,
                logger=self.logger,
                env=self.envs,
                policy_class=DilemmaPolicy,
                policy_kwargs=dilemma_policy_kwargs,
                learning_rate=dilemma_lr,
                n_steps=self.train_freq.frequency,
                batch_size=self.all_args.mini_batch,
                n_epochs=dilemma_n_epochs,
                gamma=self.gamma,
                normalize_advantage=self.all_args.normalize_advantage,
                device=self.device,
                ent_coef_fraction=self.all_args.dilemma_ent_coef_fraction,
                ent_coef_initial_eps=self.all_args.dilemma_ent_coef_initial_eps,
                ent_coef_final_eps=self.all_args.dilemma_ent_coef_final_eps,
            )
            self.dilemma_trainer.append(dilemma_tr)

        # Restore a pre-trained model if the model directory is specified
        have_load_buffer = False
        if self.model_dir is not None:
            print("Loading model from: ", self.model_dir)

        # crete replay dilemma_buffer for each agent
        if not have_load_buffer:
            for agent_idx in range(self.num_agents):
                d_b = DilemmaRolloutBuffer(
                    self.train_freq.frequency,  # the actual rollout length for each env
                    observation_space=self.envs.observation_spaces[
                        "agent_{}".format(agent_idx)
                    ]["dilemma"],
                    action_space=self.envs.action_spaces["agent_{}".format(agent_idx)][
                        "d"
                    ],
                    n_envs=self.n_rollout_threads,
                    gae_lambda=self.all_args.gae_lambda,
                    gamma=self.gamma,
                    device=self.device,
                )
                r_b = RepuRolloutBuffer(
                    self.train_freq.frequency,
                    observation_space=self.envs.observation_spaces[
                        "agent_{}".format(agent_idx)
                    ]["repu"],
                    action_space=self.envs.action_spaces["agent_{}".format(agent_idx)][
                        "r_a"
                    ],
                    n_envs=self.n_rollout_threads,
                    device=self.device,
                    neighbour_count=self.neighbour_count,
                )

                self.dilemma_buffer.append(d_b)
                self.repu_buffer.append(r_b)

        print("\nStrat Training...\n")

    def _setup_logging(self, config):
        """
        Setup logging for the training process
        """
        if self.use_wandb:
            self.save_dir = str(wandb.run.dir)
            self.run_dir = str(wandb.run.dir)
            self.logger = None
            self.gif_dir = str(self.run_dir + "/gifs")
            self.plot_dir = str(self.run_dir + "/plots")
        else:
            #  Configure directories for logging and saving models manually
            self.run_dir = config["run_dir"]
            self.log_dir = str(self.run_dir / "logs")
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)
            self.logger = SummaryWriter(self.log_dir)
            self.save_dir = str(self.run_dir / "models")
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
            self.gif_dir = str(self.run_dir / "gifs")
            self.plot_dir = str(self.run_dir / "plots")

        if not os.path.exists(self.gif_dir):
            os.makedirs(self.gif_dir)

        if not os.path.exists(self.plot_dir):
            os.makedirs(self.plot_dir)

    def env_report(self):
        """
        Print the observation and action space of the environment
        """
        print("==== Environment Configuration ====")
        print(
            "Reputaion observation_space: ",
            self.envs.observation_spaces["agent_0"]["repu"],
        )
        print("Reputation action_space: ", self.envs.action_spaces["agent_0"]["r_a"])
        print(
            "Dilemma observation_space: ",
            self.envs.observation_spaces["agent_0"]["dilemma"],
        )
        print("Dilemma action_space: ", self.envs.action_spaces["agent_0"]["d"])

    def _convert_train_freq(self) -> None:
        """
        Convert `train_freq` parameter (int or tuple)
        to a TrainFreq object.
        """
        if not isinstance(self.train_freq, TrainFreq):
            train_freq = self.train_freq

            # The value of the train frequency will be checked later
            if not isinstance(train_freq, tuple):
                train_freq = (train_freq, "step")

            try:
                train_freq = (train_freq[0], TrainFrequencyUnit(train_freq[1]))
            except ValueError as e:
                raise ValueError(
                    f"The unit of the `train_freq` must be either 'step' or 'episode' not '{train_freq[1]}'!"
                ) from e

            if not isinstance(train_freq[0], int):
                raise ValueError(
                    f"The frequency of `train_freq` must be an integer and not {train_freq[0]}"
                )
            self.train_freq = TrainFreq(*train_freq)

    def collect_rollouts(self, _last_rollout=False):
        """
        Collect experiences and store them into a ``ReplayBuffer``.

        :param end_of_episode: (bool) whether the current episode is finished
        """
        # Sample both reputaion assessment and dilemma action from the policy
        dilemma_actions_all, dilamma_values_all, log_probs_all, entropy_all = (
            self.get_actions()
        )
        dilemma_next_obs, rews, sub_rews = self.envs.action_step(dilemma_actions_all)


        # rews+=0.5*entropy_all
        # get the sum reward for each agent in each env
        # sum the reward between agent with every neighbour
        # rews_sum = np.sum(rews, axis=2)
        # breakpoint()
        # sub_rews= (sub_rews - sub_rews.mean()) / (sub_rews.std() + 1e-8)
        for i, sr_thread in enumerate(sub_rews):
            for j, sr_agent in enumerate(sr_thread):
                sub_rews[i][j]=(sub_rews[i][j]-sr_thread.mean())/(sr_thread.std()+1e-8)
                # breakpoint()
                # sub_rews[i][j] = (sub_rews[i][j] - sub_rews[i][j].mean()) / (
                #     sub_rews[i][j].std() + 1e-8
                # )

        for agent_idx in range(self.num_agents):
            step = self.repu_buffer[agent_idx].step
            # step = self.repu_buffer[agent_idx].step - 1
            # for repu, need to store the reward between agent with each neighbour
            self.repu_buffer[agent_idx].rewards_list[step] = sub_rews[:, agent_idx]
            self.repu_buffer[agent_idx].rewards[step] = rews[:, agent_idx]

        # Store dilemma action data in replay buffer
        self.log_probs_all = log_probs_all.copy()

        # store the old dilemma obs for using in the buffer store
        self._old_dilemma_obs = self._last_dilemma_obs.copy()

        self._last_repu_obs = dilemma_next_obs.copy()

        assert self._last_repu_obs is not None, "self._last_repu_obs was not set"
        (
            repu_access_each_env_all,
            repu_action_all,
            repu_values_all,
            repu_log_value_all,
        ) = self.get_repu_action()

        # new dilemma obs will be assigned here
        # the repu reward also added to the dilemma reward herer
        (
            repu_next_obs,
            repu_reward,
            repu_all,
            average_repu_stra,
            repu_std,
            terminations,
            truncations,
            infos, # info in the last timestep in one episode
        ) = self.update_env_buffer_repu(
            repu_access_each_env_all,
            repu_action_all,
            repu_values_all,
            repu_log_value_all,
            update_env=True,
        )

        action_reward=utils.calculate_action_reward(repu_reward.flatten(), dilemma_actions_all.flatten())

        normalized_avg_coop_repu, normalized_avg_defect_repu = (
            utils.calculate_normalized_average_reputation(
                np.array(repu_all), np.array(dilemma_actions_all)
            )
        )

        self._store_dilemma_to_buffer(
            dilemma_actions_all,
            repu_reward,
            dilamma_values_all,
            log_probs_all,
            truncations,
            infos,
        )
        # if this is the end of one rollout
        if _last_rollout:
            last_dilemma_values = []
            last_repu_values = []


            dilemma_last_obs = self.process_repu_obs_to_dilemma_obs(repu_next_obs)
            dilemma_last_actions, _, _, _ = self.get_actions(init_obs=dilemma_last_obs)
            repu_last_obs, _, _ = self.envs.action_step(
                dilemma_last_actions, move=False
            )
            repu_last_obs = self.process_single_repu_obs(repu_last_obs)

            for agent_idx in range(self.num_agents):
                dilemma_obs = self.dilemma_trainer[agent_idx].policy.obs_to_tensor(
                    dilemma_last_obs[:, agent_idx]
                )

                with torch.no_grad():
                    if "PPO" in self.repu_trainer[0].__class__.__name__:
                        r_values = []
                        obses = np.array(
                            [
                                item
                                for subarray in repu_last_obs[:, agent_idx]
                                for item in subarray
                            ]
                        )
                        repu_obs = self.repu_trainer[agent_idx].policy.obs_to_tensor(
                            obses
                        )
                        r_values = self.repu_trainer[agent_idx].policy.predict_values(
                            repu_obs
                        )
                        r_values = (
                            r_values.clone().cpu().numpy().flatten().reshape(-1, self.neighbour_count)
                        )
                        last_repu_values.append(r_values)

                    dilemma_values = self.dilemma_trainer[
                        agent_idx
                    ].policy.predict_values(dilemma_obs)
                    last_dilemma_values.append(
                        dilemma_values.clone().cpu().numpy().flatten()
                    )

            # must compute returns and advantage after insert all data
            for agent_idx in range(self.num_agents):
                if "PPO" in self.repu_trainer[0].__class__.__name__:
                    self.repu_buffer[agent_idx].compute_returns_and_advantage(
                        np.array(last_repu_values)[agent_idx, :], dones=truncations
                    )
                self.dilemma_buffer[agent_idx].compute_returns_and_advantage(
                    np.array(last_dilemma_values)[agent_idx, :], dones=truncations
                )

        # if this is the end of episode
        # the new dilemma obs will be access the reputation in the first
        if True in truncations:
            # breakpoint()
            if self.all_args.init_repu:
                repu_access_each_env_all, _, _, _ = self.get_repu_action(
                    init_repu=repu_next_obs
                )
                self.update_env_buffer_repu(
                    repu_access_each_env_all, store=False, update_env=False
                )

        rollout_infos = {
            "current_coop": np.mean([info["current_coop"] for info in infos]),
            "repu_std": np.mean(repu_std),
            "average_repu_stra": average_repu_stra,
            "normalized_repu": [normalized_avg_coop_repu, normalized_avg_defect_repu],
            "rews": rews,
            "dilemma_act": dilemma_actions_all,
            "action_reward": action_reward,
        }

        return rollout_infos

    def _store_dilemma_to_buffer(
        self, dilemma_action, rewards, values, log_probs, truncations, infos
    ) -> None:
        """
        Store transition in the replay buffer.
        """
        rewards = rewards.copy()

        # if this is the end of one episode
        if True in truncations:
            final_obs = np.array([info["final_observation"] for info in infos])

            dilemma_final_obs = self.process_repu_obs_to_dilemma_obs(final_obs)
            dilemma_final_actions, _, _, _ = self.get_actions(
                init_obs=dilemma_final_obs
            )
            repu_final_obs, _, _ = self.envs.action_step(
                dilemma_final_actions, move=False
            )
            repu_final_obs = self.process_single_repu_obs(repu_final_obs)

            single_final_obs = self.process_single_repu_obs(final_obs)

            for agent_idx in range(self.num_agents):
                step = self.repu_buffer[agent_idx].step - 1
                self.repu_buffer[agent_idx].next_observations[step] = single_final_obs[
                    :, agent_idx
                ]

                dilemma_obs = self.dilemma_trainer[agent_idx].policy.obs_to_tensor(
                    dilemma_final_obs[:, agent_idx]
                )

                with torch.no_grad():
                    if "PPO" in self.repu_trainer[0].__class__.__name__:
                        obses = np.array(
                            [
                                item
                                for subarray in repu_final_obs[:, agent_idx]
                                for item in subarray
                            ]
                        )
                        repu_obs = self.repu_trainer[agent_idx].policy.obs_to_tensor(
                            obses
                        )
                        terminal_repu_values = self.repu_trainer[
                            agent_idx
                        ].policy.predict_values(repu_obs)
                        terminal_repu_values = (
                            terminal_repu_values.clone()
                            .cpu()
                            .numpy()
                            .flatten()
                            .reshape(-1, self.neighbour_count)
                        )
                        self.repu_buffer[agent_idx].rewards_list[
                            self.repu_buffer[agent_idx].step - 1
                        ] += (self.gamma * terminal_repu_values)

                    terminal_dilemma_values = self.dilemma_trainer[
                        agent_idx
                    ].policy.predict_values(dilemma_obs)
                    rewards[:, agent_idx] += (
                        (self.gamma * terminal_dilemma_values)
                        .clone()
                        .cpu()
                        .numpy()
                        .flatten()
                    )

        # Store the transition in the buffer
        for agent_idx in range(self.num_agents):
            self.dilemma_buffer[agent_idx].insert(
                obs=self._old_dilemma_obs[:, agent_idx].copy(),
                # obs=self._last_dilemma_obs[:, agent_idx],
                action=dilemma_action[:, agent_idx].copy(),
                reward=rewards[:, agent_idx].copy(),
                # type: ignore[arg-type]
                episode_start=self._last_episode_starts.copy(),
                value=values[:, agent_idx].copy(),
                log_prob=log_probs[:, agent_idx].copy(),
            )
        self._last_episode_starts = truncations.copy()
        # if truncations[0]==True:
        #     breakpoint()

    def get_actions(self):
        raise NotImplementedError

    def run(self):
        raise NotImplementedError

    def warmup(self):
        raise NotImplementedError

    def log_rollouts(self, infos):
        """
        Log the information from the rollouts
        """
        rollout_infos = {
            "rollout/step_cooperation_level": infos["current_coop"],
        }
        self.log_metrics(rollout_infos)

    def log_metrics(self, logging_infos):
        """
        Log episode info.
        :param logging_infos: (dict) information about episode update.
        :param total_num_steps: (int) total number of training env steps.
        """
        for k, v in logging_infos.items():
            if self.use_wandb:
                wandb.log({k: v}, step=self._internal_timesteps)
            else:
                self.logger.add_scalars(k, {k: v}, self._internal_timesteps)

    def train_both(self):
        """
        Train the agents eval gat network and dilemma action network
        """
        # start_time = time.time()
        # Train the reputaion accessment network
        if not self.skip_repu_train  and self.train_upper:
            repu_train_infos = []  # train nfo for each agent
            # for agent_idx in np.random.permutation(self.num_agents):
            for agent_idx in range(self.num_agents):
                if "GAT" in self.repu_trainer[0].__class__.__name__:
                    repu_ti = self.repu_trainer[agent_idx].train(
                        batch_size=self.all_args.mini_batch,
                        rollout_buffer=self.repu_buffer[agent_idx],
                    )
                else:
                    repu_ti = self.repu_trainer[agent_idx].train(
                        batch_size=self.all_args.mini_batch,
                        rollout_buffer=self.repu_buffer[agent_idx],
                    )
                repu_train_infos.append(repu_ti)
            self.repu_ti = repu_ti.copy()


            self.repu_loss = np.array(
                [
                    entry["repu_train/loss"]
                    for entry in repu_train_infos
                    if "repu_train/loss" in entry
                ]
            )
            self.repu_gradient_loss = np.array(
                [
                    entry["repu_train/policy_gradient_loss"]
                    for entry in repu_train_infos
                    if "repu_train/policy_gradient_loss" in entry
                ]
            )
            self.repu_value_loss = np.array(
                [
                    entry["repu_train/value_loss"]
                    for entry in repu_train_infos
                    if "repu_train/value_loss" in entry
                ]
            )
            self.repu_approx_kl = np.array(
                [
                    entry["repu_train/approx_kl"]
                    for entry in repu_train_infos
                    if "repu_train/approx_kl" in entry
                ]
            )

        if not self.train_upper:
            # start_time = time.time()
            # Train the dilemma action network
            ppo_train_infos = []
            for agent_idx in np.random.permutation(self.num_agents):
                dilemma_ti = self.dilemma_trainer[agent_idx].train(
                    batch_size=self.all_args.mini_batch,
                    rollout_buffer=self.dilemma_buffer[agent_idx],
                )
                ppo_train_infos.append(dilemma_ti)
            self.dilemma_ti = dilemma_ti.copy()

            self.dilemma_gradient_loss = np.array(
                [
                    entry["dilemma_train/policy_gradient_loss"]
                    for entry in ppo_train_infos
                ]
            )
            self.dilemma_value_loss = np.array(
                [entry["dilemma_train/value_loss"] for entry in ppo_train_infos]
            )
            self.dilemma_loss = np.array(
                [entry["dilemma_train/loss"] for entry in ppo_train_infos]
            )
            self.dilemma_approx_kl = np.array(
                [entry["dilemma_train/approx_kl"] for entry in ppo_train_infos]
            )

        self.train_upper = not self.train_upper
        # sequential_duration  = time.time() - start_time
        ti = {
            "repu_train/loss": np.mean(self.repu_loss),
            "repu_train/lr": self.repu_ti["repu_train/lr"],
            "repu_train/ent_coef": self.repu_ti["repu_train/ent_coef"],
            "repu_train/policy_gradient_loss": np.mean(self.repu_gradient_loss),
            "repu_train/value_loss": np.mean(self.repu_value_loss),
            "repu_train/approx_kl": np.mean(self.repu_approx_kl),
            "repu_train/n_updates": self.repu_ti["repu_train/n_updates"],
            "repu_train/n_epochs": self.repu_ti["repu_train/n_epochs"],
            "dilemma_train/lr": self.dilemma_ti["dilemma_train/lr"],
            "dilemma_train/ent_coef": self.dilemma_ti["dilemma_train/ent_coef"],
            "dilemma_train/n_updates": self.dilemma_ti["dilemma_train/n_updates"],
            "dilemma_train/n_epochs": self.dilemma_ti["dilemma_train/n_epochs"],
            "dilemma_train/policy_gradient_loss": np.mean(self.dilemma_gradient_loss),
            "dilemma_train/value_loss": np.mean(self.dilemma_value_loss),
            "dilemma_train/loss": np.mean(self.dilemma_loss),
            "dilemma_train/approx_kl": np.mean(self.dilemma_approx_kl),
        }
        return ti

    def _dump_logs(self, episode):
        """
        Dump the logs to the logger
        """
        log_info = {}
        time_elapsed = max(
            (time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon
        )
        # Number of frames per seconds (includes time taken by gradient update)
        fps = int(self._internal_timesteps * self.n_rollout_threads / time_elapsed)
        log_info["time/fps"] = fps
        log_info["time/episode"] = episode

        print(
            "\n Dillemma_T({})S({}) Exp {} Network {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.".format(
                self.all_args.dilemma_T,
                self.all_args.dilemma_S,
                self.experiment_name,
                self.network_type,
                episode,
                self.total_episodes,
                self._internal_timesteps,
                self.num_env_steps,
                fps,
            )
        )

        for k, v in log_info.items():
            if self.use_wandb:
                wandb.log({k: v}, step=self._internal_timesteps)
            else:
                self.logger.add_scalars(k, {k: v}, self._internal_timesteps)

    def print_episode_state(self, episode_infos, extra_info):
        """
        Print the episode state
        """
        (
            episode_repu_loss,
            episode_dilemma_policy_loss,
            episode_dilemma_value_loss,
            episode_dilemma_loss,
            episode_dilemma_approx_kl,
        ) = extra_info
        print("-" * 44)
        # Reward
        print("| Reward/ {:>32} |".format(" " * 10))
        print(
            "|    Average Episode Rewards  {:>12.3f} |".format(
                episode_infos["results/avg_ep_rewards"]
            )
        )
        print(
            "|    Final Coop Reward  {:>18.3f} |".format(
                episode_infos["results/final_coop_reward"]
            )
        )
        print(
            "|    Final Defect Reward  {:>16.3f} |".format(
                episode_infos["results/final_defect_reward"]
            )
        )
        # Repu Train
        print("| Repu Train/ {:>29}|".format(" " * 10))
        print("|    Average Repu Loss  {:>18.3f} |".format(episode_repu_loss))
        print(
            "|    Average Repu std  {:>19.3f} |".format(episode_infos["repu/repu_std"])
        )
        print(
            "|    Repu n_updates(epoches)  {:>9.0f},{:>2.0f} |".format(
                episode_infos["repu_train/n_updates"],
                episode_infos["repu_train/n_epochs"],
            )
        )
        print(
            "|    Repu Learning Rate  {:>17.3f} |".format(
                episode_infos["repu_train/lr"]
            )
        )
        print(
            "|    Entropy Coefficient  {:>16.3f} |".format(
                episode_infos["repu_train/ent_coef"]
            )
        )

        # Dilemma Train
        print("| Dilemma Train/ {:>26}|".format(" " * 10))
        print(
            "|    Average Dilemma Value Loss  {:>9.3f} |".format(
                episode_dilemma_value_loss
            )
        )
        print("|    Average Dilemma Loss  {:>15.3f} |".format(episode_dilemma_loss))
        print("|    Approx KL  {:>26.3f} |".format(episode_dilemma_approx_kl))
        print(
            "|    Dilemma n_updates(epoches)  {:>6.0f},{:>2.0f} |".format(
                episode_infos["dilemma_train/n_updates"],
                episode_infos["dilemma_train/n_epochs"],
            )
        )
        print(
            "|    Dilemma Learning Rate  {:>14.3f} |".format(
                episode_infos["dilemma_train/lr"]
            )
        )
        print(
            "|    Entropy Coefficient  {:>16.3f} |".format(
                episode_infos["dilemma_train/ent_coef"]
            )
        )
        # Reputation
        print("| Reputation/ {:>29}|".format(" " * 10))
        print(
            "|    Average Coop Repu  {:>18.3f} |".format(
                episode_infos["repu/avg_coop_repu"]
            )
        )
        print(
            "|    Average Defect Repu  {:>16.3f} |".format(
                episode_infos["repu/avg_defect_repu"]
            )
        )
        print("|" + " " * 36 + "-" * 4 + " " * 2 + "|")
        print(
            "|    Normalized Coop Repu   {:>14.3f} |".format(
                episode_infos["repu/avg_coop_repu_normalized"]
            )
        )
        print(
            "|    Normalized Defect Repu   {:>12.3f} |".format(
                episode_infos["repu/avg_defect_repu_normalized"]
            )
        )
        # Environment
        print("| Environment/ {:>28}|".format(" " * 10))
        print(
            "|    Average Coop Level  {:>17.3f} |".format(
                episode_infos["results/avg_cooperation_level"]
            )
        )
        print(
            "|    Finnal Coop Level  {:>18.3f} |".format(
                episode_infos["results/final_cooperation_level"]
            )
        )
        print("-" * 44, "\n")

    def write_to_video(self, all_frames, episode, video_type="train"):
        """
        record this episode and
        save the gif to local or wandb
        """
        import imageio

        images = []
        for png in all_frames:
            img = imageio.imread(png)
            images.append(img)

        if self.all_args.use_wandb:
            import wandb

            imageio.mimsave(
                str(self.gif_dir) + "/episode.gif",
                images,
                duration=self.all_args.ifi,
            )
            wandb.log(
                {
                    video_type: wandb.Video(
                        str(self.gif_dir) + "/episode.gif", fps=4, format="gif"
                    )
                },
                step=self._internal_timesteps,
            )

        elif self.all_args.save_gifs:
            imageio.mimsave(
                str(self.gif_dir) + "/{}_episode_{}.gif".format(video_type, episode),
                images,
                duration=self.all_args.ifi,
            )
