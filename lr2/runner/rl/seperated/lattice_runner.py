import time
import numpy as np
from .base_runner import Runner
from stable_baselines3.common.utils import should_collect_more_steps
import torch
import lr2.utils.tools as utils
from gymnasium import spaces


class LatticeRunner(Runner):
    """
    Runner class to perform training, evaluation. and data collection for the RL Lattice.
    See parent class for details.
    """

    def __init__(self, config):
        super(LatticeRunner, self).__init__(config)
        self.continue_training = True
        self.last_best_mean_payoff = -np.inf
        self.last_best_cooperation_level = -np.inf
        self.no_improvement_evals = 0
        self.max_no_improvement_evals=10
        self.continue_training=True

    def run(self):
        """
        Run the training and evaluation loop
        """
        self.warmup()

        self.have_train = 0
        _internal_episode_step = 0  # internal episode step counter per rollout
        episode = 1  # episode counter
        self.br_start_idx_in_eposide = 0

        episode_rollouts_info = []  # episode information
        log_episode_repu_loss = []  # episode gat loss
        log_episode_dilemma_policy_loss = []
        log_episode_dilemma_value_loss = []
        log_episode_dilemma_loss = []
        log_episode_approx_kl = []
        all_frames = []  # frames for video recording
        action_arraty_log=[]
        repu_array_log=[]

        while self._internal_timesteps < self.num_env_steps:
            if not self.continue_training:
                break

            num_collected_steps, num_collected_episodes = 0, 0

            # reset repu buffer befter any new train collect
            for br in self.repu_buffer:
                br.reset()
            for br in self.dilemma_buffer:
                br.reset()

            # how many steps need to be collected before training
            while should_collect_more_steps(
                self.train_freq, num_collected_steps, num_collected_episodes
            ):
                # if one episode is done
                if _internal_episode_step >= self.episode_length:
                    if self.have_train>0 and (
                        episode % self.log_interval == 1
                        or episode == self.total_episodes
                        or self.log_interval == 1
                    ):
                        extra_info = (
                            np.mean(log_episode_repu_loss),
                            np.mean(log_episode_dilemma_policy_loss),
                            np.mean(log_episode_dilemma_value_loss),
                            np.mean(log_episode_dilemma_loss),
                            np.mean(log_episode_approx_kl),
                        )
                        # process and log the episode information
                        self.process_log_episode(
                            episode,
                            self.train_infos,
                            episode_rollouts_info,
                            extra_info,
                        )
                        self.StopTrainingOnNoModelImprovement()
                        if self.have_train>=2:
                            log_episode_repu_loss.clear()
                            log_episode_dilemma_policy_loss.clear()
                            log_episode_dilemma_value_loss.clear()
                            log_episode_dilemma_loss.clear()
                            log_episode_approx_kl.clear()
                            self.have_train = 0

                    if self.all_args.use_render and (
                        episode % self.video_interval == 0
                        or episode == self.total_episodes - 1
                        or episode == 1  # first episode
                    ):
                        self.write_to_video(all_frames, episode)

                    num_collected_episodes += 1
                    _internal_episode_step = 0
                    episode += 1
                    episode_rollouts_info.clear()
                    all_frames.clear()
                    

                # record every step for current episode
                if self.all_args.use_render and (
                    episode % self.video_interval == 0
                    or episode == self.total_episodes - 1
                    or episode == 1
                ):
                    image,action_array,repu_array = self.render(self._internal_timesteps)
                    all_frames.append(image[-1])

                    if self.all_args.save_distribution and _internal_episode_step+1 == self.episode_length:
                        action_arraty_log.append(action_array)
                        repu_array_log.append(repu_array)
                        utils.save_array(
                            action_arraty_log, self.plot_dir, "agent_action.npz"
                        )                        
                        utils.save_array(
                            repu_array_log, self.plot_dir, "agent_repu.npz"
                        )         

                _last_rollout = not should_collect_more_steps(
                    self.train_freq, num_collected_steps + 1, num_collected_episodes
                )
                # info from roullouts process
                one_step_rollouts_infos = self.collect_rollouts(
                    _last_rollout
                )  # based run method

                episode_rollouts_info.append(one_step_rollouts_infos)
                # log the rollouts
                self.log_rollouts(one_step_rollouts_infos)

                _internal_episode_step += 1
                num_collected_steps += 1
                # update the number of timesteps
                self._internal_timesteps += self.n_rollout_threads

            # this round collect finish Traing starts
            self.have_train +=1
            self.train_infos = self.train_both()
            for agent_idx in range(self.num_agents):
                self.repu_trainer[agent_idx]._update_current_progress_remaining(
                    self._internal_timesteps, self.num_env_steps
                )
                self.dilemma_trainer[agent_idx]._update_current_progress_remaining(
                    self._internal_timesteps, self.num_env_steps
                )
            # breakpoint()
            self.log_metrics(self.train_infos)
            log_episode_repu_loss.append(self.train_infos["repu_train/loss"])
            log_episode_dilemma_policy_loss.append(
                self.train_infos["dilemma_train/policy_gradient_loss"]
            )
            log_episode_dilemma_value_loss.append(
                self.train_infos["dilemma_train/value_loss"]
            )
            log_episode_dilemma_loss.append(self.train_infos["dilemma_train/loss"])
            log_episode_approx_kl.append(self.train_infos["dilemma_train/approx_kl"])

    def warmup(self):
        """
        Warmup the environment
        """
        if self._last_dilemma_obs is None:
            assert self.envs is not None
            # reset env
            init_obs, coop_level = self.envs.reset()
            condense_obs = self.process_repu_obs_to_dilemma_obs(init_obs)
            self._last_dilemma_obs = condense_obs.copy()

        if self._last_repu_obs is None:
            self._last_episode_starts = np.ones((self.envs.num_envs,), dtype=bool)

        print(
            "====== Initial Cooperative Level {:.2f} ======".format(np.mean(coop_level))
        )

        # total episode number
        self.total_episodes = (
            int(self.num_env_steps) // self.episode_length // self.n_rollout_threads
        )
        # training start time
        self.start_time = time.time_ns()

        # current timesteps within the single thread
        self._internal_timesteps = 0

    @torch.no_grad()
    def get_actions(self,init_obs=None):
        """
        Collect(predict) actions according to the current step observations.
        """
        dilemma_obs = init_obs.copy() if init_obs is not None else (self._last_dilemma_obs).copy()
        dilemma_action_all = []
        dilemma_values_all = []
        log_probs_all = []
        entropy_all = []   

        # Update the dilemma_buffer with the reputation accessment
        assert dilemma_obs is not None, "self._last_dilemma_obs was not set"
        # dilemme action phase
        for agent_idx in range(self.num_agents):
            # get the dilemma action, value and log_prob according to the observation
            with torch.no_grad():
                dilemma_actions, dilemma_values, log_probs, entropy = self.dilemma_trainer[
                    agent_idx
                ].predict(dilemma_obs[:, agent_idx].copy())
            # Convert the actions, values, and log_probs to NumPy arrays

            dilemma_actions = utils.t2n(dilemma_actions)

            # dilemma_actions=np.zeros_like(dilemma_actions) if random.random()<0.9 else np.ones_like(dilemma_actions)

            dilemma_values = utils.t2n(dilemma_values)
            log_probs = utils.t2n(log_probs)
            entropy = utils.t2n(entropy)
            # Append the agent action to the list
            dilemma_action_all.append(dilemma_actions)
            dilemma_values_all.append(dilemma_values)
            log_probs_all.append(log_probs)
            entropy_all.append(entropy)

        return (
            np.column_stack(dilemma_action_all),
            np.column_stack(dilemma_values_all),
            np.column_stack(log_probs_all),
            np.column_stack(entropy_all),
        )
    
    @torch.no_grad()
    def get_repu_action(self, init_repu=None):
        repu_obs = init_repu.copy() if init_repu is not None else (self._last_repu_obs).copy()

        single_repu_obs = self.process_single_repu_obs(repu_obs)

        repu_action_all = []
        repu_values_all = []
        repu_log_probs_all = []
        reputation_accessmention_all = []
        # accessment phase
        for agent_idx in range(self.num_agents):
            # Give the reputation accessment for its four neighbour given the observation
            if "GAT" in self.repu_trainer[0].__class__.__name__:
                reputaion_accessment, weight = self.repu_trainer[agent_idx].predict(
                    repu_obs[:, agent_idx]
                )
                weight = utils.t2n(weight)
            else:           
                obses=np.array([item for subarray in single_repu_obs[:, agent_idx] for item in subarray])
                reputaion_accessment, repu_values, repu_log_probs,entropy = self.repu_trainer[
                    agent_idx
                ].predict(obses)
                repu_values = utils.t2n(repu_values).reshape(-1,self.neighbour_count)
                repu_log_probs = utils.t2n(repu_log_probs).reshape(-1,self.neighbour_count)
                reputaion_accessment = utils.t2n(reputaion_accessment).reshape(-1,self.neighbour_count)


                repu_values_all.append(repu_values)
                repu_log_probs_all.append(repu_log_probs)

            repu_action_all.append(reputaion_accessment)
            # Rescale and perform action
            clipped_reputaion_accessment = reputaion_accessment.copy()
            if isinstance(self.repu_trainer[agent_idx].policy.action_space, spaces.Box):
                # Otherwise, clip the actions to avoid out of bound error
                # as we are sampling from an unbounded Gaussian distribution
                clipped_reputaion_accessment = np.clip(
                    reputaion_accessment,
                    self.repu_trainer[agent_idx].policy.action_space.low,
                    self.repu_trainer[agent_idx].policy.action_space.high,
                )
            # Append the agent action to the list
            reputation_accessmention_all.append(clipped_reputaion_accessment)
        
        reputation_accessmention_all = np.array(
            reputation_accessmention_all
        )  


        # Reshape to handle multi-env reputation accessment
        # from each agent all env to  each env all agents
        repu_access_each_env_all = utils.collect_reputation_accessment(
            reputation_accessmention_all
        )  

        # update agent reputation in env and dilemma_buffer dilemma observation
        repu_action_all = np.stack(repu_action_all, axis=1)
        repu_values_all = np.stack(repu_values_all, axis=1)
        repu_log_probs_all = np.stack(repu_log_probs_all, axis=1)

        return (
            repu_access_each_env_all,
            repu_action_all,
            repu_values_all ,
            repu_log_probs_all 
        )

    def get_action_by_sample(self, agent_idx):
        """
        Get action by sample
        """
        r_a_actions = np.array(
            [
                self.repu_trainer[agent_idx].policy.action_space.sample()
                for _ in range(self.n_rollout_threads)
            ]
        )
        d_actions = np.array(
            [
                self.dilemma_trainer[agent_idx].policy.action_space.sample()
                for _ in range(self.n_rollout_threads)
            ]
        )
        d_values = np.random.rand(self.n_rollout_threads, 1)
        d_log_probs = np.random.rand(self.n_rollout_threads, 1)

        return d_actions, r_a_actions, d_values, d_log_probs

    def update_env_buffer_repu(
        self,
        repu_access_each_env_all,
        repu_action_all=None,
        repu_value_all=None,
        repu_log_value_all=None,
        update_env=True,
        store=True,
    ) -> np.ndarray:
        """
        Update the reputation based on neighbour's acceessment

        :return: average reputation for cooperation and defection
        """
        # Update the reputation accessment for all agents
        new_repu_obs,repu_reward, repu_all, average_repu_stra, repu_std,terminations, truncations, infos = self.envs.update_repu(
            repu_access_each_env_all, update_env
        )
        if store:
            repu_obs= (self._last_repu_obs).copy()
            next_repu_obs = new_repu_obs.copy()
            single_repu_obs = self.process_single_repu_obs(repu_obs)
            next_single_repu_obs = self.process_single_repu_obs(next_repu_obs)
            # update the dilemma_buffer repu_obs info with the updated observation
            for agent_idx in range(self.num_agents):
                # insert repu accessment into repu_buffer
                self.repu_buffer[agent_idx].insert(
                    obs=single_repu_obs[:, agent_idx],
                    next_obs=next_single_repu_obs[:, agent_idx],
                    action=(
                        repu_action_all[:, agent_idx]
                        if repu_action_all is not None
                        else None
                    ),
                    episode_start=self._last_episode_starts,
                    value=(
                        repu_value_all[:, agent_idx]
                        if repu_value_all is not None
                        else None
                    ),
                    log_prob=(
                        repu_log_value_all[:, agent_idx]
                        if repu_log_value_all is not None
                        else None
                    ),
                )

        condense_obs = self.process_repu_obs_to_dilemma_obs(new_repu_obs)
        # condense_obs = new_repu_obs
        self._last_dilemma_obs = condense_obs.copy()

        return new_repu_obs, repu_reward,repu_all,average_repu_stra, repu_std,terminations, truncations, infos

    def process_log_episode(
        self, episode, episode_info, episode_rollouts_info, extra_info
    ):
        """
        Log the episode information
        """

        self._dump_logs(episode)

        repu_std = []
        noramlized_repu = [[], []]
        stra_repu = [[], []]
        action_reward=[[],[]]
        episode_avg_rwds = []
        episode_acts = []

        # iterate each rollout
        for rollout in episode_rollouts_info:
            # iterate each rollout inside env
            
            repu_std.append(rollout["repu_std"])
            stra_repu[0].append(rollout["average_repu_stra"][:, 0]) # coop reputation
            stra_repu[1].append(rollout["average_repu_stra"][:, 1]) # defect reputaion
            action_reward[0].append(rollout["action_reward"][0])
            action_reward[1].append(rollout["action_reward"][1])
            noramlized_repu[0].append(rollout["normalized_repu"][0])
            noramlized_repu[1].append(rollout["normalized_repu"][1])
            episode_avg_rwds.append(rollout["rews"])
            episode_acts.append(rollout["dilemma_act"])

    
        episode_info["repu/repu_std"] = np.mean(repu_std)
        episode_info["repu/avg_coop_repu"] = np.nanmean(stra_repu[0])
        episode_info["repu/avg_defect_repu"] = np.nanmean(stra_repu[1])
        episode_info["repu/avg_coop_repu_normalized"] =len(noramlized_repu[0]) / np.sum(1.0 / np.array(noramlized_repu[0]))
        episode_info["repu/avg_defect_repu_normalized"] =len(noramlized_repu[1]) / np.sum(1.0 / np.array(noramlized_repu[1]))
        episode_info["repu/final_coop_repu"] = np.nanmean([arr[int(len(arr) * 0.7):] for arr in stra_repu[0]])
        episode_info["repu/final_defect_repu"] = np.nanmean([arr[int(len(arr) * 0.7):] for arr in stra_repu[1]])
        # Episode env action information
        episode_info["results/avg_ep_rewards"] = np.mean(episode_avg_rwds)
        episode_info["results/finnal_rewards"] = np.mean([arr[int(len(arr) * 0.7):] for arr in episode_avg_rwds])
        episode_info["results/avg_cooperation_level"] = 1 - np.mean(episode_acts)
        episode_info["results/avg_coop_reward"] = np.nanmean(action_reward[0])
        episode_info["results/avg_defect_reward"] = np.nanmean(action_reward[1])
        episode_info["results/final_coop_reward"] = np.nanmean(action_reward[0][int(len(action_reward[0]) * 0.7):])
        episode_info["results/final_defect_reward"] = np.nanmean(action_reward[1][int(len(action_reward[1]) * 0.7):])
        self.best_mean_cooperation_level=1 - np.mean(episode_acts)
        self.best_mean_payoff=np.mean(episode_avg_rwds)

        episode_info["results/final_cooperation_level"]=1-np.mean([arr[int(len(arr) * 0.7):] for arr in episode_acts])
        episode_info["time/no_more_improvement"] = self.no_improvement_evals

        self.print_episode_state(episode_info, extra_info)
        self.log_metrics(episode_info)

    def extract_episode_info_from_buffer(self):
        """
        Extract the episode information from the dilemma_buffer
        """
        episode_rwds = []
        episode_acts = []

        # iterate each agent's dilemma_buffer
        for br in self.dilemma_buffer:
            episode_rwds.append(br.rewards)
            episode_acts.append(br.actions)

        return episode_rwds, episode_acts

    def process_repu_obs_to_dilemma_obs(self, repu_obs: np.ndarray) -> np.ndarray:
        """
        Condense the reputation observation to dilemma observation based on network type.
        """
        # Prepare the shapes and dtype from a sample observation
        obs_shape = self.dilemma_buffer[0].obs_shape
        num_keys = len(obs_shape)

        # Number of rollouts to process
        process_rollout = self.n_rollout_threads

        # Initialize the dilemma_obs array with zeros
        dilemma_obs = np.zeros(
            (process_rollout, self.num_agents, num_keys, self.neighbour_count),
            dtype=np.float32,
        )

        for rollout_idx in range(process_rollout):
            for agent_idx in range(self.num_agents):
                for key_idx, (key, shape) in enumerate(obs_shape.items()):
                    dilemma_obs[rollout_idx, agent_idx, key_idx] = repu_obs[
                            rollout_idx, agent_idx
                        ][key]
    
        # Convert the structured dilemma_obs back to the desired dictionary format
        final_dilemma_obs = [
            [
                {
                    key: dilemma_obs[rollout_idx, agent_idx, key_idx]
                    if key != "s_r"
                    else dilemma_obs[rollout_idx, agent_idx, key_idx][0]  # Single value for "s_r"
                    for key_idx, key in enumerate(obs_shape.keys())
                }
                for agent_idx in range(self.num_agents)
            ]
            for rollout_idx in range(process_rollout)
        ]


        return np.array(final_dilemma_obs)



    def process_single_repu_obs(self, process_obs: np.ndarray) -> np.ndarray:
        """
        Process reputation observations for agents based on the network type.
        """
        process_rollout = self.n_rollout_threads
        # Prepare the shapes and dtype from a sample observation
        obs_shape = self.repu_buffer[0].obs_shape
        num_keys = len(obs_shape)

        repu_obs = np.zeros(
            (process_rollout, self.num_agents, num_keys),
            dtype=np.float32,
        )

        for rollout_idx in range(process_rollout):
            for agent_idx in range(self.num_agents):
                repu_obs_list = []
                for n_index in range(self.neighbour_count):  # Dynamically adjust for 4 or 8 neighbors
                    single_repu_obs = {}
                    for key_idx, (key, shape) in enumerate(obs_shape.items()):
                        if key == "n_d":
                            # Self-reputation remains unchanged
                            single_repu_obs[key] = np.array(process_obs[rollout_idx, agent_idx][key][n_index],dtype=np.float32)
                        else:
                            # Process observations for the current neighbor
                            single_repu_obs[key] = process_obs[rollout_idx, agent_idx][key][n_index]
                    repu_obs_list.append(single_repu_obs)
                repu_obs_list = np.array(repu_obs_list)
                process_obs[rollout_idx, agent_idx] = repu_obs_list
        return process_obs


    @torch.no_grad()
    def render(self, _internal_timesteps, render_env=0):
        """
        Visualize the env at current state
        """
        if render_env == 0:
            envs = self.envs
            render_mod = "train"
        else:
            envs = self.eval_envs
            render_mod = "eval"
        image,action_array,repu_array = envs.render(render_mod, _internal_timesteps)
        return image,action_array,repu_array

    def StopTrainingOnNoModelImprovement(self):
        '''
        Stop the training early if there is no new best model (new best mean reward)
        after more than N consecutive evaluations.
        '''

        continue_training = True
        c_l='{:.2f}'.format(self.best_mean_cooperation_level)
        c_l = float(c_l)

        if 0.04 < c_l < 0.96:
            self.no_improvement_evals = 0
        else:
            self.no_improvement_evals += 1
            print("="*50)
            print(
                f"No new best model found in the last {self.no_improvement_evals:d} evaluations"
            )
            print("="*50)
            if self.no_improvement_evals > self.max_no_improvement_evals:
                continue_training = False

        self.last_best_mean_payoff = self.best_mean_payoff
        self.last_best_cooperation_level=c_l

        if not continue_training:
            print(
                f"Stopping training because there was no new best model in the last {self.no_improvement_evals:d} evaluations"
            )

        self.continue_training=continue_training