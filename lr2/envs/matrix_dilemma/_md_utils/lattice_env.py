import gymnasium
import numpy as np
from pettingzoo import AECEnv
from pettingzoo.utils.agent_selector import agent_selector
from gymnasium import spaces
from matplotlib import colors
import matplotlib.pyplot as plt
from .._md_utils import utils

class LatticeEnv(AECEnv):
    """
    A Matirx game environment has gym API for soical dilemma
    """

    # The metadata holds environment constants
    metadata = {
        "name": "lattice_v0",
        "is_parallelizable": True,
        "render_modes": ["rgb_array"],
    }

    def __init__(
        self,
        scenario,
        world,
        max_cycles,
        render_mode=None,
        args=None,
    ):
        super().__init__()
        self.render_mode = render_mode

        self._seed()

        self.args = args
        #  game terminates after the number of cycles
        self.max_cycles = max_cycles
        self.scenario = scenario
        self.world = world
        # get the agent name list
        self.agents = [agent.name for agent in self.world.agents]
        self.possible_agents = self.agents[:]
        # get agent index by name
        self._index_map = {
            agent.name: idx for idx, agent in enumerate(self.world.agents)
        }
        self._agent_selector = agent_selector(self.agents)

        # current dilemma actions for all agents
        self.current_actions = [agent.action.s for agent in self.world.agents]
        self.current_repu = [agent.reputation for agent in self.world.agents]
        # current reputation accessment for all agents
        self.current_repu_accessment = np.array(
            [agent.action.ra for agent in self.world.agents], dtype=np.float32
        )

        self.get_spaces()

    def _seed(self, seed=None):
        self.seed = seed

    def get_spaces(self):
        """
        Get action and observation spaces
        """
        self.action_spaces = dict()
        self.observation_spaces = dict()
        self.graph_structure = dict()

        repu_obs_space = spaces.Dict(
            {
                "n_d": spaces.Discrete(  # 2-hop neighbour dilemma action accoridng to the graph structure
                    2
                ),
                "n_r_a": spaces.MultiDiscrete(  # neighbour's neighbour's reputation accessement to its four neighbours
                       [2, 2, 2, 2] if self.args.network_type == "lattice" 
                        else [2] * 3 if self.args.network_type == "honeycomb"
                        else [2] * 8 if self.args.network_type == "moore" 
                        else [2] *self.args.mixed_num if self.args.network_type == "mixed"
                        else ValueError(f"Unsupported network type: {self.args.network_type}")
                ),
            }
        )
        if self.args.chang_obs and self.args.disable_repu:
            dilemma_obs_space = spaces.Dict(
                {
                    "n_d": spaces.MultiDiscrete(  # 1-hop neighbour dilemma action accoridng to the graph structure
                        [2,2,2,2]
                    ),
                }
            )
        else:
            dilemma_obs_space = spaces.Dict(
                {
                    "n_r": spaces.Box(  # neighbour's reputation
                        low=0,
                        high=1,
                        shape=(4,) if self.args.network_type == "lattice" 
                        else (3,) if self.args.network_type == "honeycomb"
                        else (8,) if self.args.network_type == "moore" 
                        else (self.args.mixed_num,) if self.args.network_type == "mixed"
                        else ValueError(f"Unsupported network type: {self.args.network_type}"),
                        dtype=np.float32,
                    ),
                    "s_r": spaces.Box(  # self reputation
                        low=0,
                        high=1,
                        shape=(1,),
                        dtype=np.float32,
                    ),
                }
            )
        act_space = spaces.Dict(
            {
                "d": spaces.Discrete(2),  # dilemma action
                "r_a": spaces.Discrete(2),
            }
        )

        for agent in self.world.agents:
            self.observation_spaces[agent.name] = {
                "repu": repu_obs_space,
                "dilemma": dilemma_obs_space,
            }
            self.action_spaces[agent.name] = act_space

    def reset(self, seed=None, options="truncation"):
        """
        Reset the environment
        """
        # reset world
        if seed is not None:
            self._seed(seed=seed)

        # reset world for agent strategy
        self.scenario.reset_world(self.world, self.seed)

        # reset agents
        self.agents = self.possible_agents[:]
        self.rewards = {name: 0.0 for name in self.agents}
        self._cumulative_rewards = {name: 0.0 for name in self.agents}
        self.terminations = {name: False for name in self.agents}
        self.truncations = {name: False for name in self.agents}
        self.infos = {name: {} for name in self.agents}

        self.agent_selection = self._agent_selector.reset()

        # get current actions
        self.current_actions = [agent.action.s for agent in self.world.agents]
        # self.current_actions = [1.0 for agent in self.world.agents]
        self.final_actions = [agent.action.s for agent in self.world.agents]
        self.current_repu = [agent.reputation for agent in self.world.agents]
        self.current_repu_accessment = np.array(
            [agent.action.ra for agent in self.world.agents], dtype=np.float32
        )
        if options == "truncation":
            self.steps = 0

        obs_all = self.observe_all()

        # Get initial cooperative level
        cl = self.state()
        return obs_all, cl

    def observe_single(self, agent):
        """
        observe repu info for one sepcific agent
        """
        return self.scenario.observation(
            self.world.agents[self._index_map[agent]], self.world
        )

    def observe_all(self):
        """
        observe_single current scenario info for all agents
        :return: list of observations
        """
        observations = []  # Create an empty list to store observations.
        for agent in self.world.agents:
            observation = self.observe_single(
                agent.name
            )  # Get the observation for each agent.
            observations.append(observation)  # Append the observation to the list.
        return observations

    def state(self) -> np.ndarray:
        """
        return the cooperative state for whole system
        """
        coop_level = np.mean(self.current_actions)
        return 1 - coop_level

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def _execute_world_step(self, move=True):
        """
        Apply the actions of all agents in this round to the environment
        """
        # Access reward
        sub_reward_n = []  # reward that interact with each neighbour
        reward_n = []
        action_n = []

        # update agent's action (for agent class)
        for i, agent in enumerate(self.world.agents):
            action = self.current_actions[i] if move else self.final_actions[i]
            self._set_action(action, agent)
            action_n.append(action)

        # no action for the world
        self.world.step()
        if move:
            for agent in self.world.agents:
                agent_reward, reward_list = self.scenario.reward(agent, self.world)


                # set agent reward
                agent.reward = agent_reward
                agent.reward_list = reward_list
                self.rewards[agent.name] = agent_reward
                # add repution to the agent reward

                sub_reward_n.append(reward_list)
                reward_n.append(agent.reward)
        obs_n = self.observe_all()

        return obs_n, np.array(reward_n), np.array(sub_reward_n)

    def step(self, action, move=True):
        """
        Contains most of the logic of this environment
        Automatically switches control to the next agent.
        """
        # get current agent index
        cur_agent = self.agent_selection
        current_idx = self._index_map[self.agent_selection]
        next_idx = (current_idx + 1) % self.num_agents
        # set agent_selection to next agent
        self.agent_selection = self._agent_selector.next()

        # set action for all current agents
        if move:
            self.current_actions[current_idx] = action
        else:
            self.final_actions[current_idx] = action

        if next_idx == 0:
            # get obs after taking action
            obs_n, reward_n, sub_reward_n = self._execute_world_step(move)

            return obs_n, reward_n, sub_reward_n

        else:  # clear rewards for all agents
            self._clear_rewards()

        self._cumulative_rewards[cur_agent] = 0
        self._accumulate_rewards()

    def _update_world_reputation(self):
        """
        update agent reputaion accessment action and assigned reputation
        """
        repu_reward_n = []

        # set reputation accessment for all agents
        for idx, agent in enumerate(self.world.agents):
            agent.action.ra = self.current_repu_accessment[idx]
        # update agent reputation
        average_repu_stra, repu_std = self.world.update_reputation_assignment()

        if self.args.network_type == "mixed":
            # if self.args.mixed_num==0:
            utils.gen_random_reciprocal_neighbours(self.world.agents,self.args.mixed_num)


        # observe reputation for all agents
        obs_all = self.observe_all()

        # Check if the state is below a certain threshold for termination
        termination = False
        coop_level = self.state()


        infos = {
            "current_coop": coop_level,
        }

        # Check if comparison of rewards is enabled
        if self.args.compare_reward_pattern == "all":
            # Initialize a list to store rewards for each strategy
            strategy_reward = [[], []]
            # Iterate through agents to categorize rewards based on their actions
            for idx, agent in enumerate(self.world.agents):
                strategy_reward[agent.action.s].append(agent.reward)
            strategy_mean_reward = [
                np.nanmean(s_r) if s_r else 0 for s_r in strategy_reward
            ]

        for agent in self.world.agents:
            repu_reward =  agent.reputation * self.args.repu_reward_coef

            if self.args.compare_reward_pattern != "none":
                if self.args.compare_reward_pattern == "neighbour":
                    strategy_reward = [[], []]
                    dim_lengths = [0, 0]
                    for idx, neighbour_idx in enumerate(agent.neighbours):
                        strategy_reward[
                            self.world.agents[neighbour_idx].action.s
                        ].append(self.world.agents[neighbour_idx].reward)
                        dim_lengths[self.world.agents[neighbour_idx].action.s] += 1
                    strategy_mean_reward = [
                        np.nanmean(s_r) if s_r else 0 for s_r in strategy_reward
                    ]
                    ratios = [element / np.sum(dim_lengths) for element in dim_lengths]
                else:
                    # count neighbour strategy
                    dim_lengths = [0, 0]
                    for _, neighbour_idx in enumerate(agent.neighbours):
                        dim_lengths[self.world.agents[neighbour_idx].action.s] += 1

                    # add self strategy into count
                    dim_lengths[agent.action.s] += 1
                    ratios = [element / np.sum(dim_lengths) for element in dim_lengths]

                if agent.action.s == 0:
                    agent_strategy_reward = (
                        agent.reward * ratios[0] - strategy_mean_reward[1] * ratios[1]
                    )
                else:
                    agent_strategy_reward = (
                        agent.reward * ratios[1] - strategy_mean_reward[0] * ratios[0]
                    )

                repu_reward_n.append(agent_strategy_reward + repu_reward)
                # repu_reward_n.append(agent_strategy_reward)
            else:
                if self.args.algorithm_name=='IPPO'or self.args.disable_repu: # for IPPO, the reputation reward is the same as the agent reward
                    repu_reward_n.append(agent.reward)
                elif self.args.repu_reward_coef>=1:
                    repu_reward_n.append(agent.reward+repu_reward)
                else: # if repu_reward_coef set to 0, then the total reward is the same as the agent reward
                    if agent.consider_repu_reward:
                        repu_reward_n.append(
                            self.args.repu_reward_coef * agent.reward
                            + (1 - self.args.repu_reward_coef)
                            * (agent.reputation * agent.reward)
                        )
                    else:
                        repu_reward_n.append(agent.reward)


        return obs_all, repu_reward_n, average_repu_stra, repu_std, termination, infos

    def repu_step(self, repu_accessment, update_env=True):
        """
        Step for update agent reputation accessment actions
        when all update actions are done, then update the reputation
        """
        current_idx = self._index_map[self.agent_selection]
        next_idx = (current_idx + 1) % self.num_agents
        # set agent_selection to next agent
        self.agent_selection = self._agent_selector.next()

        # set action for all current agents
        self.current_repu_accessment[current_idx] = repu_accessment

        # do _execute_world_step only when all agents have gone through a step once
        if next_idx == 0:
            (
                repu_obs_all,
                repu_reward_n,
                average_repu_stra,
                repu_std,
                termination,
                infos,
            ) = self._update_world_reputation()
            truncation = False
            if update_env:
                self.steps += 1
                if self.steps >= self.max_cycles:
                    truncation = True
                    for name in self.agents:
                        self.terminations[name] = True
                    infos["final_observation"] = repu_obs_all.copy()
                    infos["cumulative_payoffs"] = list(
                        self._cumulative_rewards.values()
                    )

            self.current_repu = [agent.reputation for agent in self.world.agents]


            return (
                repu_obs_all,
                repu_reward_n,
                self.current_repu,
                average_repu_stra,
                repu_std,
                termination,
                truncation,
                infos,
            )

    def _set_action(self, action, agent):
        """
        Set current strategy action for a particular agent.
        """
        agent.action.s = action

    def render(self, mode, step):
        self.render_mode = mode
        if self.render_mode is None:
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode."
            )
            return

        # Define color map for rendering
        color_set = np.array(
            [
                "#ff1414",
                "#e9e8fe",
                "#0c056d",
            ]
        )

        cmap = colors.ListedColormap(np.array(["#0c056d", "red"]))
        # Create the colormap
        cmap_repu = colors.LinearSegmentedColormap.from_list("my_list", color_set, N=9)

        # Convert agent actions to a 2D NumPy array
        action_n = np.array(self.current_actions).reshape(
            self.scenario.env_dim, self.scenario.env_dim
        )
        reputaion_n = np.array(self.current_repu).reshape(
            self.scenario.env_dim, self.scenario.env_dim
        )
        fig, axs = plt.subplots(1, 2, figsize=(6, 3))
        for idx, ax in enumerate(axs.flat):
            if idx == 0:
                im = ax.imshow(action_n, cmap=cmap, vmin=0, vmax=1)
            else:
                im = ax.imshow(reputaion_n, cmap=cmap_repu, vmin=0, vmax=1)
                fig.colorbar(im, ax=ax)
            ax.axis("off")
            fig.suptitle(
                "Mode {}, Step {}, Dilemma {}".format(
                    mode, step, self.world.payoff_matrix[1][0]
                )
            )

        fig.tight_layout()

        # Save the plot as an image in memory
        import io

        with io.BytesIO() as buffer:  # use buffer memory
            plt.savefig(buffer, format="png", dpi=plt.gcf().dpi)
            buffer.seek(0)
            image = buffer.getvalue()
            buffer.flush()
        plt.close()
        return image,action_n,reputaion_n

    def close(self):
        pass
