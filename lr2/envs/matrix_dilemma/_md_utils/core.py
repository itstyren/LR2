import random
import numpy as np
from .._md_utils import utils


class Action:
    """
    Action class for Agent
    """

    def __init__(self):
        # dilemma strategy action (0-coop, 1-defect)
        self.s = None
        # reputation assessment to neighbour (right, bottom, left, top)
        self.ra = None


class Agent:
    """
    Properties of agent entities
    """

    def __init__(self, args):
        # name & index
        self.name = ""
        self.index = 0
        # dilemma action and reputation assignment
        self.action = Action()

        if args.network_type == "lattice":
            neighbor_count = 4 
        elif args.network_type == "honeycomb":
            neighbor_count = 3
        elif args.network_type == "moore":
            neighbor_count = 8
        elif args.network_type == "mixed":
            neighbor_count = args.mixed_num

        self.reward = 0.0
        self.reward_list = np.zeros(neighbor_count, dtype=np.float32)
        # Index of agnet's neighbour (right, bottom, left, top)
        self.neighbours = np.array([], dtype=np.int32)
        # reputaion assigned by neighbour agents (right, bottom, left, top)
        self.assigned_reputation = np.zeros(neighbor_count, dtype=np.float32)
        # current reputation
        self.reputation = 0.0
        self.consider_repu_reward=True if random.random() < args.repu_reward_fraction else False


class World:
    """
    Multi-agent world as Lattice Network
    """

    def __init__(self, initial_ratio, dilemma_parameter, repu_alpha, norm_type):
        """
        list of agents and payoff matrix
        """
        # Payoff matrix for the dilemma
        self.initial_ratio = initial_ratio

        self.agents = []
        self.payoff_matrix = dilemma_parameter
        self.stra_repu = [0.0, 0.0]
        self.repu_alpha = repu_alpha
        self.norm_type = norm_type

    def initialize(self, network_type,mixed_num):
        """
        Initialize the world
        """
        if network_type == "lattice":
            # Set the neighbor indices for lattice agents
            utils.gen_lattice_neighbours(self.agents)
        elif network_type == "moore":
            utils.gen_moore_neighbours(self.agents)
        elif network_type == "honeycomb":
            utils.gen_honeycomb_neighbours(self.agents)
        elif network_type == "mixed":
            # if mixed_num==1:
            utils.gen_random_reciprocal_neighbours(self.agents,mixed_num)
            # elif mixed_num==4:
            #     utils.gen_lattice_neighbours(self.agents)
                
        # Generate the pytorch-geometric graph structure for the agent (2-hop neighbours)
        # utils.gen_neighbour_graph_structure(self.agents)
        # breakpoint()

    def step(self):
        """
        Update state of the world
        """
        pass

    def update_reputation_assignment(self, init=False):
        """
        Update reputation assignment based on neighbour's acceessment

        :init: bool, default False, if True, initialize the reputation of the agent
        :return: Average reputaiion of the agent based on its strategy
        """
        # repu list for coop and defect
        average_repu_stra = [[], []]
        agent_repu_list = []
        for agent in self.agents:
            # iterate over the neighbour agents  (right, bottom, left, top)
            for n_idx_inner, n_idx_global in enumerate(agent.neighbours):
                # print(self.agents[n_idx_global].neighbours)
                # Locate the index of the current agent in its neighbour's list
                try:
                    # if agent.index ==0:
                    #     print(f'neighbour idex {n_idx_global}, his neighbour is {self.agents[n_idx_global].neighbours}')
                    #     print(f'agent idex {self.agents[n_idx_global].neighbours[0]}, his neighbour is {self.agents[self.agents[n_idx_global].neighbours[0]].neighbours}')
                    #     print(self.agents[n_idx_global].neighbours,agent.index)
                    if len(self.agents[n_idx_global].neighbours)>1:
                        agent_idx_in_neighbour = np.where(
                            np.array(self.agents[n_idx_global].neighbours) == agent.index
                        )[0][0]
                        # print(f'agent idex {agent.index}, his neighbour is {self.agents[n_idx_global].neighbours}')
                    else:
                        agent_idx_in_neighbour=0
                        # print(f'agent idex {agent.index}, his neighbour is {len(self.agents[n_idx_global].neighbours)}')
                except IndexError:
                    print(
                        f"Warning: Agent {agent.index} not found in neighbours of Agent {n_idx_global}"
                    )
                    continue  # Skip this iteration if the index is not found
                if self.norm_type == None:
                    # useing assigned reputation
                    agent.assigned_reputation[n_idx_inner] = self.agents[
                        n_idx_global].action.ra[agent_idx_in_neighbour]
                else:
                    # assign reputation by norm
                    agent.assigned_reputation[n_idx_inner] = utils.assign_reputation_by_norm(agent, self.agents[
                        n_idx_global], self.norm_type)
            if init:
                agent.reputation = np.mean(agent.assigned_reputation)
            else:
                agent.reputation = self.repu_alpha*agent.reputation + \
                    (1-self.repu_alpha)*np.mean(agent.assigned_reputation)

            # print(agent.assigned_reputation)

            agent_repu_list.append(agent.assigned_reputation)

            average_repu_stra[int(agent.action.s)].append(agent.reputation)

        repu_std = np.std(agent_repu_list, axis=1).mean()
        ave_strategy_repu = []
        for i, stra_repu in enumerate(average_repu_stra):
            if len(stra_repu) > 0:
                s_r = np.mean(stra_repu)
                ave_strategy_repu.append(s_r)
                self.stra_repu[i] = s_r
            else:
                ave_strategy_repu.append(self.stra_repu[i])

        return ave_strategy_repu, repu_std
