import numpy as np
from .._md_utils.scenario import BaseScenario
from .._md_utils.core import World, Agent
from .._md_utils.lattice_env import LatticeEnv
from .._md_utils import utils
import re

class raw_env(LatticeEnv):
    """
    A Matirx game environment has gym API for soical dilemma
    """

    def __init__(self, args, max_cycles=1, render_mode=None):
        scenario = Scenario()
        world = scenario.make_world(args)
        LatticeEnv.__init__(
            self,
            scenario=scenario,
            world=world,
            max_cycles=max_cycles,
            render_mode=render_mode,
            args=args,
        )
        self.metadata["name"] = "lattice_rl_v0"


env = utils.make_env(raw_env)
parallel_env = utils.parallel_wrapper_fn(env)


class Scenario(BaseScenario):
    def __init__(self) -> None:
        super().__init__()

    def make_world(self, args):
        self.env_dim = args.env_dim
        # if the reputation will not be aligned with the agent action in the begin of every episode
        self.random_repu=args.random_repu

        self.network_type=args.network_type
        self.mixed_num=args.mixed_num
        
        if args.network_type == "lattice":
            self.neighbor_count = 4 
        elif args.network_type == "honeycomb":
            self.neighbor_count = 3
        elif args.network_type == "moore":
            self.neighbor_count = 8
        elif args.network_type == "mixed":
            self.neighbor_count = args.mixed_num

        agent_num = args.env_dim**2

        # Payoff matrix for the dilemma
        dilemma_R = 1
        dilemma_P = 0

        dilemma_S=args.dilemma_S

        dilemma_parameter = np.array(
            [[dilemma_R, dilemma_S], [args.dilemma_T, dilemma_P]]
        )

        match = re.match(r'PPO_(\w+)', args.algorithm_name)
        norm_type=None
        if match:
            norm_type=match.group(1)

        # create world
        world = World(np.array(args.initial_ratio), dilemma_parameter,args.repu_alpha,norm_type)
        # create agents
        world.agents = [Agent(args) for i in range(agent_num)]

        for i, agent in enumerate(world.agents):
            agent.name = f"agent_{i}"
            agent.index = i


        # initialize the world (neighbour index, graph structure)
        world.initialize(args.network_type,args.mixed_num)


        return world

    def reset_world(self, world, seed):
        """
        Randomly initialize the world and agnets strategy
        0 for coop and 1 for defect
        """


        num_ones = int(len(world.agents) * world.initial_ratio.ravel()[0])
        num_zeros = len(world.agents) - num_ones
        dillema_strategy = np.array([1] * num_ones + [0] * num_zeros)
        np.random.shuffle(dillema_strategy)

        for agent,dilemma_action in zip(world.agents,dillema_strategy):
            agent.action.prevoius_s = agent.action.s.copy() if agent.action.s is not None else None
            agent.action.s = int(
                dilemma_action)
        
        

        for i,agent in enumerate(world.agents):
            if self.random_repu:
                agent.action.ra=np.round(np.random.rand(self.neighbor_count), 3).astype(np.float32)
            else:
                agent.action.ra=[int(1-world.agents[agent.neighbours[i]].action.s) for i in range(self.neighbor_count)]


        world.update_reputation_assignment(init=True)


    def observation(self, agent, world):
        """
        Get observation information for the current agent based on network type.
        """
        # Define neighborhood indices based on network type
        if self.network_type == "lattice":
            neighbor_count = 4
        elif self.network_type == "honeycomb":
            neighbor_count = 3
        elif self.network_type == "moore":
            neighbor_count = 8
        elif self.network_type == "mixed":
            neighbor_count = self.mixed_num
        else:
            raise ValueError(f"Unsupported network type: {self.network_type}")


        neighbour_dilemma_action=[]
        neighbour_reputation=[]
        neighbour_reputation_access=[]

        # Iterate over the neighbors (based on network type)
        for first_hoop_idx, neighbor_idx in enumerate(agent.neighbours):
            neighbour_dilemma_action.append(world.agents[neighbor_idx].action.s)
            neighbour_reputation.append(world.agents[neighbor_idx].reputation)
            neighbour_reputation_access.append(world.agents[neighbor_idx].assigned_reputation)

        # Assemble the observation dictionary
        obs = {
            'n_d': neighbour_dilemma_action,
            'n_r_a': neighbour_reputation_access,
            'n_r': neighbour_reputation,
            's_r': [agent.reputation] * neighbor_count  # Self-reputation replicated for all neighbors
        }

        return obs

    
    def reward(self, agent, world):
        """
        calculate current reward by matrix, play with all neighbours
        """
        reward = 0.0
        reward_list=[]

        for idx, neighbout_idx in enumerate(agent.neighbours):
            current_reward = float(world.payoff_matrix[agent.action.s][world.agents[neighbout_idx].action.s])
            reward_list.append(current_reward)
            reward += current_reward

        return reward,reward_list