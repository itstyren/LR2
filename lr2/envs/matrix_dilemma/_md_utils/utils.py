from typing import Callable
import numpy as np
from pettingzoo.utils.conversions import aec_to_parallel_wrapper
import networkx as nx

def make_env(raw_env):
    def env(args, **kwargs):
        env = raw_env(args, **kwargs)
        return env

    return env


def parallel_wrapper_fn(env_fn: Callable) -> Callable:
    def par_fn(args, **kwargs):
        env = env_fn(args, **kwargs)
        env = aec_to_parallel_wrapper(env)
        return env

    return par_fn

def gen_random_reciprocal_neighbours(agents_list, num_neighbours=1):
    """
    Assign exactly `degree` neighbors to each agent using a random regular graph.
    """
    num_agents = len(agents_list)

    # Check if the degree is valid
    if num_agents * num_neighbours % 2 != 0:
        raise ValueError("The number of agents and degree must allow for a valid graph (num_agents * degree must be even).")
    if num_neighbours >= num_agents:
        raise ValueError("Degree must be less than the number of agents.")

    # Generate a random d-regular graph
    random_graph = nx.random_regular_graph(num_neighbours, num_agents)

    # Assign neighbors to agents
    for agent_index in range(num_agents):
        agents_list[agent_index].neighbours = list(random_graph.neighbors(agent_index))



def gen_lattice_neighbours(agents_list):
    """
    Set the neighbor indices for lattice agents
    order is right,bottom,left,top
    """
    DIM_SIZE = int(len(agents_list) ** 0.5)

    # Define relative positions of neighbors
    neighbor_offsets = [(1, 0), (0, 1), (-1, 0), (0, -1)]

    for i in range(DIM_SIZE):
        for j in range(DIM_SIZE):
            agent_index = DIM_SIZE * j + i

            for offset_x, offset_y in neighbor_offsets:
                neighbor_i = (i + offset_x) % DIM_SIZE
                neighbor_j = (j + offset_y) % DIM_SIZE
                neighbor_index = DIM_SIZE * neighbor_j + neighbor_i

                agents_list[agent_index].neighbours = np.append(
                    agents_list[agent_index].neighbours, neighbor_index
                )

def gen_moore_neighbours(agents_list):
    """
    Set the neighbor indices for lattice agents in a Moore neighborhood (8 neighbors).
    Includes diagonal neighbors as well as right, bottom, left, and top neighbors.
    """
    DIM_SIZE = int(len(agents_list) ** 0.5)

    # Define relative positions of neighbors (Moore neighborhood)
    neighbor_offsets = [
        (1, 0),  (0, 1),  (-1, 0),  (0, -1),  # Right, Bottom, Left, Top
        (1, 1),  (-1, 1), (-1, -1), (1, -1)   # Diagonals
    ]

    for i in range(DIM_SIZE):
        for j in range(DIM_SIZE):
            agent_index = DIM_SIZE * j + i

            for offset_x, offset_y in neighbor_offsets:
                neighbor_i = (i + offset_x) % DIM_SIZE
                neighbor_j = (j + offset_y) % DIM_SIZE
                neighbor_index = DIM_SIZE * neighbor_j + neighbor_i

                # Add the neighbor to the agent's list
                agents_list[agent_index].neighbours = np.append(
                    agents_list[agent_index].neighbours, neighbor_index
                ) 


def gen_honeycomb_neighbours(agents_list):
    """
    Set the neighbor indices for lattice agents in a honeycomb neighborhood.
    Honeycomb lattices have each node connected to 3 neighbors in a staggered hexagonal pattern.
    """
    DIM_SIZE = int(len(agents_list) ** 0.5)

    # Define relative positions of neighbors for the honeycomb lattice
    neighbor_offsets_even_row = [(1, 0), (-1, 0), (0, -1)]  # Neighbors for even rows
    neighbor_offsets_odd_row = [(1, 0), (-1, 0), (0, 1)]   # Neighbors for odd rows

    for i in range(DIM_SIZE):
        for j in range(DIM_SIZE):
            agent_index = DIM_SIZE * j + i

            # Choose neighbor offsets based on the row parity
            if j % 2 == 0:
                neighbor_offsets = neighbor_offsets_even_row
            else:
                neighbor_offsets = neighbor_offsets_odd_row

            for offset_x, offset_y in neighbor_offsets:
                neighbor_i = (i + offset_x) % DIM_SIZE
                neighbor_j = (j + offset_y) % DIM_SIZE
                neighbor_index = DIM_SIZE * neighbor_j + neighbor_i

                # Add the neighbor to the agent's list
                agents_list[agent_index].neighbours = np.append(
                    agents_list[agent_index].neighbours, neighbor_index
                )


def assign_reputation_by_norm(donor_agent, recipient_agent, norm_type):
    """
    Assign reputation based on the norm type
    """
    assign_reputation = None

    if norm_type == "SJ":
        # donnor choose coop
        if donor_agent.action.s == 0:
            if recipient_agent.reputation >= 0.5:
                assign_reputation = 1
            else:
                assign_reputation = 0
        # donnor choose defect
        else:
            if recipient_agent.reputation >= 0.5:
                assign_reputation = 0
            else:
                assign_reputation = 1
    elif norm_type == "SS":
        # donnor choose coop
        if donor_agent.action.s == 0:
            assign_reputation = 1
        # donnor choose defect
        else:
            if recipient_agent.reputation >= 0.5:
                assign_reputation = 0
            else:
                assign_reputation = 1
    elif norm_type == "SH":
        # donnor choose coop
        if donor_agent.action.s == 0 and recipient_agent.reputation >= 0.5:
            assign_reputation = 1
        # all other cases
        else:
            assign_reputation = 0
    elif norm_type == "IS":
        if donor_agent.action.s == 0:
            assign_reputation = 1
        else:
            assign_reputation = 0
    return assign_reputation

