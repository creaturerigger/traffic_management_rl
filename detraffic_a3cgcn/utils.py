import numpy as np
import sumo_rl


def get_adjacency_matrix(env):
    # TODO: The adjacency matrix should represent the connections
    #       between the agents
    NUM_AGENTS = env.num_agents
    adj_matrix = np.ones((NUM_AGENTS, NUM_AGENTS))
    return adj_matrix