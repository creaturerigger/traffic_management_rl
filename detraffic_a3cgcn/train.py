from networkx import adjacency_matrix
import torch
from torch.utils.tensorboard import SummaryWriter

from env import SumoTrafficLightEnv
from gcn import DGN
from a3c import A3C
from utils import get_adjacency_matrix_grid, get_adjacency_matrix_city



DISCOUNT_FACTOR = 0.99
LEARNING_RATE = 0.001
NUM_EPOCHS = 1000

def main():
    env = SumoTrafficLightEnv()
    # xml_file = "nets/RESCO/grid4x4/grid4x4.net.xml"
    xml_file_ingolstadt = 'nets/RESCO/ingolstadt21/ingolstadt21.net.xml'
    adj_matrix = get_adjacency_matrix_city(xml_file_ingolstadt)
    print(adj_matrix)
    NUM_AGENTS = env.num_agents
    STATE_SIZE = env.num_observations_per_intersection
    ACTION_SIZE = env.num_actions
    # TODO: Number of features and nodes will be changed
    dgn_network = DGN(NUM_AGENTS, 33, 16, ACTION_SIZE)
    print("number of observation per intersection: ", env.num_observations_per_intersection)
    agent = A3C(NUM_AGENTS, 21, ACTION_SIZE, 16, DISCOUNT_FACTOR, LEARNING_RATE)

    writer = SummaryWriter()

    for epoch in range(NUM_EPOCHS):
        actions = {agent: env.env.action_space(agent).sample() for agent in env.env.agents}
        state = env.get_observation_from_env(action=actions)
        print("State shape is: ", state.shape)
        done = False
        total_reward = 0
        while not done:
            # Get current features
            _, gcn_features = dgn_network(torch.tensor(state, dtype=torch.float), torch.Tensor(adj_matrix))
            
            # Concatanate gcn features with other state elements
            state_with_gcn = torch.cat([gcn_features, torch.tensor(state, dtype=torch.float).permute(0, 2, 1)], dim=1)
            print("State with gcn is: ", state_with_gcn.shape)
            action = agent.choose_action(state_with_gcn, torch.Tensor(adj_matrix))

            # Take action
            next_state, reward, done, info = env.step(action)
            total_reward += reward

            agent.learn(state_with_gcn, [action], [reward], [next_state], [done], [adj_matrix])

            state = next_state

        print(f"Epoch: {epoch+1} of {NUM_EPOCHS}, Reward: {total_reward}")
        writer.add_scalar("Total Reward", total_reward, epoch)

    torch.save(agent.actor_network.state_dict(), "a3c_actor.pth")
    torch.save(agent.critic_network.state_dict(), "a3c_critic.pth")


if __name__ == "__main__":
    main()

'''
a = [[0 1 0 0 1 0 0 0 0 0 0 0 0 1 0 0 0 1 0 0 0],
        [1 0 0 0 1 0 0 0 0 0 0 0 0 1 0 0 0 1 0 0 0],
        [0 0 0 1 1 1 1 1 1 1 1 1 0 1 1 0 1 0 0 1 1],
        [0 0 1 0 1 1 1 1 1 1 1 1 0 1 1 0 1 0 0 1 1],
        [1 1 1 1 0 1 1 1 1 1 1 1 0 1 1 0 1 1 0 1 1],
        [0 0 1 1 1 0 1 1 1 1 1 1 0 1 1 0 1 0 0 1 1],
        [0 0 1 1 1 1 0 1 1 1 1 1 0 1 1 0 1 0 0 1 1],
        [0 0 1 1 1 1 1 0 1 1 1 1 0 1 1 0 1 0 0 1 1],
        [0 0 1 1 1 1 1 1 0 1 1 1 0 1 1 0 1 0 0 1 1],
        [0 0 1 1 1 1 1 1 1 0 1 1 0 1 1 0 1 0 0 1 1],
        [0 0 1 1 1 1 1 1 1 1 0 1 0 1 1 0 1 0 0 1 1],
        [0 0 1 1 1 1 1 1 1 1 1 0 0 1 1 0 1 0 0 1 1],
        [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0],
        [1 1 1 1 1 1 1 1 1 1 1 1 0 0 1 0 1 0 0 1 1],
        [0 0 1 1 1 1 1 1 1 1 1 1 0 1 0 0 1 0 0 1 1],
        [0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 0 0 1 1],
        [0 0 1 1 1 1 1 1 1 1 1 1 0 1 1 1 0 0 0 1 1],
        [1 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0],
        [0 0 1 1 1 1 1 1 1 1 1 1 0 1 1 0 1 0 0 1 1],
        [0 0 1 1 1 1 1 1 1 1 1 1 0 1 1 0 1 0 1 0 1],
        [0 0 1 1 1 1 1 1 1 1 1 1 0 1 1 0 1 0 0 1 0]]


'''



