import torch
from torch.utils.tensorboard import SummaryWriter

from env import SumoTrafficLightEnv
from gcn import GCNSubnetwork
from a3c import A3C
from utils import get_adjacency_matrix



DISCOUNT_FACTOR = 0.99
LEARNING_RATE = 0.001
NUM_EPOCHS = 1000

def main():
    env = SumoTrafficLightEnv()
    adj_matrix = get_adjacency_matrix(env)
    NUM_AGENTS = env.num_agents
    STATE_SIZE = env.num_observations_per_intersection
    ACTION_SIZE = env.num_actions
# TODO: Number of features and nodes will be changed
    gcn_subnetwork = GCNSubnetwork(num_features=, num_nodes=NUM_AGENTS)
    agent = A3C(STATE_SIZE, ACTION_SIZE, DISCOUNT_FACTOR, LEARNING_RATE)

    writer = SummaryWriter()

    for epoch in range(NUM_EPOCHS):
        actions = {agent: env.env.action_space(agent).sample() for agent in env.env.agents}
        state = env.get_observation_from_env(action=actions)
        done = False
        total_reward = 0
        while not done:
            # Get current features
            gcn_features = gcn_subnetwork.forward(torch.tensor(state, dtype=torch.float), adj_matrix)

            # Concatanate gcn features with other state elements
            state_with_gcn = torch.cat([gcn_features, torch.tensor(state, dtype=torch.float)], dim=1)

            action = agent.choose_action(state_with_gcn)

            # Take action
            next_state, reward, done, info = env.step(action)
            total_reward += reward

            agent.learn(state_with_gcn, [action], [reward], [next_state], [done])

            state = next_state

        print(f"Epoch: {epoch+1} of {NUM_EPOCHS}, Reward: {total_reward}")
        writer.add_scalar("Total Reward", total_reward, epoch)

    torch.save(agent.actor_network.state_dict(), "a3c_actor.pth")
    torch.save(agent.critic_network.state_dict(), "a3c_critic.pth")


if __name__ == "__main__":
    main()
