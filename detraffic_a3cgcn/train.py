import torch
from torch.utils.tensorboard import SummaryWriter

from env import SumoTrafficLightEnv
from a3c import A3C
from utils import (
    get_adjacency_matrix_grid, 
    get_adjacency_matrix_city,
    sum_reward
    )



DISCOUNT_FACTOR = 0.99
LEARNING_RATE = 0.001
NUM_EPOCHS = 1000



def main():
    env = SumoTrafficLightEnv()
    xml_file_ingolstadt = 'nets/RESCO/ingolstadt21/ingolstadt21.net.xml'
    adj_matrix = get_adjacency_matrix_city(xml_file_ingolstadt)
    NUM_AGENTS = env.num_agents    
    ACTION_SIZE_DICT = {agent: env.env.action_space(agent).n for agent in env.env.agents}
    agent = A3C(NUM_AGENTS, env.num_observations_per_intersection, ACTION_SIZE_DICT, 16, DISCOUNT_FACTOR, LEARNING_RATE)

    writer = SummaryWriter()

    mask = torch.Tensor(adj_matrix)

    # Each epoch is one full SUMO episode. The agent updates online,
    # once per simulation step, from the latest transition.
    for epoch in range(NUM_EPOCHS):
        state = env.reset()
        total_reward = {agt: 0.0 for agt in env.env.agents}
        done = False

        while not done:
            actions = agent.choose_action(state, mask)

            next_state, reward, dones, _ = env.get_observation_from_env(action=actions)

            total_reward = sum_reward(total_reward, reward)

            agent.learn(state, actions, reward, next_state, dones, mask)

            state = next_state
            done = all(dones.values())

        print(f"Epoch: {epoch+1} of {NUM_EPOCHS}, Reward: {total_reward}")

        writer.add_scalars('total_reward', total_reward, epoch)
    writer.close()
    torch.save(agent.actor_network.state_dict(), "a3c_actor.pth")
    torch.save(agent.critic_network.state_dict(), "a3c_critic.pth")

if __name__ == "__main__":
    main()