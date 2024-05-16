from networkx import adjacency_matrix
import torch
from torch.utils.tensorboard import SummaryWriter

from env import SumoTrafficLightEnv
from gcn import DGN
from a3c import A3C
from utils import (
    get_adjacency_matrix_grid, 
    get_adjacency_matrix_city,
    sum_reward
    )



DISCOUNT_FACTOR = 0.99
LEARNING_RATE = 0.001
NUM_EPOCHS = 1000


writer = SummaryWriter()


DISCOUNT_FACTOR = 0.99
LEARNING_RATE = 0.001
NUM_EPOCHS = 1000

def main():
    env = SumoTrafficLightEnv()
    xml_file_ingolstadt = 'nets/RESCO/ingolstadt21/ingolstadt21.net.xml'
    adj_matrix = get_adjacency_matrix_city(xml_file_ingolstadt)
    NUM_AGENTS = env.num_agents
    STATE_SIZE = env.num_observations_per_intersection
    ACTION_SIZE = env.num_actions

    agent = A3C(NUM_AGENTS, 33, ACTION_SIZE, 16, DISCOUNT_FACTOR, LEARNING_RATE)

    writer = SummaryWriter()

    for epoch in range(NUM_EPOCHS):
        actions = {agent: env.env.action_space(agent).sample() for agent in env.env.agents}
        # print("Initial actions are: ", actions)
        state, reward, done, _ = env.get_observation_from_env(action=actions)
        done = False
        total_reward = dict(map(lambda agt, val: (agt, val), env.env.agents, [0.0] * len(env.env.agents)))
        
        while not done:
            mask = torch.Tensor(adj_matrix)
            action = agent.choose_action(state, mask)

            actions = dict(map(lambda agt, act: (agt, act), env.env.agents, action.numpy()[0]))
            
            next_state, reward, done, _ = env.get_observation_from_env(action=actions)

            total_reward = sum_reward(total_reward, reward)
            
            agent.learn(state, actions, reward, next_state, done, mask)

            state = next_state

        print(f"Epoch: {epoch+1} of {NUM_EPOCHS}, Reward: {total_reward}")
        
        writer.add_scalars('total_reward', total_reward, epoch)
    writer.close()
    torch.save(agent.actor_network.state_dict(), "a3c_actor.pth")
    torch.save(agent.critic_network.state_dict(), "a3c_critic.pth")

if __name__ == "__main__":
    main()