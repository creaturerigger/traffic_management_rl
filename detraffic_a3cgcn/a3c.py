import torch
from torch import nn
from torch.distributions import Categorical
from torch.optim import Adam

class ActorNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(ActorNetwork, self).__init__()

        # ActorNetwork definition
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.output = nn.Linear(32, action_size)


    def forward(self, state):
        x = torch.reul(self.fc1(state))
        x = torch.relu(self.fc2(x))
        probs = torch.softmax(self.output(x), dim=1)
        dist = Categorical(probs)
        return dist
    

class CriticNetwork(nn.Module):
    def __init__(self, state_size):
        super(CriticNetwork, self).__init__()

        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.output = nn.Linear(32, 1)


    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        value = self.output(x)
        return value
    

class A3C(object):
    def __init__(self, state_size, action_size, discount_factor, learning_rate):
        self.actor_network = ActorNetwork(state_size, action_size)
        self.critic_network = CriticNetwork(state_size)
        self.actor_optimizer = Adam(self.actor_network.parameters(), lr=learning_rate)
        self.critic_optimizer = Adam(self.critic_network.parameters(), lr=learning_rate)
        self.discount_factor = discount_factor


    def choose_action(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        dist = self.actor_network(state)
        action = dist.sample()
        return action.item()
    

    def learn(self, states, actions, rewards, next_states, dones):
        G_t = torch.zeros_like(rewards)
        for i in reversed(range(len(rewards))):
            if dones[i]:
                G_t[i] = rewards[i]
            else:
                G_t[i] = rewards[i] + self.discount_factor * self.critic_network(torch.tensor(next_states[i], dtype=torch.float))
            

            V(s_t) = self.critic_network(states)
            