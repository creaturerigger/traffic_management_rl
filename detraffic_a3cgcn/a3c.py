import torch
from torch import nn
from torch.distributions import Categorical
from torch.optim import Adam
import numpy as np
from gcn import Encoder, AttModel

class ActorNetwork(nn.Module):
    def __init__(self, num_agent: int, state_size: int, actions_dict: dict, hidden_dim: int) -> None:
        super(ActorNetwork, self).__init__()
        
        # ActorNetwork definition
        self.num_agent = num_agent
        self.encoder = Encoder(state_size, hidden_dim)
        self.att = AttModel(self.num_agent, hidden_dim, hidden_dim, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.output_layers = nn.ModuleDict({
            agent_id: nn.Linear(32, action_size) 
            for agent_id, action_size in actions_dict.items()
        })


    def forward(self, state: np.ndarray, mask: torch.Tensor):
        x = self.encoder(state)
        h, _ = self.att(x, mask)
        x = torch.relu(self.fc1(h))
        x = torch.relu(self.fc2(x))
        dists = {}

        for agent_id, output_layer in self.output_layers.items():
            probs = torch.softmax(output_layer(x), dim=1)            
            dist = Categorical(probs)
            dists[agent_id] = dist
        return dists
    

class CriticNetwork(nn.Module):
    def __init__(self, num_agent: int, state_size: int, hidden_dim: int):
        super(CriticNetwork, self).__init__()

        self.num_agent = num_agent
        self.encoder = Encoder(state_size, hidden_dim)
        self.att = AttModel(self.num_agent, hidden_dim, hidden_dim, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.output = nn.Linear(32, 1)


    def forward(self, state: torch.Tensor, mask: torch.Tensor):
        x = self.encoder(state)
        h, _ = self.att(x, mask)
        x = torch.relu(self.fc1(h))
        x = torch.relu(self.fc2(x))
        value = self.output(x)
        return value
    

class A3C(object):
    def __init__(self, num_agent: int, state_size: int, actions_dict: dict,  hidden_dim: int, discount_factor: float, learning_rate: float):
        self.num_agent = num_agent
        self.actor_network = ActorNetwork(num_agent, state_size, actions_dict, hidden_dim)
        self.critic_network = CriticNetwork(num_agent, state_size, hidden_dim)
        self.actor_optimizer = Adam(self.actor_network.parameters(), lr=learning_rate)
        self.critic_optimizer = Adam(self.critic_network.parameters(), lr=learning_rate)
        self.discount_factor = discount_factor


    def choose_action(self, state, mask):
        state = torch.tensor(state, dtype=torch.float)
        dist_dict = self.actor_network(state, mask)
        action_matrix = torch.stack([dist.sample().squeeze(0) for dist in dist_dict.values()], dim=1)
        actions = {}
        for i, agent_id in enumerate(dist_dict.keys()):
            # Get the most frequent action in the i-th column
            most_frequent_action = torch.mode(action_matrix[:, i])[0].item()
            actions[agent_id] = most_frequent_action

        return actions
    

    def learn(self, states, actions, rewards, next_states, dones, masks):
        rewards = torch.tensor(list(rewards.values()))
        dones = torch.tensor(list(dones.values()), dtype=torch.int)
        G_t = torch.zeros_like(torch.Tensor(rewards))
        
        G_t = rewards * dones.bool()
        dones = ~dones.bool()
        
        V_prime = self.discount_factor * self.critic_network(next_states, masks)
        V_prime = V_prime.permute(0, 2, 1)
        V_prime = torch.squeeze(torch.squeeze(V_prime, dim=0), dim=0)
        G_t += rewards * dones.bool() + V_prime
        
        states = torch.tensor(states, dtype=torch.float)
        masks = torch.tensor(masks, dtype=torch.float)
        actions_tensor = torch.tensor([actions[agent_id] for agent_id in actions.keys()], dtype=torch.int64)
        #actions = torch.tensor(list(actions.values()), dtype=torch.int64)
        rewards = torch.tensor(rewards, dtype=torch.float)

        V_s_t = self.critic_network(states, masks)
        critic_loss = torch.nn.functional.mse_loss(V_s_t, G_t.detach())

        advantages = G_t - V_s_t

        policy_probs = self.actor_network(states, masks)
        print("Policy probs are: ", policy_probs)
        log_probs = torch.cat([policy_probs[agent_id].log_prob(actions_tensor[i]).unsqueeze(0) for i, agent_id in enumerate(actions.keys())])
        actor_loss = -torch.mean(log_probs * advantages.detach())

        # Update networks
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()



















'''import torch
from torch import nn
from torch.distributions import Categorical
from torch.optim import Adam
import numpy as np

class ActorNetwork(nn.Module):
    def __init__(self, state_size: int, action_size: int) -> None:
        super(ActorNetwork, self).__init__()

        # ActorNetwork definition
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.output = nn.Linear(32, action_size)


    def forward(self, state: np.ndarray):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        probs = torch.softmax(self.output(x), dim=1)
        dist = Categorical(probs)
        return dist
    

class CriticNetwork(nn.Module):
    def __init__(self, state_size: int):
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
        state = torch.tensor(state, dtype=torch.float)
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
            

            V_s_t = self.critic_network(states)
            critic_loss = torch.nn.functional.mse_loss(V_s_t, G_t.detach())

            advantages = G_t - V_s_t

            policy_probs = self.actor_network(states)
            log_probs = policy_probs.log_prob(actions)
            actor_loss = -torch.mean(log_probs * advantages.detach())

            # Update networks
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            '''