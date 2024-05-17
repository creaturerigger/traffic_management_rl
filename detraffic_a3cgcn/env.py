from multiprocessing import process
import gym
from gym.spaces import Discrete, Box
from sumo_rl import ingolstadt21
import numpy as np
import torch

class SumoTrafficLightEnv(gym.Env):
    def __init__(self):
        super(SumoTrafficLightEnv, self).__init__()
        self.env = ingolstadt21(
                  use_gui=True,
                  yellow_time=2,
                  render_mode='human')
        self.action_size_dict = {agt: self.env.action_space(agt).n for agt in self.env.possible_agents}
        self.obs, self.info = self.env.reset()        
        self.num_observations_per_intersection = len(next(iter(self.obs.values())))
        self.num_agents = len(self.env.possible_agents)
        
        
    def step(self, action):
        next_obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return next_obs, reward, done, info
    

    def reset(self):
        self.env.reset()
        return self.get_observation_from_env()
    

    def get_observation_from_env(self, action=None, normalization=False):
        """
        Retrieves the observation from the SUMO environment and normalizes it.

        Returns:
            observation: A list containing the processed observation for the agent.
        """
        
        # Get the raw observation from the environment
        raw_observations, reward, terminated, truncated, info = self.env.step(action)  
        eps = 1e-12
        # Pre-process the raw observation
        processed_observation = []
        for _, state in raw_observations.items():
            processed_observation.append(list(state))
        max_len = max(len(s) for s in processed_observation)
        self.num_observations_per_intersection = max_len
        processed_observation = [self.pad_after(s, max_len) for s in processed_observation if len(s)]
        
        observation = np.array(processed_observation)        
        # observation = observation.reshape(self.num_agents, max_len)

        # Min-max normalization
        if normalization:
            observation = (observation - observation.min(axis=0)) / ((observation.max(axis=0) - observation.min(axis=0)) + eps)
        observation = np.expand_dims(observation, axis=0)
        return torch.Tensor(observation), reward, terminated or truncated, info
    

    def pad_after(self, state, max_length):
        if len(state) == max_length:
            return state
        pad_length = max_length - len(state)
        pad = [0] * pad_length
        return state + pad
