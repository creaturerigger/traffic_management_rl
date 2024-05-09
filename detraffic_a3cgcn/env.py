import gym
from gym.spaces import Discrete, Box
import sumo_rl
import numpy as np

class SumoTrafficLightEnv(gym.Env):
    def __init__(self):
        print("init started: ")
        self.env = sumo_rl.parallel_env(net_file='nets/RESCO/ingolstadt21/ingolstadt21.net.xml',
                  route_file='nets/RESCO/ingolstadt21/ingolstadt21.rou.xml',
                  use_gui=True,
                  num_seconds=80000)
        observations, info = self.env.reset()
        print(dir(self.env))
        print("Agents are: ", self.env.agents)
        self.num_actions = self.env.action_space(self.env.agents[0]).n
        self.action_space = Discrete(n=self.num_actions)
        self.obs, self.info = self.env.reset()        
        self.num_observations_per_intersection = len(next(iter(self.obs.values())))
        self.num_agents = len(self.env.agents)
        self.observation_space = Box(low=0.0, high=1.0,
                                     shape=(self.num_agents, self.num_observations_per_intersection))
        print("init finished: ")

    
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
        raw_observations, _, _, _, _ = self.env.step(action)  
        eps = 1e-12
        # Pre-process the raw observation
        processed_observation = []
        for _, state in raw_observations.items():
            processed_observation.append(state)

        observation = np.array(processed_observation)
        observation = observation.reshape(self.num_agents, self.num_observations_per_intersection)

        # Min-max normalization
        if normalization:
            observation = (observation - observation.min(axis=0)) / ((observation.max(axis=0) - observation.min(axis=0)) + eps)

        return observation