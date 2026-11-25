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
        self.action_size_dict = {agt: self.env.action_space(agt).n
                                 for agt in self.env.possible_agents}
        self.obs, self.info = self.env.reset()
        self.num_observations_per_intersection = max(len(o) for o in
                                                     self.obs.values())
        self.num_agents = len(self.env.possible_agents)

    def step(self, action):
        next_obs, reward, terminated, truncated, info = self.env.step(action)
        dones = {agent: terminated[agent] or truncated[agent]
                 for agent in terminated}
        return next_obs, reward, dones, info

    def reset(self, normalization=False):
        """
        Starts a new episode and returns the first processed observation.
        """
        raw_observations, _ = self.env.reset()
        return self._process_observation(raw_observations, normalization)

    def get_observation_from_env(self, action=None, normalization=False):
        """
        Steps the SUMO environment with the given actions and processes
        the observation.

        Returns:
            observation: Tensor of shape (1, num_agents, obs_size),
                         zero-padded.
            reward: Dict of per-agent rewards.
            dones: Dict of per-agent done flags (terminated or truncated).
            info: Dict returned by the environment.
        """
        raw_observations, reward, terminated, \
            truncated, info = self.env.step(action)

        dones = {agent: terminated[agent] or truncated[agent]
                 for agent in terminated}
        return self._process_observation(raw_observations, normalization),
        reward, dones, info

    def _process_observation(self, raw_observations, normalization=False):
        eps = 1e-12
        # Pre-process the raw observation
        processed_observation = []
        for _, state in raw_observations.items():
            processed_observation.append(list(state))
        max_len = max(len(s) for s in processed_observation)
        self.num_observations_per_intersection = max_len
        processed_observation = [self.pad_after(s, max_len)
                                 for s in processed_observation if len(s)]

        observation = np.array(processed_observation)
        # observation = observation.reshape(self.num_agents, max_len)

        # Min-max normalization
        if normalization:
            observation = (observation - observation.min(axis=0)) / \
                ((observation.max(axis=0) - observation.min(axis=0)) + eps)
        observation = np.expand_dims(observation, axis=0)
        return torch.Tensor(observation)

    def pad_after(self, state, max_length):
        if len(state) == max_length:
            return state
        pad_length = max_length - len(state)
        pad = [0] * pad_length
        return state + pad
