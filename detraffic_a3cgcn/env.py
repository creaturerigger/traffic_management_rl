import gym
from gym.spaces import Discrete, Box
import sumo_rl
import numpy as np
import math

class SumoTrafficLightEnv(gym.Env):
    def __init__(self):
        self.env = sumo_rl.parallel_env(net_file='nets/RESCO/grid4x4/grid4x4.net.xml',
                  route_file='nets/RESCO/grid4x4/grid4x4_1.rou.xml',
                  use_gui=True,
                  num_seconds=3600)
        
        num_actions = self.env.action_space(self.env.agents[0]).n
        self.action_space = Discrete(n=len(num_actions))
        self.obs, self.info = self.env.reset()        
        self.num_observations_per_intersection = len(next(iter(self.obs.values())))
        self.num_agents = len(self.env.agents)
        self.observation_space = Box(low=0.0, high=1.0,
                                     shape=(self.num_agents, self.num_observations_per_intersection))
        

    
    def step(self, action):
        light_states = self.convert_action_to_light_states(action)
        next_obs, reward, terminated, truncated, info = self.env.step(light_states)
        done = terminated or truncated
        return next_obs, reward, done, info
    

    def reset(self):
        self.env.reset()
        return self.get_observation_from_env()
    

    def convert_action_to_light_states(self, action):
        """
        Converts a single action into light states for all 16 intersections.

        Args:
            action: An integer representing the chosen action from the action space.

        Returns:
            light_states: A list containing the light states for all intersections.
        """

        # Define the number of light states per intersection
        num_states_per_intersection = self.num_observations_per_intersection

        # Calculate the number of bits needed to represent each light state
        light_state_bits = int(math.ceil(np.log2(num_states_per_intersection)))

        # Initialize empty list for light states
        light_states = [0] * self.num_agents

        # Loop through each intersection
        for intersection_id in range(self.num_agents):
            # Extract the relevant bits for this intersection from the action value
            offset = intersection_id * light_state_bits
            intersection_action = (action >> offset) & ((1 << light_state_bits) - 1)

            # Convert the extracted bits to light state
            light_state = intersection_action % num_states_per_intersection

            # Update the light states list for this intersection
            light_states[intersection_id] = light_state

        return light_states
    

    def get_observation_from_env(self, action=None):
        """
        Retrieves the observation from the SUMO environment and processes it.

        Returns:
            observation: A list containing the processed observation for the agent.
        """

        # Get the raw observation from the environment
        raw_observations, _, _, _, _, _ = self.env.step(action)  

        # Pre-process the raw observation
        processed_observation = []
        for _, state in raw_observations.items():
            processed_observation.append(state)

        observation = np.array(processed_observation)
        observation = observation.reshape(self.num_agents, self.num_observations_per_intersection)

        # Min-max normalization
        
        observation = (observation - observation.min(axis=0)) / (observation.max(axis=0) - observation.min(axis=0))

        return observation