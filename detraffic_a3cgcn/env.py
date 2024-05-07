import gym
from gym.spaces import Discrete, Box
import sumo_rl

class SumoTrafficLightEnv(gym.Env):
    def __init__(self, num_light_states):
        self.env = sumo_rl.parallel_env(net_file='nets/RESCO/grid4x4/grid4x4.net.xml',
                  route_file='nets/RESCO/grid4x4/grid4x4_1.rou.xml',
                  use_gui=True,
                  num_seconds=3600)

        self.action_space = Discrete(n=num_light_states)

        self.obs = self.env.reset()
        self.num_intersections = len(self.obs)
        self.num_observations_per_intersection = len(next(iter(self.obs.values())))

        self.observation_space = Box(low=0.0, high=1.0,
                                     shape=(self.num_observations_per_intersection * self.num_intersections,))
        self.observation_space.from_jsonable()

    
    def step(self, action):
        light_states = self.convert_action_to_light_states(action)
        next_obs, reward, terminated, truncated, info = self.env.step(light_states)
        done = terminated or truncated
        return next_obs, reward, done, info
    

    def reset(self):
        self.env.reset()
        return self.get_observation_from_env()
    

    def convert_action_to_light_states(self, action):
        # Initialize light states to 0
        light_states = [0] * self.action_space.n
        # TODO: A conversion logic will be determined that might
        #       involve encoding schemes (bit manipulation)
        if action < self.action_space.n:
            light_states = list(action)

        return light_states
    

    def get_observation_from_env(self):
        # TODO: observation will be retrieved from the env
        #       depending on the state space
        pass
