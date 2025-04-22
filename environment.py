import numpy as np, gymnasium as gym
from minigrid.wrappers import ImgObsWrapper

class ChannelFirst(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = {}
        self.observation_space = gym.spaces.Box(0, 255, shape=(3, 7, 7))

    def observation(self, observation):
        return np.swapaxes(observation, 2, 0)

class ScaledFloatFrame(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)

    def observation(self, observation):
        return np.array(observation).astype(np.float32)

class MinigridDoorKey6x6ImgObs(gym.Wrapper):
    def __init__(self):
        env = gym.make('MiniGrid-DoorKey-6x6-v0')
        env = ScaledFloatFrame(ChannelFirst(ImgObsWrapper(env)))
        super().__init__(env)