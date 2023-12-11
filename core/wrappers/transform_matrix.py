import gym
import numpy as np


class transformation_wrapper(gym.Wrapper):
    """
    Applies a given transformation
    """
    def __init__(self, env: gym, 
                 transformation_matrix: np.matrix = np.matrix('1,0,1,0,0;0,1,1,0,1;1,0,1,0,0;0,0,0,1,1'),
                 transformation_function = None): # transformation_function overwrites transform function
        super(transformation_wrapper, self).__init__(env)
        if transformation_function is not None:
            self.transform = transformation_function
        
        self.transform_matrix = transformation_matrix
    
    @staticmethod
    def transform(obs, matrix):
        return obs@matrix
    
    def step(self, action_axes, action_magnitude):
        obs, reward, truncated, done, info  = self.env.step(action_axes, action_magnitude)
        info["original_obs"] = obs
        obs = self.transform(obs, self.transform_matrix)
        return obs, reward, truncated, done, info
