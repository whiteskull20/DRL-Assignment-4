import gymnasium as gym
import numpy as np
from train import Agent as StudentAgent

# Do not modify the input of the 'act' function and the '__init__' function. 
class Agent(object):
    """Agent that acts randomly."""
    def __init__(self):
        self.action_space = gym.spaces.Box(-1.0, 1.0, (21,), np.float64)
        self.agent = StudentAgent(67,21,self.action_space)
        self.agent.load('humanoid_sac')

    def act(self, observation):
        return self.agent.select_action(observation,evaluate=True)
