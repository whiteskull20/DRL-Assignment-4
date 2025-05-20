import gymnasium as gym
import numpy as np
from train import Agent as StudentAgent

# Do not modify the input of the 'act' function and the '__init__' function. 
s_agent = StudentAgent(67,21,None)
s_agent.load('humanoid_sac')
class Agent(object):
    """Agent that acts randomly."""
    def __init__(self):
        self.action_space = gym.spaces.Box(-1.0, 1.0, (21,), np.float64)

    def act(self, observation):
        return self.agent.select_action(observation,evaluate=True)
