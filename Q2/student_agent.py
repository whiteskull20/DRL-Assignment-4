import gymnasium
import numpy as np
from train import Agent as StudentAgent
import torch

# Do not modify the input of the 'act' function and the '__init__' function. 
class Agent(object):
    """Agent that acts randomly."""
    def __init__(self):
        self.action_space = gymnasium.spaces.Box(-1.0, 1.0, (1,), np.float64)
        self.agent = StudentAgent()
        self.agent.q_net.load_state_dict(torch.load('cart_model.pth'))

    def act(self, observation):
        return [self.agent.select_action(observation, deterministic=True)[0]]
