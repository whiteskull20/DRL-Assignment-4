import gymnasium as gym
import numpy as np
from train import Agent as StudentAgent
import torch
# Do not modify the input of the 'act' function and the '__init__' function. 
class Agent(object):
    """Agent that acts randomly."""
    def __init__(self):
        # Pendulum-v1 has a Box action space with shape (1,)
        # Actions are in the range [-2.0, 2.0]
        self.action_space = gym.spaces.Box(-2.0, 2.0, (1,), np.float32)
        self.agent = StudentAgent()
        self.agent.q_net.load_state_dict(torch.load('pendulum_model.pth'))

    def act(self, observation):
        return [self.agent.select_action(observation, deterministic=True)[0]]