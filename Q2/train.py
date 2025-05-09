import gymnasium as gym
import torch
import torch.nn as nn
import random
import numpy as np
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from collections import deque
from torch.distributions import Normal
from dmc import make_dmc_env
import numpy as np
device = ("cuda" if torch.cuda.is_available() else "cpu" )
def make_env():
	# Create environment with state observations
	env_name = "cartpole-balance"
	env = make_dmc_env(env_name, np.random.randint(0, 1000000), flatten=True, use_pixels=False)
	return env



class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        # Actor now outputs mean and log_std instead of discrete bins
        self.actor1 = nn.Sequential(
            nn.Linear(5, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.actor_mean = nn.Linear(256, 1)  # Mean for Gaussian policy
        self.actor_log_std = nn.Linear(256, 1)  # Log standard deviation (learnable)
        self.critic = nn.Sequential(
            nn.Linear(5, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)  # Single value function output
        )

    def forward(self, x):
        a = self.actor1(x)
        a = F.relu(a)
        mean = self.actor_mean(a).clamp(-1,1)
        log_std = self.actor_log_std(a).clamp(-5,2) - 3  # Keep log_std in a reasonable range
        std = log_std.exp()  # Convert log_std to std
        value = self.critic(x)
        return mean, std, value

class Agent:
    def __init__(self, gamma=0.99, policy_noise=0.1):
        self.policy_noise = policy_noise
        self.q_net = ActorCritic().to(device)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=3e-4)
        self.gamma = gamma
        self.eps = policy_noise

    def select_action(self, state, deterministic=False):
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
            mean, std, value = self.q_net.forward(state)
            action_dist = Normal(mean, std)
            if deterministic:
                action = mean  # Deterministic policy uses mean
            else:
                action = action_dist.sample()  # Sample from Gaussian distribution
            action = action.clamp(-1, 1)  # Ensure action stays in valid range
            return action.item(), value.item()
    def train(self, batch):
        # Unpacking batch data
        state, action, advantage, returns = zip(*batch)
        # Convert batch data to PyTorch tensors
        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            action = torch.FloatTensor(action).unsqueeze(1).to(device)
            advantage = torch.FloatTensor(advantage).unsqueeze(1).to(device)
            returns = torch.FloatTensor(returns).unsqueeze(1).to(device)
        # Forward pass to get policy outputs
        mean, std, current_value = self.q_net.forward(state)
        #print(advantage,returns,current_value)
        action_dist = Normal(mean, std)
        # Compute new log probabilities
        log_prob = action_dist.log_prob(action)

        # Compute ratio for PPO loss
        ratio = torch.exp(log_prob - log_prob.detach())
        
        # Surrogate objective
        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage
        p_loss = -torch.min(surr1, surr2).mean()

        # Critic loss using value estimation
        v_loss = nn.MSELoss()(current_value, returns)

        # Entropy loss for better exploration
        entropy = -action_dist.entropy().mean()

        # Final PPO loss
        loss = p_loss + 0.5 * v_loss + 0.01 * entropy
        # Gradient step
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=0.5)  # Prevent exploding gradients
        self.optimizer.step()
        return loss.item()
    def save(self, filename):
        torch.save(self.q_net.state_dict(), filename, _use_new_zipfile_serialization=False)
        print(f"Model saved to {filename}")

def compute_gae(rewards, values, dones, gamma=0.99, lambda_gae=0.95):
    advantages = np.zeros_like(rewards)
    returns = np.zeros_like(rewards)
    gae = 0

    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
        gae = delta + gamma * lambda_gae * (1 - dones[t]) * gae
        advantages[t] = gae
        returns[t] = gae + values[t]

    return advantages, returns

def train_ppo(agent, env, epochs=500, steps_per_epoch=2048, update_epochs=10):
    rewards = []
    for epoch in range(epochs):
        state, _ = env.reset()
        total_reward = 0
        state_buffer, action_buffer, reward_buffer, done_buffer, value_buffer = [], [], [], [], []
        while True:
            action, value = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step([action])
            total_reward += reward
            done = terminated or truncated
            state_buffer.append(state.copy())
            action_buffer.append(action)
            reward_buffer.append(reward)
            done_buffer.append(done)
            value_buffer.append(value)
            if done:
                state, _ = env.reset()
                rewards.append(total_reward)
                break
            else:
                state = next_state.copy()
        advantages, returns = compute_gae(
            np.array(reward_buffer),
            np.array(value_buffer + [0]),  # Append zero for final state
            np.array(done_buffer),
            gamma=0.99,
            lambda_gae=0.95
        )
        #print(advantages,returns)
        loss = 0
        for _ in range(update_epochs):
            loss += agent.train(
                list(zip(state_buffer, action_buffer, advantages, returns))
            )

        loss /= update_epochs
        print(f"Epoch {epoch + 1}/{epochs}, Average Reward: {np.mean(rewards[-10:]):.2f}, Std: {np.std(rewards[-10:]):.2f}, Loss: {loss}")
        if np.mean(rewards[-10:]) - np.std(rewards[-10:]) > 955:
            return
        if epoch + 1 % 100 == 0:
            agent.save("cart_model.pth")


if __name__ == "__main__":
    # Create the environment
    env = make_env()
    agent = Agent()
    train_ppo(agent, env, epochs=20000, steps_per_epoch=2048, update_epochs=1)
    env.close()
    # Save the model
    agent.save("cart_model.pth")
    
