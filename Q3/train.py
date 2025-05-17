import gymnasium as gym
import torch
import torch.nn as nn
import random
import numpy as np
import torch.nn.functional as F
from collections import deque
from torch.distributions import Normal
from dmc import make_dmc_env # Ensure this wrapper is correctly installed and provides a gym-like API

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def make_env():
    """Creates the DeepMind Control Suite environment."""
    env_name = "humanoid-walk"
    env = make_dmc_env(env_name, np.random.randint(0, 1000000), flatten=True, use_pixels=False)
    return env

class Critic(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        # Combined state-action input
        input_dim = obs_dim + action_dim
        self.critic1 = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.critic2 = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, state, action):
        # Concatenate along the feature dimension (dim=1 for batches)
        x = torch.cat([state, action], dim=1)
        return self.critic1(x), self.critic2(x)

    def Q1(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.critic1(x)

class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, max_action=1.0): # max_action for scaling
        super().__init__()
        self.actor_base = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.actor_mean = nn.Linear(128, action_dim)
        self.actor_log_std = nn.Linear(128, action_dim)
        
        # Action scaling: SAC actions are tanh squashed to [-1, 1], then scaled to env limits.
        # Humanoid-walk typically has action bounds of [-1, 1], so max_action is 1.0.
        self.max_action = max_action
        self.LOG_STD_MIN = -20
        self.LOG_STD_MAX = 1


    def forward(self, state):
        a = self.actor_base(state)
        mean = self.actor_mean(a)
        log_std = self.actor_log_std(a)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX) # Clamping log_std
        std = log_std.exp()
        return mean, std

    def sample(self, state):
        mean, std = self.forward(state)
        dist = Normal(mean, std)
        action_raw = dist.rsample()  # Use rsample() for reparameterization trick
        
        # Apply tanh squashing
        action_squashed = torch.tanh(action_raw)
        action_scaled = action_squashed * self.max_action # Scale to environment action range

        # Compute log probability, accounting for tanh transformation and scaling
        # The formula for log_prob change due to tanh is:
        # log_prob = dist.log_prob(raw_action) - sum(log(1 - tanh(raw_action)^2 + eps))
        # If actions are further scaled by max_action, it's log_prob - sum(log(max_action * (1 - tanh(raw_action)^2) + eps))
        # However, the standard SAC correction is usually applied to the squashed action before scaling.
        log_prob = dist.log_prob(action_raw)
        log_prob -= torch.log(1 - action_squashed.pow(2) + 1e-6) # Add epsilon for numerical stability
        log_prob = log_prob.sum(dim=1, keepdim=True) # Sum over action dimensions

        return action_scaled, log_prob


class Agent:
    def __init__(self, obs_dim, action_dim, action_space, # Pass action_space for bounds
                 gamma=0.99, initial_alpha=0.2, # initial_alpha instead of fixed alpha
                 lr=1e-3, # Commonly used LR for SAC
                 batch_size=256, buffer_size=1000000,
                 learn_start=10000, # Start learning after this many steps
                 tau=0.005, target_update_interval=1):
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.learn_start = learn_start
        self.target_update_interval = target_update_interval
        self.action_dim = action_dim

        # Determine action scaling factor (max_action)
        # Assumes symmetric action space [-max_action, max_action]
        # For dmc humanoid-walk, this is typically 1.0.
        self.max_action = float(action_space.high[0]) if action_space is not None else 1.0


        # Actor and Critic networks
        self.p_net = Actor(obs_dim, action_dim, self.max_action).to(device)
        self.q_net = Critic(obs_dim, action_dim).to(device)
        self.p_target = Actor(obs_dim, action_dim, self.max_action).to(device)
        self.q_target = Critic(obs_dim, action_dim).to(device)

        # Synchronize target networks
        self.p_target.load_state_dict(self.p_net.state_dict())
        self.q_target.load_state_dict(self.q_net.state_dict())

        # Optimizers
        self.q_optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.p_optimizer = torch.optim.Adam(self.p_net.parameters(), lr=lr)

        # Automatic entropy tuning (alpha)
        # Target entropy is often set to -action_dim
        self.target_entropy = -torch.tensor(action_dim, dtype=torch.float32, device=device).item()
        self.log_alpha = torch.tensor(np.log(initial_alpha), dtype=torch.float32, requires_grad=True, device=device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr) # Can use same lr or dedicated
        self.alpha = self.log_alpha.exp().item() # Current alpha value

        self.memory = deque(maxlen=buffer_size)
        self.train_steps_done = 0 # Counter for learning steps

    def select_action(self, state, evaluate=False):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            mean, std = self.p_net(state_tensor)
            if evaluate: # Deterministic action for evaluation
                action_raw = mean
            else: # Stochastic action for training/exploration
                dist = Normal(mean, std)
                action_raw = dist.sample() # No rsample needed if not backpropping here

            action_squashed = torch.tanh(action_raw)
            action_scaled = action_squashed * self.max_action
        return action_scaled.cpu().numpy().flatten()

    def add_to_memory(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def learn(self):
        if len(self.memory) < self.batch_size or len(self.memory) < self.learn_start:
            return None # Not enough samples or not time to learn yet

        batch = random.sample(self.memory, self.batch_size)
        # states, actions, rewards, next_states, dones = map(np.array, zip(*batch)) # Simpler way
        states_np = np.array([transition[0] for transition in batch])
        actions_np = np.array([transition[1] for transition in batch])
        rewards_np = np.array([transition[2] for transition in batch]).reshape(-1, 1)
        next_states_np = np.array([transition[3] for transition in batch])
        dones_np = np.array([transition[4] for transition in batch]).reshape(-1, 1)


        states = torch.tensor(states_np, dtype=torch.float32).to(device)
        actions = torch.tensor(actions_np, dtype=torch.float32).to(device)
        rewards = torch.tensor(rewards_np, dtype=torch.float32).to(device)
        next_states = torch.tensor(next_states_np, dtype=torch.float32).to(device)
        dones = torch.tensor(dones_np, dtype=torch.float32).to(device)


        # --- Update Critic ---
        with torch.no_grad():
            next_actions_scaled, next_log_probs = self.p_net.sample(next_states)
            q1_target_next, q2_target_next = self.q_target(next_states, next_actions_scaled)
            min_q_target_next = torch.min(q1_target_next, q2_target_next)
            # Alpha is self.log_alpha.exp() for the current value in calculation
            target_q_values = rewards + (1.0 - dones) * self.gamma * (min_q_target_next - self.log_alpha.exp() * next_log_probs)

        q1_current, q2_current = self.q_net(states, actions)
        q_loss = F.mse_loss(q1_current, target_q_values) + F.mse_loss(q2_current, target_q_values)

        self.q_optimizer.zero_grad()
        q_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=1.0) # Optional gradient clipping
        self.q_optimizer.step()

        # --- Update Actor and Alpha ---
        # Freeze Q-networks to prevent gradients from flowing into them during policy update
        for p in self.q_net.parameters():
            p.requires_grad = False

        new_actions_scaled, log_probs = self.p_net.sample(states)
        q1_new_actions, q2_new_actions = self.q_net(states, new_actions_scaled)
        min_q_new_actions = torch.min(q1_new_actions, q2_new_actions)
        policy_loss = (self.log_alpha.exp().detach() * log_probs - min_q_new_actions).mean() # Detach alpha for policy loss

        self.p_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.p_net.parameters(), max_norm=1.0) # Optional
        self.p_optimizer.step()

        # Unfreeze Q-networks
        for p in self.q_net.parameters():
            p.requires_grad = True
            
        # --- Update Alpha ---
        # Use the same log_probs from the policy update step
        alpha_loss = -(self.log_alpha.exp() * (log_probs.detach() + self.target_entropy)).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        torch.nn.utils.clip_grad_norm_([self.log_alpha], max_norm=1.0)
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp().item() # Update current alpha value

        # --- Soft update target networks ---
        if self.train_steps_done % self.target_update_interval == 0:
            for target_param, local_param in zip(self.q_target.parameters(), self.q_net.parameters()):
                target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
            for target_param, local_param in zip(self.p_target.parameters(), self.p_net.parameters()):
                target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
        
        self.train_steps_done += 1
        return q_loss.item(), policy_loss.item(), alpha_loss.item(), self.alpha


    def save(self, filepath='sac_model'):
        torch.save(self.p_net.state_dict(), f'{filepath}_actor.pth')
        torch.save(self.q_net.state_dict(), f'{filepath}_critic.pth')
        torch.save({
            'log_alpha': self.log_alpha,
            'alpha_optimizer_state_dict': self.alpha_optimizer.state_dict()
        }, f"{filepath}_alpha.pth")
        print(f"Models saved: {filepath}_actor.pth, {filepath}_critic.pth, {filepath}_alpha.pth")

    def load(self, filepath='sac_model'):
        self.p_net.load_state_dict(torch.load(f'{filepath}_actor.pth', map_location=device))
        self.q_net.load_state_dict(torch.load(f'{filepath}_critic.pth', map_location=device))
        
        alpha_checkpoint = torch.load(f'{filepath}_alpha.pth', map_location=device)
        self.log_alpha = alpha_checkpoint['log_alpha']
        self.alpha_optimizer.load_state_dict(alpha_checkpoint['alpha_optimizer_state_dict'])
        self.alpha = self.log_alpha.exp().item()

        self.p_target.load_state_dict(self.p_net.state_dict())
        self.q_target.load_state_dict(self.q_net.state_dict())
        print(f"Models loaded: {filepath}_actor.pth, {filepath}_critic.pth, {filepath}_alpha.pth")


def train_sac_main_loop(agent, env, total_timesteps=1000000, log_interval=1000, save_interval_eps=10):
    # Initial observation from environment
    # The dmc.make_dmc_env wrapper should provide a gym-like (obs, info) tuple
    # obs here is assumed to be a flattened numpy array.
    raw_obs, info = env.reset() 
    
    # Ensure obs is a flat numpy array. This depends on the dmc wrapper.
    # If `flatten=True` works as expected, raw_obs should already be a np.ndarray.
    # If raw_obs is a dict (e.g. from dm_control TimeStep.observation), extract and flatten.
    if isinstance(raw_obs, dict):
        # This is a heuristic; specific key might be 'observations' or similar
        # Or you might need to concatenate values: np.concatenate([v.flatten() for v in raw_obs.values()])
        try:
            # Try to extract from a common key if the wrapper nests it
            if 'observations' in raw_obs: obs = raw_obs['observations']
            elif 'state' in raw_obs: obs = raw_obs['state']
            else: # If still dict, try to concatenate all numeric values
                 obs = np.concatenate([v.ravel() if isinstance(v, np.ndarray) else np.array([v]) for k, v in raw_obs.items() if isinstance(v, (np.ndarray, float, int))]).astype(np.float32)

        except Exception as e:
            print(f"Error processing initial observation dictionary: {raw_obs}. Error: {e}")
            print("Assuming raw_obs is already the flat observation vector.")
            obs = raw_obs # Fallback
    else:
        obs = raw_obs
    obs = np.asarray(obs, dtype=np.float32)


    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0
    rewards_history = []
    best_mean_reward = -float('inf')
    episode_limit = 200
    for t_step in range(1, total_timesteps + 1):
        episode_timesteps += 1

        if t_step < agent.learn_start:
            action = env.action_space.sample() # Assumes dmc wrapper provides gym-like action_space
        else:
            action = agent.select_action(obs) # Use current observation

        next_obs, reward, terminated, truncated, info = env.step(action)
        next_obs = np.asarray(next_obs, dtype=np.float32)
        
        done = terminated or truncated
        
        agent.add_to_memory(obs, action, reward, next_obs, float(done))

        obs = next_obs # Update current observation
        episode_reward += reward

        if t_step >= agent.learn_start:
            loss_info = agent.learn()
            if loss_info and t_step % log_interval == 0:
                q_loss, p_loss, alpha_loss, current_alpha = loss_info
                print(f"T: {t_step}, Ep: {episode_num}, EpSteps: {episode_timesteps}, QL: {q_loss:.2f}, PL: {p_loss:.2f}, AlphaL: {alpha_loss:.2f}, Alpha: {current_alpha:.3f}, EpR: {episode_reward:.2f} (ongoing)")

        if done or episode_timesteps == episode_limit:
            rewards_history.append(episode_reward)
            mean_reward_last_100 = np.mean(rewards_history[-100:]) if rewards_history else -float('inf')
            if mean_reward_last_100 > episode_limit * 0.4:
                episode_limit += 200            
            print(f"Total T: {t_step}, Episode: {episode_num + 1}, Episode Steps: {episode_timesteps}, Reward: {episode_reward:.3f}, Mean Reward (last 100): {mean_reward_last_100:.3f}")
            
            if episode_num % save_interval_eps == 0 and episode_num > 0:
                if mean_reward_last_100 > best_mean_reward and len(rewards_history) >= 10: # Avoid saving too early
                    best_mean_reward = mean_reward_last_100
                    agent.save(filepath=f'humanoid_sac')
                    print(f"****** New best model saved with mean reward: {best_mean_reward:.3f} at episode {episode_num} ******")


            raw_obs, info = env.reset() # Reset environment
            if isinstance(raw_obs, dict):
                try:
                    if 'observations' in raw_obs: obs = raw_obs['observations']
                    elif 'state' in raw_obs: obs = raw_obs['state']
                    else:
                        obs = np.concatenate([v.ravel() if isinstance(v, np.ndarray) else np.array([v]) for k, v in raw_obs.items() if isinstance(v, (np.ndarray, float, int))]).astype(np.float32)
                except Exception as e:
                    print(f"Error processing reset observation dictionary: {raw_obs}. Error: {e}")
                    obs = raw_obs # Fallback
            else:
                obs = raw_obs
            obs = np.asarray(obs, dtype=np.float32)

            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1


if __name__ == "__main__":
    env = make_env()

    # --- Critical: Determine Observation and Action Dimensions ---
    # This part is highly dependent on your `dmc.make_dmc_env` wrapper's output.
    # The wrapper should ideally provide standard Gym spaces.
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_space = env.action_space # For action bounds
    print(f"Successfully retrieved dimensions from Gym-like spaces: Obs Dim: {obs_dim}, Action Dim: {action_dim}")
    # Verify against your original hardcoded values (obs=67, act=21 for humanoid-walk is typical)


    agent = Agent(obs_dim=obs_dim,
                  action_dim=action_dim,
                  action_space=action_space, # Pass the action_space object
                  learn_start=0,        # Your original value
                  lr=0.001,                  # Common SAC LR
                  initial_alpha=0.8,        # A common starting alpha
                  buffer_size=1000000,
                  batch_size=256,
                  tau=0.005,
                  target_update_interval=1) # SAC often updates targets every step

    # Example: Load a previously saved model
    # try:
    #     agent.load(filepath='humanoid_sac_best_epXXX') # Replace XXX
    #     print("Agent successfully loaded from checkpoint.")
    # except FileNotFoundError:
    #     print("No checkpoint found, starting training from scratch.")

    print(f"Using device: {device}")
    print(f"Starting training with Agent: obs_dim={obs_dim}, action_dim={action_dim}, max_action={agent.max_action}")
    #agent.load('humanoid_sac_final')
    train_sac_main_loop(agent, env, total_timesteps=10000000, log_interval=5000, save_interval_eps=20)

    env.close()
    agent.save(filepath='humanoid_sac_final')
    print("Training finished. Final model saved.")