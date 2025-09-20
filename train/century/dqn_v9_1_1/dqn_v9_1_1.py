import random
import time
import os
import pickle
import glob
import argparse
from collections import deque
import math

"""
DQN implementation for Century: Golem Edition.
This version (v9_1) uses a strategic agent as an opponent during training
instead of a random agent, which should result in more effective learning
and a more robust policy.
"""

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F

import gymnasium_env
from gymnasium.wrappers import FlattenObservation
import sys; sys.path.append('..'); from phase_agent import StrategicAgent


class DQNConfig:
    def __init__(self, **kwargs):
        self.gamma = kwargs.get('gamma', 0.99)  # Increased to prioritize long-term rewards
        self.epsilon = kwargs.get('epsilon', 1.0)
        self.epsilon_min = kwargs.get('epsilon_min', 0.05)  # Lower minimum epsilon for more exploitation
        self.epsilon_decay = kwargs.get('epsilon_decay', 0.985)  # Faster decay for quicker learning
        self.learning_rate = kwargs.get('learning_rate', 0.0007)  # Slightly higher learning rate
        self.update_rate = kwargs.get('update_rate', 150)  # More frequent target network updates
        self.batch_size = kwargs.get('batch_size', 256)  # Larger batch size for more stable learning
        self.replay_buffer_size = kwargs.get('replay_buffer_size', 100000)  # Larger buffer for diverse experiences
        self.num_timesteps = kwargs.get('num_timesteps', 2000)
        self.checkpoint_freq = kwargs.get('checkpoint_freq', 100)  # More frequent checkpoints
        self.model_save_freq = kwargs.get('model_save_freq', 50)  # More frequent model saves


class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 512)
        self.dropout1 = nn.Dropout(0.25)  # Slightly reduced dropout for faster learning
        self.fc2 = nn.Linear(512, 512)
        self.dropout2 = nn.Dropout(0.25)  # Slightly reduced dropout
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, action_size)
        
        # Initialize weights to break symmetry more effectively
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.fill_(0.01)
                
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = F.relu(self.fc3(x))
        return self.fc4(x)


class DQNAgent:
    def __init__(self, state_size, action_size, config):
        self.state_size = state_size
        self.action_size = action_size

        # Initialize Replay Buffer
        self.replay_buffer = deque(maxlen=config.replay_buffer_size)
        
        # Track rewards for learning rate scheduling
        self.rewards = []

        # Track timesteps for plotting
        self.timesteps_per_episode = []
        self.cumulative_timesteps = []
        self.total_timesteps = 0

        # Set algorithm hyperparameters
        self.gamma = config.gamma
        self.epsilon = config.epsilon
        self.epsilon_min = config.epsilon_min
        self.epsilon_decay = config.epsilon_decay
        self.learning_rate = config.learning_rate
        self.update_rate = config.update_rate
        
        # For adaptive epsilon decay
        self.total_episodes = 1000  # Expected total training episodes
        self.epsilon_decay_schedule = {}  # Will store episode-to-epsilon mapping
        self._initialize_epsilon_schedule()

        # Create both Main and Target Neural Networks
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.main_network = DQN(state_size, action_size).to(self.device)
        self.target_network = DQN(state_size, action_size).to(self.device)
        
        # Initialize Target Network with Main Network's weights
        self.target_network.load_state_dict(self.main_network.state_dict())
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.main_network.parameters(), lr=self.learning_rate, weight_decay=1e-5)  # Added weight decay
        
        # Add learning rate scheduler - more responsive to win rate changes
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='max',  # We want to maximize win rate
            factor=0.5,  # Reduce LR by half when triggered
            patience=30, # More responsive learning rate adjustment
            threshold=0.01, # Smaller threshold to detect smaller changes
            verbose=True, 
            min_lr=1e-6
        )
        
        # Track win status for more stable win rate calculation
        self.recent_wins = deque(maxlen=100)
        
        # Performance metrics
        self.training_loss = []
        self.avg_q_values = []

    def _initialize_epsilon_schedule(self):
        """Initialize an adaptive epsilon decay schedule based on expected complexity"""
        # First 100 episodes: high exploration (slow decay)
        # Middle episodes: faster decay to exploit knowledge
        # Final episodes: very low epsilon with minimal decay
        
        # Parameters for the schedule
        epsilon_start = 1.0
        epsilon_mid = 0.3
        epsilon_end = 0.05
        
        episodes_start = 0
        episodes_mid = int(self.total_episodes * 0.3)  # 30% of total episodes
        episodes_end = self.total_episodes
        
        # Create schedule for initial phase (high exploration)
        for ep in range(episodes_start, episodes_mid):
            progress = (ep - episodes_start) / (episodes_mid - episodes_start)
            self.epsilon_decay_schedule[ep] = epsilon_start - progress * (epsilon_start - epsilon_mid)
            
        # Create schedule for later phase (increasing exploitation)
        for ep in range(episodes_mid, episodes_end):
            progress = (ep - episodes_mid) / (episodes_end - episodes_mid)
            # Use exponential decay for the second half
            decay = math.exp(-5 * progress)  # Steeper exponential decay
            self.epsilon_decay_schedule[ep] = epsilon_mid - (1 - decay) * (epsilon_mid - epsilon_end)

    def update_epsilon(self, episode):
        """Update epsilon according to the predefined schedule or win rate adaptive logic"""
        # Check if we have a scheduled epsilon value for this episode
        if episode in self.epsilon_decay_schedule:
            self.epsilon = self.epsilon_decay_schedule[episode]
            return
            
        # Fallback to standard decay logic
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon_min, self.epsilon)  # Ensure we don't go below minimum

    def update_target_network(self):
        """Method to set the Main NN's weights on the Target NN"""
        self.target_network.load_state_dict(self.main_network.state_dict())

    def save_experience(self, state, action, reward, next_state, terminal):
        # Set higher priority for:
        # 1. Terminal states (game endings)
        # 2. High rewards (likely golem acquisitions)
        # 3. Actions that are rare in the buffer
        
        priority = 1.0
        
        # Terminal states
        if terminal:
            priority *= 3.0
        
        # High rewards
        if reward > 5.0:
            priority *= 2.0
        
        # Check action type
        if action >= 89 and action <= 124:  # Golem acquisition actions (getG1-getG36)
            priority *= 4.0  # Heavily prioritize golem actions
        
        # Apply merchant card hoarding penalty
        # Actions 1-43 correspond to getM3 through getM45
        if action >= 1 and action <= 43:
            # Count how many merchant cards the agent already has (excluding the starting M1, M2)
            # In the state, merchant cards status is 0 (not owned), 1 (owned but unplayable), 2 (owned and playable)
            # We extract the slice of the state containing merchant card status
            merchant_cards = state[14:59]  # This gets the status of all 45 merchant cards
            owned_cards = sum(1 for card_status in merchant_cards if card_status > 0)
            
            # Apply diminishing returns - reduce reward based on cards already owned
            # Start reducing reward after 4 additional cards (6 total including starting M1, M2)
            if owned_cards > 4:
                # Exponential penalty factor that grows as more cards are acquired
                penalty_factor = 1.0 - min(0.9, 0.25 * (owned_cards - 4))  # Increased penalty
                # Modify the actual experience's reward value (not just the priority)
                reward = reward * penalty_factor
                
                # Also reduce the priority of collecting more cards when we already have many
                priority *= penalty_factor
        
        # Add action count tracking if it doesn't exist
        if not hasattr(self, 'action_counts'):
            self.action_counts = np.ones(self.action_size)
        
        # Update action count
        self.action_counts[action] += 1
        
        # Prioritize rare actions
        rarity_factor = np.sum(self.action_counts) / (self.action_counts[action] * self.action_size)
        priority *= min(3.0, rarity_factor)  # Cap at 3x priority
        
        self.replay_buffer.append((state, action, reward, next_state, terminal, priority))

    def sample_experience_batch(self, batch_size):
        # Sample based on priority
        priorities = np.array([exp[5] for exp in self.replay_buffer])
        probs = priorities / np.sum(priorities)
        indices = np.random.choice(len(self.replay_buffer), batch_size, p=probs)
        exp_batch = [self.replay_buffer[idx] for idx in indices]

        # Create tensors for each component
        state_batch = torch.FloatTensor(np.array([batch[0] for batch in exp_batch])).to(self.device)
        action_batch = torch.LongTensor(np.array([batch[1] for batch in exp_batch])).to(self.device)
        reward_batch = torch.FloatTensor(np.array([batch[2] for batch in exp_batch])).to(self.device)
        next_state_batch = torch.FloatTensor(np.array([batch[3] for batch in exp_batch])).to(self.device)
        terminal_batch = torch.FloatTensor(np.array([batch[4] for batch in exp_batch])).to(self.device)

        return state_batch, action_batch, reward_batch, next_state_batch, terminal_batch

    def pick_epsilon_greedy_action(self, state, info):
            """Epsilon-greedy action selection with valid action masking"""
            valid_actions = info["valid_actions"]
            
            # Select random valid action with probability ε
            if random.uniform(0, 1) < self.epsilon:
                valid_indices = np.where(valid_actions == 1)[0]
                # Handle case where no valid actions exist during random selection (shouldn't happen in Century)
                if len(valid_indices) == 0:
                    # Fallback: maybe rest action if valid, otherwise error or default action
                    if valid_actions[0] == 1: return 0 
                    else: raise ValueError("No valid actions available for random choice!")
                return np.random.choice(valid_indices)

            # Pick action with highest Q-Value among valid actions
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.main_network(state_tensor)[0].cpu().numpy()
                
                # Track average Q-values for monitoring
                self.avg_q_values.append(np.mean(q_values))
                
                # Apply merchant card acquisition penalty
                # Actions 1-43 correspond to getM3 through getM45
                merchant_cards = state[14:59]  # Updated slice for v16
                owned_cards = sum(1 for card_status in merchant_cards if card_status > 0)
                
                # If agent has many cards, reduce Q-values for getting more merchant cards
                if owned_cards > 4: # Lower threshold to start penalizing earlier
                    # Stronger quadratic penalty as agent acquires more cards
                    penalty = (owned_cards - 4) ** 2 * 0.8 # Increased penalty multiplier
                    
                    # Apply penalty to getM3-getM45 actions (indices 1-43)
                    for action_idx in range(1, 44):
                        if valid_actions[action_idx] == 1:  # Only penalize valid actions
                            q_values[action_idx] -= penalty
                    
                    # Also boost Q-values for card usage actions (playM)
                    # Actions 44-88 correspond to playM1 through playM45
                    boost = min(5.0, owned_cards * 0.5)  # Proportional to owned cards, up to a limit
                    for action_idx in range(44, 89):
                        if valid_actions[action_idx] == 1:  # Only boost valid actions
                            q_values[action_idx] += boost
                
                masked_q_values = np.where(valid_actions == 1, q_values, -np.inf)
                # Handle case where all valid actions have -inf Q-value (e.g., due to heavy penalty)
                if np.all(masked_q_values == -np.inf):
                    # Fallback: Choose a random valid action instead
                    valid_indices = np.where(valid_actions == 1)[0]
                    if len(valid_indices) == 0: 
                         if valid_actions[0] == 1: return 0 # Try resting
                         else: raise ValueError("No valid actions available even after fallback!")
                    return np.random.choice(valid_indices)
                    
                return np.argmax(masked_q_values)

    def train(self, batch_size):
        # Sample a batch of experiences
        state_batch, action_batch, reward_batch, next_state_batch, terminal_batch = self.sample_experience_batch(batch_size)

        # Create a mask for valid actions in the next state
        # This would require storing valid_actions in replay buffer
        valid_action_mask = torch.zeros((batch_size, self.action_size), device=self.device)
        
        # Double DQN: Use main network to select actions, target network to evaluate them
        # Get next Q values from main network for action selection
        with torch.no_grad():
            next_q_values_main = self.main_network(next_state_batch)
            # Apply valid action masking for next state
            next_q_values_main[~valid_action_mask.bool()] = -1e9
            # Select best actions from main network (Double DQN)
            best_actions = next_q_values_main.max(1)[1]
            
            # Get action values from target network
            next_q_values_target = self.target_network(next_state_batch)
            # Use the best actions from main network to get Q-values from target network
            max_next_q = next_q_values_target.gather(1, best_actions.unsqueeze(1)).squeeze()
            target_q_values = reward_batch + (1 - terminal_batch) * self.gamma * max_next_q

        # Get current Q values
        current_q_values = self.main_network(state_batch).gather(1, action_batch.unsqueeze(1))

        # Compute loss and update weights
        loss = F.smooth_l1_loss(current_q_values.squeeze(), target_q_values)  # Huber loss for stability
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.main_network.parameters(), 1.0)
        
        self.optimizer.step()
        
        # Track loss
        self.training_loss.append(loss.item())

    def save_checkpoint(self, episode, checkpoint_dir="checkpoints"):
        """Save the current state of training to resume later"""
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Save new checkpoint and buffer without removing old ones
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_ep{episode}.pt")
        buffer_path = os.path.join(checkpoint_dir, f"replay_buffer_ep{episode}.pkl")

        checkpoint = {
            'main_network': self.main_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'episode': episode,
            'rewards': self.rewards,
            'epsilon_values': self.epsilon_values,
            'merchant_card_counts': getattr(self, 'merchant_card_counts', []),
            'training_loss': getattr(self, 'training_loss', []),
            'avg_q_values': getattr(self, 'avg_q_values', []),
            'timesteps_per_episode': getattr(self, 'timesteps_per_episode', []),
            'cumulative_timesteps': getattr(self, 'cumulative_timesteps', []),
            'total_timesteps': getattr(self, 'total_timesteps', 0)
        }
        torch.save(checkpoint, checkpoint_path)

        with open(buffer_path, 'wb') as f:
            pickle.dump(self.replay_buffer, f)

        # Also save a reference to the latest checkpoint
        latest_path = os.path.join(checkpoint_dir, "latest_checkpoint.txt")
        with open(latest_path, 'w') as f:
            f.write(f"checkpoint_ep{episode}.pt\n")
            f.write(f"replay_buffer_ep{episode}.pkl\n")

        print(f"Checkpoint saved at episode {episode}")
        return checkpoint_path
    
    def load_checkpoint(self, checkpoint_path=None, buffer_path=None, checkpoint_dir="checkpoints"):
        """Load a saved checkpoint to resume training"""
        # If no specific checkpoint provided, try to load the latest one
        if checkpoint_path is None:
            latest_path = os.path.join(checkpoint_dir, "latest_checkpoint.txt")
            if os.path.exists(latest_path):
                with open(latest_path, 'r') as f:
                    lines = f.readlines()
                    checkpoint_filename = lines[0].strip()
                    buffer_filename = lines[1].strip() if len(lines) > 1 else None
                    
                checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)
                if buffer_filename and buffer_path is None:
                    buffer_path = os.path.join(checkpoint_dir, buffer_filename)
                print(f"Loading latest checkpoint: {checkpoint_path}")
            else:
                raise FileNotFoundError(f"No latest checkpoint found in {checkpoint_dir}")
                
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        
        # Load model checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.main_network.load_state_dict(checkpoint['main_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        episode = checkpoint['episode']
        
        # Load rewards and epsilon values if available
        self.rewards = checkpoint.get('rewards', [])
        self.epsilon_values = checkpoint.get('epsilon_values', [])
        
        # Load merchant card counts if available
        self.merchant_card_counts = checkpoint.get('merchant_card_counts', [])
        
        # Load training metrics if available
        self.training_loss = checkpoint.get('training_loss', [])
        self.avg_q_values = checkpoint.get('avg_q_values', [])
        
        # Load timestep tracking data if available
        self.timesteps_per_episode = checkpoint.get('timesteps_per_episode', [])
        self.cumulative_timesteps = checkpoint.get('cumulative_timesteps', [])
        self.total_timesteps = checkpoint.get('total_timesteps', 0)
        
        # Load replay buffer if provided
        if buffer_path and os.path.exists(buffer_path):
            with open(buffer_path, 'rb') as f:
                self.replay_buffer = pickle.load(f)
            print(f"Loaded replay buffer from {buffer_path}")
        
        print(f"Loaded checkpoint from episode {episode}")
        return episode
    
    def get_episode_to_timestep_mapping(self):
        """
        Returns a dictionary mapping episode numbers to cumulative timesteps.
        Useful for plotting metrics against timesteps instead of episodes.
        """
        return {i: self.cumulative_timesteps[i] for i in range(len(self.cumulative_timesteps))}
    
    def save_model(self, episode, models_dir="models"):
        """Save just the model (not for resuming training, but for inference)"""
        os.makedirs(models_dir, exist_ok=True)
        model_path = os.path.join(models_dir, f"trained_model_{episode}.pt")
        torch.save(self.main_network.state_dict(), model_path)
        
        # Save the timestep mapping alongside the model
        timestep_data = {
            'episode': episode,
            'total_timesteps': self.total_timesteps,
            'cumulative_timesteps': self.cumulative_timesteps
        }
        timestep_path = os.path.join(models_dir, f"timestep_data_{episode}.pkl")
        with open(timestep_path, 'wb') as f:
            pickle.dump(timestep_data, f)
            
        print(f"Model saved at episode {episode}")
        return model_path
        
        
def train_dqn(config, num_episodes, checkpoint_path=None):
    """
    Train a DQN agent to play Century Golem against a strategic opponent.
    
    The DQN agent learns to play against a strategic agent which uses heuristics
    and game knowledge to make better decisions than random play. This results in
    more challenging training and potentially better policy learning.
    
    Args:
        config: Configuration object with hyperparameters
        num_episodes: Number of episodes to train for
        checkpoint_path: Optional path to resume training from checkpoint
        
    Returns:
        None
    """
    env = gym.make("gymnasium_env/CenturyGolem-v16")
    env = FlattenObservation(env)
    state, info = env.reset()

    # Define state and action size
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # Define number of timesteps per episode and batch size
    num_timesteps = config.num_timesteps
    batch_size = config.batch_size
    time_step = 0
    
    # Initialize agent and opponent
    dqn_agent = DQNAgent(state_size, action_size, config) 
    opponent = StrategicAgent(action_size)
    
    # Initialize rewards and epsilon_values lists for tracking metrics
    dqn_agent.rewards = []
    dqn_agent.epsilon_values = []
    
    # Track merchant card hoarding
    dqn_agent.merchant_card_counts = []
    
    # Start episode counter
    start_episode = 0
    
    # Load checkpoint if provided
    if checkpoint_path:
        if checkpoint_path.lower() == "latest":
            # Use the latest checkpoint feature
            start_episode = dqn_agent.load_checkpoint()
        else:
            buffer_path = checkpoint_path.replace("checkpoint", "replay_buffer")
            start_episode = dqn_agent.load_checkpoint(checkpoint_path, buffer_path)
        print(f"Resuming training from episode {start_episode}")

    try:
        for ep in range(start_episode, num_episodes):
            dqn_total_reward = 0
            opponent_total_reward = 0
            
            state, info = env.reset()
            
            # Track merchant card acquisitions in this episode
            merchant_cards_acquired = 0
            
            # Track timesteps in this episode
            episode_timesteps = 0

            print(f'\nTraining on EPISODE {ep+1} with epsilon {dqn_agent.epsilon:.4f}')
            start = time.time()

            for t in range(num_timesteps):
                time_step += 1
                episode_timesteps += 1
                dqn_agent.total_timesteps += 1

                # Update Target Network every update_rate timesteps
                if time_step % dqn_agent.update_rate == 0:
                    dqn_agent.update_target_network()

                if info['current_player'] == 0:
                    action = dqn_agent.pick_epsilon_greedy_action(state, info)
                    
                    # Track merchant card acquisitions (actions 1-43 are getM3-getM45)
                    if action >= 1 and action <= 43:
                        merchant_cards_acquired += 1
                        
                    next_state, reward, terminal, _, info = env.step(action)
                    
                    dqn_total_reward += reward
                    
                    if not terminal and info['current_player'] == 1:
                        opponent_action = opponent.pick_action(next_state, info)
                        next_state_after_opponent, opponent_reward, terminal, _, info = env.step(opponent_action)
                        
                        opponent_total_reward += opponent_reward
                        
                        dqn_agent.save_experience(state, action, reward, next_state_after_opponent, terminal)
                        
                        state = next_state_after_opponent
                    else:
                        dqn_agent.save_experience(state, action, reward, next_state, terminal)
                        state = next_state
                
                elif info['current_player'] == 1:
                    opponent_action = opponent.pick_action(state, info)
                    next_state, opponent_reward, terminal, _, info = env.step(opponent_action)
                    
                    opponent_total_reward += opponent_reward
                    state = next_state

                if terminal:
                    print('Episode: ', ep+1, ',' ' terminated with Reward ', dqn_total_reward)
                    break

                # Train the Main NN when ReplayBuffer has enough experiences
                if len(dqn_agent.replay_buffer) > batch_size:
                    dqn_agent.train(batch_size)

            # Track timesteps for this episode
            dqn_agent.timesteps_per_episode.append(episode_timesteps)
            dqn_agent.cumulative_timesteps.append(dqn_agent.total_timesteps)
            
            dqn_agent.rewards.append(dqn_total_reward)
            dqn_agent.epsilon_values.append(dqn_agent.epsilon)
            dqn_agent.merchant_card_counts.append(merchant_cards_acquired)
            
            # Update epsilon using the adaptive schedule
            dqn_agent.update_epsilon(ep)
            
            # Print merchant card acquisition stats periodically
            if ep % 10 == 0:
                avg_cards = sum(dqn_agent.merchant_card_counts[-10:]) / min(10, len(dqn_agent.merchant_card_counts[-10:]))
                print(f"Avg merchant cards acquired (last 10 ep): {avg_cards:.2f}")
                
                # Print training metrics
                if hasattr(dqn_agent, 'training_loss') and len(dqn_agent.training_loss) > 0:
                    avg_loss = sum(dqn_agent.training_loss[-100:]) / min(100, len(dqn_agent.training_loss))
                    print(f"Avg training loss (last 100 updates): {avg_loss:.4f}")
                
                if hasattr(dqn_agent, 'avg_q_values') and len(dqn_agent.avg_q_values) > 0:
                    avg_q = sum(dqn_agent.avg_q_values[-100:]) / min(100, len(dqn_agent.avg_q_values))
                    print(f"Avg Q-value (last 100 actions): {avg_q:.4f}")
                
                # Print timestep stats
                print(f"Total timesteps: {dqn_agent.total_timesteps}, Avg timesteps per episode (last 10): {sum(dqn_agent.timesteps_per_episode[-10:]) / min(10, len(dqn_agent.timesteps_per_episode[-10:])):.1f}")

            # Update Epsilon value
            if ep % 25 == 0 and ep > 0:  # Check more frequently (every 25 episodes)
                # Calculate win rate over last 25 episodes
                recent_wins = sum(1 for i in range(max(0, ep-25), ep) if dqn_agent.rewards[i] > 0)
                win_rate = recent_wins / min(25, ep)
                
                # If win rate is low, increase exploration
                if win_rate < 0.3:  # Lower threshold
                    # Calculate a boost based on how low the win rate is
                    boost = 0.2 + (0.3 - win_rate) * 0.5  # More aggressive boost for very low win rates
                    dqn_agent.epsilon = min(0.7, dqn_agent.epsilon + boost)
                    print(f"Low win rate ({win_rate:.2f}) detected, increasing epsilon to {dqn_agent.epsilon:.4f}")

            # Print episode info
            elapsed = time.time() - start
            print(f'Time elapsed during EPISODE {ep+1}: {elapsed} seconds = {round(elapsed/60, 3)} minutes')
            
            # Update learning rate based on performance
            if ep > 0 and ep % 25 == 0:  # Check more frequently
                # Track win status for each episode (1 for win, 0 for loss)
                dqn_agent.recent_wins.append(1 if dqn_agent.rewards[ep-1] > 0 else 0)
                
                # Calculate moving win rate for more stable metric
                if len(dqn_agent.recent_wins) > 0:
                    moving_win_rate = sum(dqn_agent.recent_wins) / len(dqn_agent.recent_wins)
                    
                    # Pass win rate to scheduler - higher win rates will delay learning rate reduction
                    dqn_agent.scheduler.step(moving_win_rate)
                    
                    current_lr = dqn_agent.optimizer.param_groups[0]['lr']
                    print(f"Current win rate: {moving_win_rate:.2f}, learning rate: {current_lr:.6f}")
                    
                    # If win rate drops significantly from peak, take action
                    if hasattr(dqn_agent, 'peak_win_rate'):
                        if dqn_agent.peak_win_rate > 0.7 and moving_win_rate < dqn_agent.peak_win_rate - 0.15:
                            # Lower threshold for responsiveness
                            boost = 0.1 + (dqn_agent.peak_win_rate - moving_win_rate) * 0.3
                            dqn_agent.epsilon = min(0.4, dqn_agent.epsilon + boost)
                            print(f"Win rate dropped from peak {dqn_agent.peak_win_rate:.2f} to {moving_win_rate:.2f}, increasing exploration to {dqn_agent.epsilon:.4f}")
                    
                    # Update peak win rate
                    if not hasattr(dqn_agent, 'peak_win_rate') or moving_win_rate > dqn_agent.peak_win_rate:
                        dqn_agent.peak_win_rate = moving_win_rate

            # Save checkpoint every checkpoint_freq episodes
            if (ep + 1) % config.checkpoint_freq == 0:
                dqn_agent.save_checkpoint(ep + 1)
            
            # Save model every model_save_freq episodes
            if ((ep + 1) % config.model_save_freq == 0) or (ep == 0):
                dqn_agent.save_model(ep + 1)

    except KeyboardInterrupt:
        print("\nTraining interrupted manually.")
    
    finally:
        env.close()

if __name__ == '__main__':
    # Allow for command line arguments
    parser = argparse.ArgumentParser(description='Train DQN agent for Century game')
    parser.add_argument('--episodes', type=int, default=None, help='Number of episodes to train')
    parser.add_argument('--checkpoint', type=str, default=None, 
                        help='Path to checkpoint file to resume training, or "latest" to use the most recent checkpoint')
    parser.add_argument('--reset', action='store_true', help='Flag to explicitly train from scratch (overwrites existing if no checkpoint)')
    parser.add_argument('--checkpoint-freq', type=int, default=None, help='Frequency to save checkpoints')
    parser.add_argument('--model-save-freq', type=int, default=None, help='Frequency to save model versions')
    parser.add_argument('--gamma', type=float, default=None, help='Discount factor for future rewards')
    parser.add_argument('--epsilon', type=float, default=None, help='Initial epsilon for exploration')
    parser.add_argument('--epsilon-decay', type=float, default=None, help='Decay rate for epsilon')
    parser.add_argument('--epsilon-min', type=float, default=None, help='Minimum epsilon value')
    parser.add_argument('--learning-rate', type=float, default=None, help='Learning rate for optimizer')
    parser.add_argument('--update-rate', type=int, default=None, help='Update rate for target network')
    parser.add_argument('--batch-size', type=int, default=None, help='Batch size for training')
    parser.add_argument('--replay-buffer-size', type=int, default=None, help='Size of the replay buffer')
    parser.add_argument('--num-timesteps', type=int, default=None, help='Number of timesteps per episode')
    
    args = parser.parse_args()
    
    # Safety check: prevent accidental training from scratch
    if args.checkpoint is None and not args.reset and not os.path.exists(os.path.join("checkpoints", "latest_checkpoint.txt")):
        print("Error: To train from scratch, please use the --reset flag.")
        print("To resume from a checkpoint, use the --checkpoint <path> or --checkpoint latest argument.")
        exit(1)
        
    # If no checkpoint specified but latest exists, use it
    if args.checkpoint is None and not args.reset and os.path.exists(os.path.join("checkpoints", "latest_checkpoint.txt")):
        args.checkpoint = "latest"
        print("No checkpoint specified, but found latest checkpoint. Using --checkpoint latest")
    
    config_dict = {
        'episodes': args.episodes if args.episodes is not None else 1000, 
        'checkpoint': args.checkpoint if args.checkpoint is not None else None,
        'checkpoint_freq': args.checkpoint_freq if args.checkpoint_freq is not None else 100,  # More frequent checkpoints
        'model_save_freq': args.model_save_freq if args.model_save_freq is not None else 50,  # More frequent model saves
        'gamma': args.gamma if args.gamma is not None else 0.99,  # Higher discount factor
        'epsilon': args.epsilon if args.epsilon is not None else 1.0,
        'epsilon_decay': args.epsilon_decay if args.epsilon_decay is not None else 0.985,  # Faster decay
        'epsilon_min': args.epsilon_min if args.epsilon_min is not None else 0.05,  # Lower minimum for more exploitation
        'learning_rate': args.learning_rate if args.learning_rate is not None else 0.0007,  # Higher learning rate
        'update_rate': args.update_rate if args.update_rate is not None else 150,  # More frequent updates
        'batch_size': args.batch_size if args.batch_size is not None else 256,  # Larger batch size
        'replay_buffer_size': args.replay_buffer_size if args.replay_buffer_size is not None else 100000,  # Larger buffer
        'num_timesteps': args.num_timesteps if args.num_timesteps is not None else 2000,
    }

    config = DQNConfig(**config_dict)
    train_dqn(config, config_dict['episodes'], config_dict['checkpoint'])