import random
import time
import os
import pickle
import glob
import argparse
from collections import deque
import math

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F

import gymnasium_env
from gymnasium.wrappers import FlattenObservation
import sys; sys.path.append('..'); from random_agent import RandomAgent


class DQNConfig:
    def __init__(self, **kwargs):
        self.gamma = kwargs.get('gamma', 0.98)
        self.epsilon = kwargs.get('epsilon', 1.0)
        self.epsilon_min = kwargs.get('epsilon_min', 0.05)
        self.epsilon_decay = kwargs.get('epsilon_decay', 0.997)
        self.learning_rate = kwargs.get('learning_rate', 0.0005)
        self.update_rate = kwargs.get('update_rate', 200)
        self.batch_size = kwargs.get('batch_size', 128)
        self.replay_buffer_size = kwargs.get('replay_buffer_size', 50000)
        self.num_timesteps = kwargs.get('num_timesteps', 2000)
        self.checkpoint_freq = kwargs.get('checkpoint_freq', 100)
        self.model_save_freq = kwargs.get('model_save_freq', 100)
        # New parameters for card efficiency
        self.optimal_card_count = kwargs.get('optimal_card_count', 8)  # Target number of merchant cards (includes starting 2)
        self.max_card_count = kwargs.get('max_card_count', 12)  # Hard limit after which penalties become severe
        self.round_efficiency_factor = kwargs.get('round_efficiency_factor', 0.2)  # Reward multiplier for shorter games


class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 512)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(512, 512)
        self.dropout2 = nn.Dropout(0.3)
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
        self.config = config  # Store full config for later use

        # Initialize Replay Buffer
        self.replay_buffer = deque(maxlen=config.replay_buffer_size)
        
        # Track rewards for learning rate scheduling
        self.rewards = []
        self.episode_length = []  # Track episode lengths for efficiency analysis

        # Set algorithm hyperparameters
        self.gamma = config.gamma
        self.epsilon = config.epsilon
        self.epsilon_min = config.epsilon_min
        self.epsilon_decay = config.epsilon_decay
        self.learning_rate = config.learning_rate
        self.update_rate = config.update_rate

        # Create both Main and Target Neural Networks
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.main_network = DQN(state_size, action_size).to(self.device)
        self.target_network = DQN(state_size, action_size).to(self.device)
        
        # Initialize Target Network with Main Network's weights
        self.target_network.load_state_dict(self.main_network.state_dict())
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.main_network.parameters(), lr=self.learning_rate)
        
        # Add learning rate scheduler - more responsive to win rate changes
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='max',  # We want to maximize win rate
            factor=0.5,  # Reduce LR by half when triggered
            patience=50, # Reduced patience to be more responsive
            threshold=0.01, # Smaller threshold to detect smaller changes
            verbose=True, 
            min_lr=1e-6
        )
        
        # Track win status for more stable win rate calculation
        self.recent_wins = deque(maxlen=100)
        self.rounds_to_win = deque(maxlen=100)  # Track rounds taken to win

    def update_target_network(self):
        """Method to set the Main NN's weights on the Target NN"""
        self.target_network.load_state_dict(self.main_network.state_dict())

    def save_experience(self, state, action, reward, next_state, terminal, round_num=None):
        # Set higher priority for:
        # 1. Terminal states (game endings)
        # 2. High rewards (likely golem acquisitions)
        # 3. Actions that are rare in the buffer
        # 4. Efficient gameplay (fewer rounds)
        
        priority = 1.0
        
        # Terminal states
        if terminal:
            priority *= 3.0
            
            # If we track round number, prioritize shorter games
            if round_num is not None:
                # Store round number for win efficiency tracking
                if not hasattr(self, 'win_rounds'):
                    self.win_rounds = []
                
                # Add a bonus reward for winning quickly
                if reward > 0:  # If this was a win
                    self.win_rounds.append(round_num)
                    avg_rounds = sum(self.win_rounds) / len(self.win_rounds) if self.win_rounds else 20
                    
                    # Higher reward for winning in fewer rounds than average
                    if round_num < avg_rounds:
                        efficiency_bonus = self.config.round_efficiency_factor * (avg_rounds - round_num)
                        reward += efficiency_bonus
                        priority *= 2.0  # Prioritize learning from efficient wins
                        
                    # Slightly lower reward for taking longer than average
                    elif round_num > avg_rounds:
                        efficiency_penalty = 0.5 * self.config.round_efficiency_factor * (round_num - avg_rounds)
                        reward -= min(efficiency_penalty, reward * 0.3)  # Cap penalty at 30% of original reward
        
        # High rewards
        if reward > 5.0:
            priority *= 2.0
        
        # Check action type
        if action >= 89 and action <= 124:  # Golem acquisition actions (getG1-getG36)
            priority *= 5.0  # Even more heavily prioritize golem actions - increased from 4.0
            
            # Bonus reward for golem acquisition to emphasize its importance
            reward *= 1.2  # 20% bonus for golem acquisitions
        
        # Apply merchant card hoarding penalty
        # Actions 1-43 correspond to getM3 through getM45
        if action >= 1 and action <= 43:
            # Count how many merchant cards the agent already has (including the starting M1, M2)
            merchant_cards = state[14:59]  # This gets the status of all 45 merchant cards
            owned_cards = sum(1 for card_status in merchant_cards if card_status > 0)
            
            # Apply progressive penalty based on how many cards agent already has
            if owned_cards >= self.config.optimal_card_count:
                # Calculate how far beyond optimal card count
                excess = owned_cards - self.config.optimal_card_count
                
                # Progressive penalty that grows exponentially with excess cards
                penalty_factor = min(0.95, 0.4 * (1 + excess/2)**2)
                
                # Severe penalty once we reach maximum card count
                if owned_cards >= self.config.max_card_count:
                    penalty_factor = 0.95
                    
                # Apply penalty
                reward = reward * (1 - penalty_factor)
                priority *= (1 - penalty_factor)  # Lower priority for these experiences
                
                # Add explicit negative reward for excessive card acquisition
                if owned_cards > self.config.max_card_count:
                    reward -= 0.5 * (owned_cards - self.config.max_card_count)  # Direct penalty
        
        # Prioritize card usage actions especially when agent owns many cards
        if action >= 44 and action <= 88:  # Card usage actions (useM1-useM45)
            merchant_cards = state[14:59]
            owned_cards = sum(1 for card_status in merchant_cards if card_status > 0)
            
            # Higher priority for using cards when agent has many
            if owned_cards > 5:  # If we have more than starter cards + 3
                priority *= 1.5
                
                # Small bonus for using cards instead of acquiring more
                if owned_cards > self.config.optimal_card_count:
                    reward += 0.1 * min(3, owned_cards - self.config.optimal_card_count)
        
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
        """Epsilon-greedy action selection with valid action masking and strategic biasing"""
        valid_actions = info["valid_actions"]
        
        # For pure exploration, select random valid action with probability ε
        if random.uniform(0, 1) < self.epsilon:
            valid_indices = np.where(valid_actions == 1)[0]
            if len(valid_indices) == 0:
                if valid_actions[0] == 1: return 0 
                else: raise ValueError("No valid actions available for random choice!")
            return np.random.choice(valid_indices)

        # For exploitation, use Q-values with strategic biasing
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.main_network(state_tensor)[0].cpu().numpy()
            
            # Get merchant card count
            merchant_cards = state[14:59]  # Status of all 45 merchant cards
            owned_cards = sum(1 for card_status in merchant_cards if card_status > 0)
            
            # Strategic biasing based on game state
            if owned_cards > self.config.optimal_card_count:
                # Calculate progressive penalty for excessive cards
                card_excess = owned_cards - self.config.optimal_card_count
                penalty_multiplier = min(10.0, 1.0 + card_excess**1.5)  # Progressive penalty
                
                # Penalize merchant card acquisition more severely (indices 1-43)
                for action_idx in range(1, 44):
                    if valid_actions[action_idx] == 1:
                        q_values[action_idx] -= penalty_multiplier
                
                # Boost card usage Q-values (indices 44-88)
                boost = min(6.0, owned_cards * 0.6)  # Increased boost
                for action_idx in range(44, 89):
                    if valid_actions[action_idx] == 1:
                        q_values[action_idx] += boost
                
                # Also boost golem acquisition (indices 89-124)
                for action_idx in range(89, 125):
                    if valid_actions[action_idx] == 1:
                        q_values[action_idx] += boost * 1.5  # Prioritize golem acquisition even more
            
            # Apply valid action masking
            masked_q_values = np.where(valid_actions == 1, q_values, -np.inf)
            
            # Handle case where all valid actions have -inf Q-value
            if np.all(masked_q_values == -np.inf):
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
        valid_action_mask = torch.zeros((batch_size, self.action_size), device=self.device)
        
        # Double DQN: Use main network to select actions, target network to evaluate
        with torch.no_grad():
            next_q_values_main = self.main_network(next_state_batch)
            next_q_values_main[~valid_action_mask.bool()] = -1e9
            best_actions = next_q_values_main.max(1)[1]
            
            next_q_values_target = self.target_network(next_state_batch)
            max_next_q = next_q_values_target.gather(1, best_actions.unsqueeze(1)).squeeze()
            target_q_values = reward_batch + (1 - terminal_batch) * self.gamma * max_next_q

        # Get current Q values
        current_q_values = self.main_network(state_batch).gather(1, action_batch.unsqueeze(1))

        # Use Huber loss for more stability
        loss = F.smooth_l1_loss(current_q_values.squeeze(), target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.main_network.parameters(), 1.0)
        
        self.optimizer.step()
        
        return loss.item()  # Return loss value for monitoring

    def save_checkpoint(self, episode, checkpoint_dir="checkpoints"):
        """Save the current state of training to resume later, replacing previous checkpoint files"""
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Step 1: Remove old checkpoint and buffer files
        old_ckpts = glob.glob(os.path.join(checkpoint_dir, "checkpoint_ep*.pt"))
        old_buffers = glob.glob(os.path.join(checkpoint_dir, "replay_buffer_ep*.pkl"))
        for f in old_ckpts + old_buffers:
            os.remove(f)

        # Step 2: Save new checkpoint and buffer
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
            'episode_length': getattr(self, 'episode_length', []),
            'win_rounds': getattr(self, 'win_rounds', [])
        }
        torch.save(checkpoint, checkpoint_path)

        with open(buffer_path, 'wb') as f:
            pickle.dump(self.replay_buffer, f)

        print(f"Checkpoint saved at episode {episode}")
        return checkpoint_path
    
    def load_checkpoint(self, checkpoint_path, buffer_path=None):
        """Load a saved checkpoint to resume training"""
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
        
        # Load episode length stats if available
        self.episode_length = checkpoint.get('episode_length', [])
        self.win_rounds = checkpoint.get('win_rounds', [])
        
        # Load replay buffer if provided
        if buffer_path and os.path.exists(buffer_path):
            with open(buffer_path, 'rb') as f:
                self.replay_buffer = pickle.load(f)
            print(f"Loaded replay buffer from {buffer_path}")
        
        print(f"Loaded checkpoint from episode {episode}")
        return episode
    
    def save_model(self, episode, models_dir="models"):
        """Save just the model (not for resuming training, but for inference)"""
        os.makedirs(models_dir, exist_ok=True)
        model_path = os.path.join(models_dir, f"trained_model_{episode}.pt")
        torch.save(self.main_network.state_dict(), model_path)
        print(f"Model saved at episode {episode}")
        return model_path
        
        
def train_dqn(config, num_episodes, checkpoint_path=None):
    env = gym.make("gymnasium_env/CenturyGolem-v15")
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
    opponent = RandomAgent(action_size)
    
    # Initialize rewards and epsilon_values lists for tracking metrics
    dqn_agent.rewards = []
    dqn_agent.epsilon_values = []
    
    # Track merchant card hoarding
    dqn_agent.merchant_card_counts = []
    
    # Track training metrics
    training_losses = []
    
    # Start episode counter
    start_episode = 0
    
    # Load checkpoint if provided
    if checkpoint_path:
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
            
            # Track rounds in this episode
            current_round = 1

            # Episode metrics
            episode_losses = []

            print(f'\nTraining on EPISODE {ep+1} with epsilon {dqn_agent.epsilon}')
            start = time.time()

            for t in range(num_timesteps):
                time_step += 1

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
                        
                        # Track round number from environment or keep incrementing here
                        current_round = getattr(env, 'round_number', current_round)
                        
                        # Save experience with round number for efficiency rewards
                        dqn_agent.save_experience(state, action, reward, next_state_after_opponent, terminal, current_round)
                        
                        state = next_state_after_opponent
                    else:
                        dqn_agent.save_experience(state, action, reward, next_state, terminal, current_round)
                        state = next_state
                
                elif info['current_player'] == 1:
                    opponent_action = opponent.pick_action(state, info)
                    next_state, opponent_reward, terminal, _, info = env.step(opponent_action)
                    
                    opponent_total_reward += opponent_reward
                    state = next_state

                if terminal:
                    # Track episode length for efficiency analysis
                    dqn_agent.episode_length.append(current_round)
                    
                    # Calculate average rounds to win for monitoring efficiency
                    avg_rounds = sum(dqn_agent.episode_length) / len(dqn_agent.episode_length)
                    
                    win_status = "WIN" if dqn_total_reward > 0 else "LOSS"
                    print(f'Episode: {ep+1}, terminated in round {current_round} with {win_status}, Reward: {dqn_total_reward:.2f}, Avg rounds: {avg_rounds:.1f}')
                    break

                # Train the Main NN when ReplayBuffer has enough experiences
                if len(dqn_agent.replay_buffer) > batch_size:
                    loss = dqn_agent.train(batch_size)
                    episode_losses.append(loss)

            # Store average training loss for this episode
            if episode_losses:
                avg_loss = sum(episode_losses) / len(episode_losses)
                training_losses.append(avg_loss)
                print(f"Avg training loss: {avg_loss:.4f}")
            
            dqn_agent.rewards.append(dqn_total_reward)
            dqn_agent.epsilon_values.append(dqn_agent.epsilon)
            dqn_agent.merchant_card_counts.append(merchant_cards_acquired)
            
            # Print merchant card acquisition stats periodically
            if ep % 10 == 0:
                avg_cards = sum(dqn_agent.merchant_card_counts[-10:]) / min(10, len(dqn_agent.merchant_card_counts[-10:]))
                
                # Also calculate win rate over last 10 episodes
                recent_wins = sum(1 for i in range(max(0, ep-9), ep+1) if dqn_agent.rewards[i] > 0)
                win_rate = recent_wins / 10
                
                print(f"Last 10 episodes - Win rate: {win_rate:.1%}, Avg merchant cards acquired: {avg_cards:.2f}")
                
                # Show average rounds for winning episodes if available
                if hasattr(dqn_agent, 'win_rounds') and dqn_agent.win_rounds:
                    recent_win_rounds = dqn_agent.win_rounds[-10:]
                    if recent_win_rounds:
                        avg_win_rounds = sum(recent_win_rounds) / len(recent_win_rounds)
                        print(f"Avg rounds to win: {avg_win_rounds:.1f}")

            # Update Epsilon with more aggressive decay when agent is performing well
            if ep % 25 == 0 and ep > 0:
                # Calculate win rate over last 25 episodes
                recent_wins = sum(1 for i in range(max(0, ep-25), ep) if dqn_agent.rewards[i] > 0)
                win_rate = recent_wins / min(25, ep)
                
                # Dynamic epsilon adjustment based on win rate
                if win_rate < 0.4:
                    # Poor performance: increase exploration
                    dqn_agent.epsilon = min(0.7, dqn_agent.epsilon + 0.15)
                    print(f"Low win rate detected, increasing epsilon to {dqn_agent.epsilon}")
                elif win_rate > 0.7 and dqn_agent.epsilon > 0.2:
                    # Good performance: accelerate the decrease to encourage exploitation
                    dqn_agent.epsilon = max(dqn_agent.epsilon_min, dqn_agent.epsilon * (dqn_agent.epsilon_decay**2))
                    print(f"High win rate detected, decreasing epsilon to {dqn_agent.epsilon}")
                elif dqn_agent.epsilon > dqn_agent.epsilon_min:
                    # Standard decay
                    dqn_agent.epsilon = max(dqn_agent.epsilon_min, dqn_agent.epsilon * dqn_agent.epsilon_decay)

            # Print episode info
            elapsed = time.time() - start
            print(f'Time elapsed during EPISODE {ep+1}: {elapsed:.1f} seconds = {round(elapsed/60, 3)} minutes')
            
            # Update learning rate based on performance
            if ep > 0 and ep % 50 == 0:
                # Use win rate over last 50 episodes to update learning rate
                recent_wins = sum(1 for i in range(max(0, ep-50), ep) if dqn_agent.rewards[i] > 0)
                win_rate = recent_wins / min(50, ep)
                
                # Track win status for each episode (1 for win, 0 for loss)
                dqn_agent.recent_wins.append(1 if dqn_agent.rewards[ep-1] > 0 else 0)
                
                # Calculate moving win rate for more stable metric
                if len(dqn_agent.recent_wins) > 0:
                    moving_win_rate = sum(dqn_agent.recent_wins) / len(dqn_agent.recent_wins)
                    
                    # Also calculate efficiency metrics
                    if dqn_agent.episode_length:
                        avg_rounds = sum(dqn_agent.episode_length[-50:]) / len(dqn_agent.episode_length[-50:])
                        print(f"Average rounds per episode (last 50): {avg_rounds:.1f}")
                    
                    # Pass win rate to scheduler - higher win rates will delay learning rate reduction
                    # since the mode is 'max', it will reduce LR when win rate plateaus or decreases
                    dqn_agent.scheduler.step(moving_win_rate)
                    
                    current_lr = dqn_agent.optimizer.param_groups[0]['lr']
                    print(f"Current win rate: {moving_win_rate:.2f}, learning rate: {current_lr:.6f}")
                    
                    # If win rate drops significantly from peak, take action
                    if hasattr(dqn_agent, 'peak_win_rate'):
                        if dqn_agent.peak_win_rate > 0.8 and moving_win_rate < dqn_agent.peak_win_rate - 0.2:
                            # If significant drop from high performance, slightly increase exploration
                            dqn_agent.epsilon = min(0.3, dqn_agent.epsilon + 0.05)
                            print(f"Win rate dropped from peak {dqn_agent.peak_win_rate:.2f} to {moving_win_rate:.2f}, increasing exploration to {dqn_agent.epsilon:.2f}")
                    
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
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint file to resume training')
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
    parser.add_argument('--optimal-cards', type=int, default=None, help='Target number of merchant cards')
    parser.add_argument('--max-cards', type=int, default=None, help='Maximum number of merchant cards')
    parser.add_argument('--round-factor', type=float, default=None, help='Round efficiency factor')
    
    args = parser.parse_args()
    
    # Safety check: prevent accidental training from scratch
    if args.checkpoint is None and not args.reset:
        print("Error: To train from scratch, please use the --reset flag.")
        print("To resume from a checkpoint, use the --checkpoint <path> argument.")
        exit(1)
        
    config_dict = {
        'episodes': args.episodes if args.episodes is not None else 1200, # 10000
        'checkpoint': args.checkpoint if args.checkpoint is not None else None,
        'checkpoint_freq': args.checkpoint_freq if args.checkpoint_freq is not None else 100,
        'model_save_freq': args.model_save_freq if args.model_save_freq is not None else 100,
        'gamma': args.gamma if args.gamma is not None else 0.99,
        'epsilon': args.epsilon if args.epsilon is not None else 1.0,
        'epsilon_decay': args.epsilon_decay if args.epsilon_decay is not None else 0.998,
        'epsilon_min': args.epsilon_min if args.epsilon_min is not None else 0.1,
        'learning_rate': args.learning_rate if args.learning_rate is not None else 0.0001,
        'update_rate': args.update_rate if args.update_rate is not None else 200,
        'batch_size': args.batch_size if args.batch_size is not None else 128,
        'replay_buffer_size': args.replay_buffer_size if args.replay_buffer_size is not None else 50000,
        'num_timesteps': args.num_timesteps if args.num_timesteps is not None else 2000,
        'optimal_card_count': args.optimal_cards if args.optimal_cards is not None else 8,
        'max_card_count': args.max_cards if args.max_cards is not None else 12,
        'round_efficiency_factor': args.round_factor if args.round_factor is not None else 0.2,
    }

    config = DQNConfig(**config_dict)
    train_dqn(config, config_dict['episodes'], config_dict['checkpoint'])