import json
import random
import time
import pickle
from collections import deque

import gymnasium as gym
import gymnasium_env
from gymnasium.wrappers import FlattenObservation
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        if x.size(0) > 1:  # Only apply batch norm if batch size > 1
            x = self.bn1(x)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)


class PrioritizedReplayBuffer:
    def __init__(self, max_size, alpha=0.6):
        self.max_size = max_size
        self.alpha = alpha
        self.buffer = []
        self.priorities = np.zeros((max_size,), dtype=np.float32)
        self.position = 0

    def __len__(self):
        """Returns the number of elements in the buffer"""
        return len(self.buffer)

    def add(self, experience, td_error):
        max_priority = self.priorities.max() if self.buffer else 1.0
        if len(self.buffer) < self.max_size:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.max_size

    def sample(self, batch_size, beta=0.4):
        """Sample batch of experiences based on priority"""
        if len(self.buffer) == 0:
            return [], [], []

        priorities = self.priorities[:len(self.buffer)] ** self.alpha
        probs = priorities / priorities.sum()  # Normalize to get probability distribution

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)  # Sample using priorities
        experiences = [self.buffer[idx] for idx in indices]

        # Importance Sampling (IS) Weights
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()  # Normalize weights

        return experiences, indices, weights

    def update_priorities(self, indices, td_errors):
        """Update priorities based on new TD errors"""
        self.priorities[indices] = np.abs(td_errors) + 1e-6  # Avoid zero priority


class RandomAgent:
    def __init__(self, action_size):
        self.action_size = action_size

    def pick_action(self, state, info):
        valid_indices = np.where(info["valid_actions"] == 1)[0]
        return np.random.choice(valid_indices)


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        # Initialize Prioritized Replay Buffer
        self.replay_buffer = PrioritizedReplayBuffer(max_size=100000)

        # Set algorithm hyperparameters
        self.gamma = 0.95
        self.epsilon = 0.5
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.0003
        self.update_rate = 250

        # Create both Main and Target Neural Networks
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.main_network = DQN(state_size, action_size).to(self.device)
        self.target_network = DQN(state_size, action_size).to(self.device)
        
        # Initialize Target Network with Main Network's weights
        self.target_network.load_state_dict(self.main_network.state_dict())
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.main_network.parameters(), lr=self.learning_rate)

    def update_target_network(self):
        """Method to set the Main NN's weights on the Target NN"""
        self.target_network.load_state_dict(self.main_network.state_dict())

    def save_experience(self, state, action, reward, next_state, terminal):
        """Compute TD error and save prioritized experience"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
        
        # Compute TD error: TD_error = |Q(s,a) - target|
        with torch.no_grad():
            q_values = self.main_network(state_tensor)
            next_q_values = self.target_network(next_state_tensor)
            
            target = reward if terminal else reward + self.gamma * torch.max(next_q_values).item()
            td_error = abs(q_values[0, action].item() - target)

        # Prioritize golem purchases (actions 19-23 in the environment)
        if 19 <= action <= 23:
            td_error += 1.0  # Boost priority for golem purchases

        # Add experience with priority
        self.replay_buffer.add((state, action, reward, next_state, terminal), td_error)

    def pick_action(self, state, info):
        """Epsilon-greedy action selection with valid action masking"""
        valid_actions = info["valid_actions"]
        
        # Select random valid action with probability ε
        if random.uniform(0, 1) < self.epsilon:
            valid_indices = np.where(valid_actions == 1)[0]
            return np.random.choice(valid_indices)

        # Pick action with highest Q-Value among valid actions
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.main_network(state_tensor)[0].cpu().numpy()
            masked_q_values = np.where(valid_actions == 1, q_values, -np.inf)
            return np.argmax(masked_q_values)

    def train(self, batch_size, beta=0.4):
        """Train the network using prioritized experience replay"""
        experiences, indices, weights = self.replay_buffer.sample(batch_size, beta)
        if not experiences:
            return

        # Convert experiences to tensors
        states = torch.FloatTensor([e[0] for e in experiences]).to(self.device)
        actions = torch.LongTensor([e[1] for e in experiences]).to(self.device)
        rewards = torch.FloatTensor([e[2] for e in experiences]).to(self.device)
        next_states = torch.FloatTensor([e[3] for e in experiences]).to(self.device)
        terminals = torch.FloatTensor([e[4] for e in experiences]).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)

        # Compute Q values and targets
        current_q_values = self.main_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (1 - terminals) * self.gamma * next_q_values

        # Compute TD errors for updating priorities
        td_errors = torch.abs(current_q_values - target_q_values).detach().cpu().numpy()

        # Compute loss with importance sampling weights
        loss = F.mse_loss(current_q_values, target_q_values, reduction='none')
        weighted_loss = (loss * weights).mean()

        # Update network
        self.optimizer.zero_grad()
        weighted_loss.backward()
        self.optimizer.step()

        # Update priorities in buffer
        self.replay_buffer.update_priorities(indices, td_errors)

    def save_agent(self, filename_prefix):
        """Save the agent's models and buffer"""
        torch.save(self.main_network.state_dict(), f'{filename_prefix}_main_model.pth')
        torch.save(self.target_network.state_dict(), f'{filename_prefix}_target_model.pth')
        
        # Save additional metadata
        metadata = {
            'epsilon': self.epsilon,
            'replay_buffer': self.replay_buffer
        }
        with open(f'{filename_prefix}_metadata.pkl', 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"Agent saved as '{filename_prefix}'")

    def load_agent(self, filename_prefix):
        """Load the agent's models and buffer"""
        self.main_network.load_state_dict(torch.load(f'{filename_prefix}_main_model.pth'))
        self.target_network.load_state_dict(torch.load(f'{filename_prefix}_target_model.pth'))
        
        try:
            with open(f'{filename_prefix}_metadata.pkl', 'rb') as f:
                metadata = pickle.load(f)
                self.epsilon = metadata['epsilon']
                self.replay_buffer = metadata['replay_buffer']
            print(f"Agent loaded from '{filename_prefix}'")
        except FileNotFoundError:
            print(f"Metadata file not found. Only model weights loaded.")


class TrainingLogger:
    def __init__(self, filename_prefix):
        self.filename_prefix = filename_prefix
        self.metadata_file = f'{filename_prefix}_metadata.json'
        
        # Initialize or load training metadata
        try:
            with open(self.metadata_file, 'r') as f:
                self.metadata = json.load(f)
                print(f"Training metadata loaded from '{self.metadata_file}'")
        except (FileNotFoundError, json.JSONDecodeError):
            self.metadata = {
                'episode': 0, 
                'time_step': 0, 
                'rewards': [], 
                'epsilon_values': [],
                'opponent_rewards': [],
                'win_rate': [],
                'actions_taken': {}
            }
            print(f"No previous training metadata found. Starting fresh.")
    
    def update_episode_data(self, episode, dqn_reward, opponent_reward, epsilon, actions_taken, win):
        """Update data after each episode"""
        self.metadata['episode'] = episode + 1
        self.metadata['rewards'].append(dqn_reward)
        self.metadata['opponent_rewards'].append(opponent_reward)
        self.metadata['epsilon_values'].append(epsilon)
        
        # Track win/loss ratio over last 100 games
        self.metadata['win_rate'].append(1 if win else 0)
        if len(self.metadata['win_rate']) > 100:
            self.metadata['win_rate'].pop(0)
        
        # Track action distribution
        for action in actions_taken:
            if str(action) not in self.metadata['actions_taken']:
                self.metadata['actions_taken'][str(action)] = 0
            self.metadata['actions_taken'][str(action)] += 1
    
    def update_time_step(self, time_step):
        """Update current time step"""
        self.metadata['time_step'] = time_step
    
    def save(self):
        """Save current metadata to file"""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f)
        
        # Save rewards and epsilon values separately for plotting
        with open(f'{self.filename_prefix}_rewards.txt', 'w') as f:
            json.dump(self.metadata['rewards'], f)
        
        with open(f'{self.filename_prefix}_epsilon.txt', 'w') as f:
            json.dump(self.metadata['epsilon_values'], f)
        
        with open(f'{self.filename_prefix}_opponent_rewards.txt', 'w') as f:
            json.dump(self.metadata['opponent_rewards'], f)
        
        print(f"Training metadata saved to '{self.filename_prefix}' files")
    
    def get_time_step(self):
        return self.metadata['time_step']
    
    def get_episode(self):
        return self.metadata['episode']
    
    def get_win_rate(self):
        """Calculate win rate over last 100 games"""
        if not self.metadata['win_rate']:
            return 0
        return sum(self.metadata['win_rate']) / len(self.metadata['win_rate'])


def train_dqn(num_episodes=1000, continue_training=False, model_name='century_dqn',
              opponent_type='random', opponent_model=None):
    """
    Train a DQN agent in the Century environment
    
    Parameters:
    - num_episodes: Number of episodes to train for
    - continue_training: Whether to continue from a previous checkpoint
    - model_name: Prefix for saving model files
    - opponent_type: Type of opponent ('random' or 'model')
    - opponent_model: Path to opponent model if opponent_type is 'model'
    """
    # Initialize environment
    env = gym.make("gymnasium_env/CenturyGolem-v9")
    env = FlattenObservation(env)
    
    # Get state and action sizes
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    # Initialize agents
    dqn_agent = DQNAgent(state_size, action_size)
    
    # Initialize opponent based on type
    if opponent_type == 'random':
        opponent = RandomAgent(action_size)
    elif opponent_type == 'model':
        opponent = DQNAgent(state_size, action_size)
        if opponent_model:
            opponent.load_agent(opponent_model)
        else:
            print("Warning: No opponent model specified. Using random initialization.")
    else:
        raise ValueError(f"Unknown opponent type: {opponent_type}")
    
    # Initialize logger
    logger = TrainingLogger(model_name)
    
    # Load previous checkpoint if continuing training
    if continue_training:
        try:
            dqn_agent.load_agent(model_name)
            print(f"Continuing training from episode {logger.get_episode()}")
        except FileNotFoundError:
            print("No previous model found. Starting fresh training.")
    
    # Training parameters
    num_timesteps = 400
    batch_size = 128
    time_step = logger.get_time_step()
    
    try:
        # Main training loop
        for ep in range(logger.get_episode(), num_episodes):
            # Reset environment
            state, info = env.reset()
            
            dqn_total_reward = 0
            opponent_total_reward = 0
            dqn_actions_taken = []
            episode_start = time.time()
            
            print(f'\nTraining on EPISODE {ep+1} with epsilon {dqn_agent.epsilon}')
            
            # Episode loop
            for t in range(num_timesteps):
                time_step += 1
                
                # Update target network periodically
                if time_step % dqn_agent.update_rate == 0:
                    dqn_agent.update_target_network()
                
                # DQN agent's turn (player 1)
                if info['current_player'] == 0:
                    # Select and perform action
                    action = dqn_agent.pick_action(state, info)
                    dqn_actions_taken.append(action)
                    next_state, reward, terminal, _, info = env.step(action)
                    
                    dqn_total_reward += reward
                    
                    # If game continues, let opponent take turn
                    if not terminal and info['current_player'] == 1:
                        # Opponent's turn
                        opponent_action = opponent.pick_action(next_state, info)
                        next_state_after_opponent, opponent_reward, terminal, _, info = env.step(opponent_action)
                        
                        opponent_total_reward += opponent_reward
                        
                        # Store experience with state after opponent's move
                        dqn_agent.save_experience(state, action, reward, next_state_after_opponent, terminal)
                        
                        # Update state for next iteration
                        state = next_state_after_opponent
                    else:
                        # Store experience with immediate next state if game ends or it's still DQN's turn
                        dqn_agent.save_experience(state, action, reward, next_state, terminal)
                        state = next_state
                
                # Opponent's turn (player 2)
                elif info['current_player'] == 1:
                    # Select and perform opponent action
                    opponent_action = opponent.pick_action(state, info)
                    next_state, opponent_reward, terminal, _, info = env.step(opponent_action)
                    
                    opponent_total_reward += opponent_reward
                    state = next_state
                
                # Train DQN when buffer has enough samples
                if len(dqn_agent.replay_buffer) > batch_size:
                    dqn_agent.train(batch_size)
                
                # Break if episode is done
                if terminal:
                    win = dqn_total_reward > opponent_total_reward
                    win_status = "WON" if win else "LOST"
                    print(f'Episode {ep+1} terminated. DQN {win_status} with reward {dqn_total_reward:.2f} vs opponent {opponent_total_reward:.2f}')
                    break
            
            # Update epsilon
            if dqn_agent.epsilon > dqn_agent.epsilon_min:
                dqn_agent.epsilon *= dqn_agent.epsilon_decay
            
            # Update logger
            logger.update_episode_data(
                episode=ep,
                dqn_reward=dqn_total_reward,
                opponent_reward=opponent_total_reward,
                epsilon=dqn_agent.epsilon,
                actions_taken=dqn_actions_taken,
                win=dqn_total_reward > opponent_total_reward
            )
            logger.update_time_step(time_step)
            
            # Print episode stats
            elapsed = time.time() - episode_start
            print(f'Time elapsed: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)')
            print(f'Win rate (last 100 episodes): {logger.get_win_rate():.2f}')
            
            # Save checkpoint every 10 episodes
            if (ep + 1) % 10 == 0:
                dqn_agent.save_agent(model_name)
                logger.save()
                print(f"Checkpoint saved at episode {ep+1}")
            
            # Early stopping if agent performs well
            if logger.get_win_rate() > 0.95 and ep >= 100:
                print('Training stopped due to high win rate over last 100 episodes')
                break
    
    except KeyboardInterrupt:
        print("\nTraining interrupted manually. Saving checkpoint...")
    
    finally:
        # Save final model and metadata
        dqn_agent.save_agent(model_name)
        logger.save()
        env.close()
        print("Training finished. Environment closed.")


if __name__ == '__main__':
    # Default training configuration
    train_dqn(
        num_episodes=1000,
        continue_training=False,
        model_name='century_dqn_v1',
        opponent_type='random'
    ) 