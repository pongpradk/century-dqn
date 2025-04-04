import random
import time
import os
import pickle
import glob
import argparse
from collections import deque

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
        self.update_rate = kwargs.get('update_rate', 100)
        self.batch_size = kwargs.get('batch_size', 64)
        self.replay_buffer_size = kwargs.get('replay_buffer_size', 10000)
        self.num_timesteps = kwargs.get('num_timesteps', 2000)
        self.checkpoint_freq = kwargs.get('checkpoint_freq', 10)
        self.model_save_freq = kwargs.get('model_save_freq', 50)


class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 256)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(256, 256)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, action_size)
        
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


class RND(nn.Module):
    """Random Network Distillation for intrinsic motivation"""
    def __init__(self, state_size):
        super(RND, self).__init__()
        # Target network (fixed random weights)
        self.target = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        
        # Predictor network (trained to match target)
        self.predictor = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        
        # Initialize target network with random weights and freeze it
        for param in self.target.parameters():
            param.requires_grad = False
    
    def forward(self, state):
        target_feature = self.target(state)
        predict_feature = self.predictor(state)
        
        # Intrinsic reward is prediction error
        intrinsic_reward = F.mse_loss(predict_feature, target_feature, reduction='none').sum(dim=1)
        return intrinsic_reward


class DQNAgent:
    def __init__(self, state_size, action_size, config):
        self.state_size = state_size
        self.action_size = action_size

        # Initialize Replay Buffer
        self.replay_buffer = deque(maxlen=config.replay_buffer_size)

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

        # Add RND for intrinsic motivation
        self.rnd = RND(state_size).to(self.device)
        self.rnd_optimizer = optim.Adam(self.rnd.predictor.parameters(), lr=config.learning_rate)
        self.intrinsic_reward_scale = 0.1  # Scale factor for intrinsic rewards

    def update_target_network(self):
        """Method to set the Main NN's weights on the Target NN"""
        self.target_network.load_state_dict(self.main_network.state_dict())

    def save_experience(self, state, action, reward, next_state, terminal):
        # Calculate intrinsic reward
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            intrinsic_reward = self.rnd(state_tensor).item()
        
        # Scale and clip intrinsic reward
        intrinsic_reward = min(1.0, self.intrinsic_reward_scale * intrinsic_reward)
        
        # Combine with extrinsic reward
        combined_reward = reward + intrinsic_reward
        
        # Set higher priority for:
        # 1. Terminal states (game endings)
        # 2. High rewards (likely golem acquisitions)
        # 3. Actions that are rare in the buffer
        
        priority = 1.0
        
        # Terminal states
        if terminal:
            priority *= 3.0
        
        # High rewards
        if combined_reward > 5.0:
            priority *= 2.0
        
        # Check action type
        if action >= 19 and action <= 23:  # Golem acquisition actions
            priority *= 4.0  # Heavily prioritize golem actions
        
        # Add action count tracking if it doesn't exist
        if not hasattr(self, 'action_counts'):
            self.action_counts = np.ones(self.action_size)
        
        # Update action count
        self.action_counts[action] += 1
        
        # Prioritize rare actions
        rarity_factor = np.sum(self.action_counts) / (self.action_counts[action] * self.action_size)
        priority *= min(3.0, rarity_factor)  # Cap at 3x priority
        
        self.replay_buffer.append((state, action, combined_reward, intrinsic_reward, next_state, terminal, priority))

    def sample_experience_batch(self, batch_size):
        # Sample based on priority
        priorities = np.array([exp[6] for exp in self.replay_buffer])
        probs = priorities / np.sum(priorities)
        indices = np.random.choice(len(self.replay_buffer), batch_size, p=probs)
        exp_batch = [self.replay_buffer[idx] for idx in indices]

        # Create tensors for each component
        state_batch = torch.FloatTensor(np.array([batch[0] for batch in exp_batch])).to(self.device)
        action_batch = torch.LongTensor(np.array([batch[1] for batch in exp_batch])).to(self.device)
        reward_batch = torch.FloatTensor(np.array([batch[2] for batch in exp_batch])).to(self.device)
        intrinsic_reward_batch = torch.FloatTensor(np.array([batch[3] for batch in exp_batch])).to(self.device)
        next_state_batch = torch.FloatTensor(np.array([batch[4] for batch in exp_batch])).to(self.device)
        terminal_batch = torch.FloatTensor(np.array([batch[5] for batch in exp_batch])).to(self.device)

        return state_batch, action_batch, reward_batch, intrinsic_reward_batch, next_state_batch, terminal_batch

    def pick_epsilon_greedy_action(self, state, info):
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

    def train(self, batch_size):
        # Sample a batch of experiences
        state_batch, action_batch, reward_batch, intrinsic_reward_batch, next_state_batch, terminal_batch = self.sample_experience_batch(batch_size)

        # Create a mask for valid actions in the next state
        # This would require storing valid_actions in replay buffer
        valid_action_mask = torch.zeros((batch_size, self.action_size), device=self.device)
        
        # Get next Q values from target network
        with torch.no_grad():
            next_q_values = self.target_network(next_state_batch)
            # Apply valid action masking for next state
            next_q_values[~valid_action_mask.bool()] = -1e9
            max_next_q = next_q_values.max(1)[0]
            target_q_values = reward_batch + intrinsic_reward_batch + (1 - terminal_batch) * self.gamma * max_next_q

        # Get current Q values
        current_q_values = self.main_network(state_batch).gather(1, action_batch.unsqueeze(1))

        # Compute loss and update weights
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Also train the RND predictor network
        self.rnd_optimizer.zero_grad()
        intrinsic_loss = self.rnd(state_batch).mean()
        intrinsic_loss.backward()
        self.rnd_optimizer.step()

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
            'epsilon_values': self.epsilon_values
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
        
        
def preprocess_state(state_dict):
    """Create a more informative state representation"""
    # Extract existing information
    processed_state = []
    
    # Add merchant market information
    processed_state.extend(state_dict["merchant_market"])
    
    # Add golem market information
    processed_state.extend(state_dict["golem_market"])
    
    # Add player caravan info
    for crystal in ["yellow", "green"]:
        processed_state.append(state_dict["player1_caravan"][crystal][0])
    
    # Add merchant card ownership
    processed_state.extend(state_dict["player1_merchant_cards"])
    
    # Add golem count and points
    processed_state.append(state_dict["player1_golem_count"][0])
    processed_state.append(state_dict["player1_points"][0])
    
    # Add opponent info
    for crystal in ["yellow", "green"]:
        processed_state.append(state_dict["player2_caravan"][crystal][0])
    processed_state.extend(state_dict["player2_merchant_cards"])
    processed_state.append(state_dict["player2_golem_count"][0])
    processed_state.append(state_dict["player2_points"][0])
    
    # Add derived features
    
    # 1. Resource efficiency metrics
    p1_yellow = state_dict["player1_caravan"]["yellow"][0]
    p1_green = state_dict["player1_caravan"]["green"][0]
    total_crystals = p1_yellow + p1_green
    
    # Crystal ratio (how balanced are resources)
    crystal_ratio = 0.5
    if total_crystals > 0:
        crystal_ratio = p1_green / total_crystals
    processed_state.append(crystal_ratio)
    
    # Crystal utilization (how close to capacity)
    crystal_utilization = total_crystals / 10  # Normalized by max capacity
    processed_state.append(crystal_utilization)
    
    # 2. Golem acquisition readiness
    # For each golem in market, how close are we to acquiring it?
    for i, golem_id in enumerate(state_dict["golem_market"]):
        if golem_id >= 5:  # Invalid golem
            readiness = 0
        else:
            # Simplified readiness metric
            golem_costs = {
                1: {"yellow": 2, "green": 2},  # G1-Y2G2
                2: {"yellow": 3, "green": 2},  # G2-Y3G2
                3: {"yellow": 2, "green": 3},  # G3-Y2G3
                4: {"yellow": 0, "green": 4},  # G4-G4
                5: {"yellow": 0, "green": 5}   # G5-G5
            }
            if golem_id in golem_costs:
                cost = golem_costs[golem_id]
                yellow_needed = max(0, cost["yellow"] - p1_yellow)
                green_needed = max(0, cost["green"] - p1_green)
                
                # Readiness is inverse of total needed (normalized)
                total_needed = yellow_needed + green_needed
                readiness = 1.0 if total_needed == 0 else 1.0 / (1.0 + total_needed)
            else:
                readiness = 0
        processed_state.append(readiness)
    
    return np.array(processed_state, dtype=np.float32)

def train_dqn(config, num_episodes, checkpoint_path=None):
    env = gym.make("gymnasium_env/CenturyGolem-v12")
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
    
    # Start episode counter
    start_episode = 0
    
    # Load checkpoint if provided
    if checkpoint_path:
        buffer_path = checkpoint_path.replace("checkpoint", "replay_buffer")
        start_episode = dqn_agent.load_checkpoint(checkpoint_path, buffer_path)
        print(f"Resuming training from episode {start_episode}")

    try:
        # Modify state processing
        def process_observation(obs):
            return preprocess_state(obs)
        
        for ep in range(start_episode, num_episodes):
            dqn_total_reward = 0
            opponent_total_reward = 0
            
            state, info = env.reset()
            state = process_observation(state)

            print(f'\nTraining on EPISODE {ep+1} with epsilon {dqn_agent.epsilon}')
            start = time.time()

            for t in range(num_timesteps):
                time_step += 1

                # Update Target Network every update_rate timesteps
                if time_step % dqn_agent.update_rate == 0:
                    dqn_agent.update_target_network()

                if info['current_player'] == 0:
                    action = dqn_agent.pick_epsilon_greedy_action(state, info)
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

                # Update state
                state = process_observation(next_state)

            dqn_agent.rewards.append(dqn_total_reward)
            dqn_agent.epsilon_values.append(dqn_agent.epsilon)

            # Update Epsilon value
            if ep % 50 == 0 and ep > 0:
                # Calculate win rate over last 50 episodes
                recent_wins = sum(1 for i in range(max(0, ep-50), ep) if dqn_agent.rewards[i] > 0)
                win_rate = recent_wins / min(50, ep)
                
                # If win rate is low, increase exploration
                if win_rate < 0.4:
                    dqn_agent.epsilon = min(0.7, dqn_agent.epsilon + 0.2)
                    print(f"Low win rate detected, increasing epsilon to {dqn_agent.epsilon}")
                # Otherwise, keep decreasing normally
                elif dqn_agent.epsilon > dqn_agent.epsilon_min:
                    dqn_agent.epsilon *= dqn_agent.epsilon_decay

            # Print episode info
            elapsed = time.time() - start
            print(f'Time elapsed during EPISODE {ep+1}: {elapsed} seconds = {round(elapsed/60, 3)} minutes')
            
            # Save checkpoint every checkpoint_freq episodes
            if (ep + 1) % config.checkpoint_freq == 0:
                dqn_agent.save_checkpoint(ep + 1)
            
            # Save model every model_save_freq episodes
            if (ep + 1) % config.model_save_freq == 0:
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
    
    config_dict = {
        'episodes': args.episodes if args.episodes is not None else 3000,
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
    }

    config = DQNConfig(**config_dict)
    train_dqn(config, config_dict['episodes'], config_dict['checkpoint'])