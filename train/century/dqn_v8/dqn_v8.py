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
        self.gamma = kwargs.get('gamma', 0.99)  # Increased from 0.98 to improve long-term planning
        self.epsilon = kwargs.get('epsilon', 1.0)
        self.epsilon_min = kwargs.get('epsilon_min', 0.05)
        self.epsilon_decay = kwargs.get('epsilon_decay', 0.997)
        self.learning_rate = kwargs.get('learning_rate', 0.0003)  # Decreased learning rate for more stability
        self.update_rate = kwargs.get('update_rate', 150)  # Decreased update rate for more stable target network
        self.batch_size = kwargs.get('batch_size', 256)  # Increased batch size for better gradient estimates
        self.replay_buffer_size = kwargs.get('replay_buffer_size', 100000)  # Increased buffer size for more diverse experiences
        self.num_timesteps = kwargs.get('num_timesteps', 3000)  # Increased max timesteps for longer games
        self.checkpoint_freq = kwargs.get('checkpoint_freq', 100)
        self.model_save_freq = kwargs.get('model_save_freq', 100)


class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 768)  # Increased first layer size
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(768, 512)  # Increased second layer size
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(512, 256)
        self.dropout3 = nn.Dropout(0.2)  # Added a third dropout layer
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
        x = self.dropout3(x)  # Apply the new dropout layer
        return self.fc4(x)


class DQNAgent:
    def __init__(self, state_size, action_size, config):
        self.state_size = state_size
        self.action_size = action_size

        # Initialize Replay Buffer
        self.replay_buffer = deque(maxlen=config.replay_buffer_size)
        
        # Track rewards for learning rate scheduling
        self.rewards = []

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
        self.optimizer = optim.Adam(self.main_network.parameters(), lr=self.learning_rate, weight_decay=1e-5)  # Added weight decay for regularization
        
        # Add learning rate scheduler - more responsive to win rate changes
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='max',  # We want to maximize win rate
            factor=0.7,  # Reduced factor to 0.7 for more gradual reductions (was 0.5)
            patience=30, # Reduced patience to be more responsive
            threshold=0.01, # Smaller threshold to detect smaller changes
            verbose=True, 
            min_lr=5e-6   # Lower min_lr for potential better convergence
        )
        
        # Track win status for more stable win rate calculation
        self.recent_wins = deque(maxlen=100)
        
        # Track card efficiency metrics
        self.cards_per_golem = deque(maxlen=100)  # Track how many merchant cards were used per golem
        self.turns_per_golem = deque(maxlen=100)  # Track how many turns it took to get each golem
        self.optimal_cards_estimate = 9  # Initial estimate of optimal card count, increased for YGBP

    def update_target_network(self):
        """Method to set the Main NN's weights on the Target NN"""
        self.target_network.load_state_dict(self.main_network.state_dict())

    def save_experience(self, state, action, reward, next_state, terminal):
        # Track metrics for card efficiency
        merchant_cards = state[14:59]  # Use correct slice for v15 (45 merchant cards)
        owned_cards = sum(1 for card_status in merchant_cards if card_status > 0)
        
        # Track progression in the game
        golem_count_before = state[59]  # Index for player1_golem_count in v15
        golem_count_after = next_state[59]  # Index for player1_golem_count after action
        
        # Set higher priority for:
        # 1. Terminal states (game endings)
        # 2. High rewards (likely golem acquisitions)
        # 3. Actions that are rare in the buffer
        # 4. Efficient play patterns
        
        priority = 1.0
        
        # Terminal states
        if terminal:
            priority *= 3.0
            
            # If this is a win, calculate final efficiency metrics
            if reward > 0:  # Winning reward is positive
                # Update the optimal card estimate based on this game's performance
                # This creates a learning mechanism for optimal card count
                if hasattr(self, 'episode_card_count') and self.episode_card_count > 0:
                    self.cards_per_golem.append(self.episode_card_count / max(1, golem_count_before))
                    # Recalculate optimal card estimate
                    if len(self.cards_per_golem) >= 10:  # Wait until we have enough data
                        # Get the 25th percentile of successful games as our target
                        # This biases toward more efficient games
                        card_counts = sorted(list(self.cards_per_golem))
                        idx = max(0, int(len(card_counts) * 0.25) - 1)
                        self.optimal_cards_estimate = max(4, min(12, card_counts[idx] * 2))  # Increased max to 12 for YGBP
                        print(f"Updated optimal card estimate: {self.optimal_cards_estimate}")
        
        # High rewards
        if reward > 5.0:
            priority *= 2.0
        
        # Check action type
        if action >= 89 and action <= 124:  # Golem acquisition actions (getG1-getG36)
            # Track merchant card count at time of golem acquisition for efficiency metrics
            if hasattr(self, 'episode_card_count'):
                self.cards_per_golem.append(owned_cards)
            
            # Heavily prioritize golem actions
            priority *= 5.0
            
            # Add bonus reward for efficient golem acquisition
            if owned_cards <= self.optimal_cards_estimate:
                efficiency_bonus = max(0, (self.optimal_cards_estimate - owned_cards)) * 0.5
                reward += efficiency_bonus  # Bonus for acquiring golem with fewer cards
        
        # Apply merchant card hoarding penalty
        # Actions 1-43 correspond to getM3 through getM45
        if action >= 1 and action <= 43:
            # Count how many merchant cards the agent already has (excluding the starting M1, M2)
            # In the state, merchant cards status is 0 (not owned), 1 (owned but unplayable), 2 (owned and playable)
            merchant_cards = state[14:59]  # This gets the status of all 45 merchant cards
            owned_cards = sum(1 for card_status in merchant_cards if card_status > 0)
            
            # Determine game phase based on golem count to adjust penalties
            golem_count = state[59]  # Player's golem count
            early_game = golem_count < 1
            mid_game = 1 <= golem_count < 3
            late_game = golem_count >= 3
            
            # Calculate penalty threshold based on game phase - adjusted for YGBP
            if early_game:
                penalty_threshold = 6  # Allow more cards in early game
            elif mid_game:
                penalty_threshold = 5  # Tighten in mid game
            else:
                penalty_threshold = 4  # Be strict in late game
            
            # Apply diminishing returns - reduce reward based on cards already owned
            # Start reducing reward after the threshold
            if owned_cards > penalty_threshold:
                # Progressive penalty based on game phase and card count
                base_penalty = 0.3  # Base penalty factor
                if mid_game:
                    base_penalty = 0.4
                elif late_game:
                    base_penalty = 0.5
                
                # Exponential penalty factor that grows as more cards are acquired
                excess = owned_cards - penalty_threshold
                penalty_factor = 1.0 - min(0.95, base_penalty * (excess ** 1.5))
                
                # Modify the actual experience's reward value (not just the priority)
                reward = reward * penalty_factor
                
                # Also reduce the priority of collecting more cards when we already have many
                priority *= penalty_factor
                
                # For severe hoarding, apply a direct negative reward component
                if owned_cards > penalty_threshold + 3:
                    direct_penalty = -0.2 * (owned_cards - (penalty_threshold + 3))
                    reward += direct_penalty  # Direct penalty for excessive hoarding
        
        # For card usage actions, give bonus for using cards when we have many
        if action >= 44 and action <= 88:  # Card usage actions (useM1-useM45)
            # Bonus for using cards when we have many
            if owned_cards > 5:
                usage_bonus = min(1.0, 0.1 * (owned_cards - 5))
                reward += usage_bonus
                priority *= (1.0 + usage_bonus)  # Also boost priority
                
            # Special handling for upgrade cards (useM2 = action 45 and useM11 = action 54)
            if action == 45 or action == 54:
                # Extract crystal quantities from state
                yellow = state[8]  # Player's yellow crystal count
                green = state[9]   # Player's green crystal count
                blue = state[10]   # Player's blue crystal count
                pink = state[11]   # Player's pink crystal count
                
                # Encourage upgrade strategy that maximizes value
                # In YGBP, the most efficient upgrades are:
                # - Yellow -> Green -> Blue -> Pink
                if yellow > 0 or green > 0 or blue > 0:
                    # Prioritize upgrading lower crystals first
                    if yellow > 0:
                        upgrade_bonus = 0.3  # Bonus for upgrading yellow
                    elif green > 0:
                        upgrade_bonus = 0.2  # Bonus for upgrading green
                    elif blue > 0:
                        upgrade_bonus = 0.1  # Bonus for upgrading blue
                    
                    reward += upgrade_bonus
                    priority *= (1.0 + upgrade_bonus)
        
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
                
                # Apply merchant card acquisition penalty
                # Actions 1-43 correspond to getM3 through getM45
                merchant_cards = state[14:59]  # Use correct slice for v15
                owned_cards = sum(1 for card_status in merchant_cards if card_status > 0)
                
                # Extract player progress indicators to determine game phase
                player_golem_count = state[59]  # Index for player1_golem_count
                player_points = state[60]  # Index for player1_points
                
                # Extract crystal quantities 
                yellow = state[8]  # Player's yellow crystal count
                green = state[9]   # Player's green crystal count
                blue = state[10]   # Player's blue crystal count
                pink = state[11]   # Player's pink crystal count
                
                # Calculate total crystals for crystal management
                total_crystals = yellow + green + blue + pink
                
                # Determine game phase based on golem count and points
                early_game = player_golem_count < 1
                mid_game = 1 <= player_golem_count < 3
                late_game = player_golem_count >= 3
                
                # Calculate dynamic penalty threshold based on game phase - adjusted for YGBP
                if early_game:
                    penalty_threshold = 7  # Allow more cards in early game for YGBP
                elif mid_game:
                    penalty_threshold = 6  # Tighten in mid game
                else:
                    penalty_threshold = 5  # Be strict in late game
                
                # Adjust penalties based on golem progress and card efficiency
                optimal_card_count = self.optimal_cards_estimate
                
                # If agent has many cards, reduce Q-values for getting more merchant cards
                if owned_cards > penalty_threshold:
                    # Progressive penalty based on how far above optimal we are
                    excess_ratio = (owned_cards - penalty_threshold) / 5  # Normalized ratio
                    
                    # Cubic penalty that grows much faster than quadratic
                    penalty_base = (owned_cards - penalty_threshold) ** 3 * 0.3
                    
                    # Scale penalty based on game phase
                    phase_multiplier = 1.0
                    if mid_game:
                        phase_multiplier = 1.5  # Stronger penalty in mid-game
                    elif late_game:
                        phase_multiplier = 2.0  # Even stronger in late-game
                    
                    penalty = penalty_base * phase_multiplier
                    
                    # Apply penalty to getM3-getM45 actions (indices 1-43)
                    for action_idx in range(1, 44):
                        if valid_actions[action_idx] == 1:  # Only penalize valid actions
                            q_values[action_idx] -= penalty
                    
                    # Also boost Q-values for card usage actions (playM)
                    # Actions 44-88 correspond to useM1 through useM45
                    # Progressive boost based on excess cards
                    boost_base = min(10.0, owned_cards * 0.8)  # Increased boost
                    
                    # Scale boost based on game phase
                    if mid_game or late_game:
                        boost_base *= 1.5  # Stronger boost in later phases
                    
                    # Apply boost to all useM actions
                    for action_idx in range(44, 89):
                        if valid_actions[action_idx] == 1:  # Only boost valid actions
                            q_values[action_idx] += boost_base
                
                # Special boost for golem acquisition actions when possible
                # Actions 89-124 correspond to getG1-getG36
                golem_boost = 5.0 + player_golem_count * 2.0  # Progressive boost for golems
                for action_idx in range(89, 125):
                    if valid_actions[action_idx] == 1:  # Only boost valid golem actions
                        q_values[action_idx] += golem_boost
                        
                        # Additional boost for high-value golems (generally later in the enum)
                        # Later golems (higher indexes) often require pink crystals and are more valuable
                        late_golem_bonus = (action_idx - 89) * 0.1  # Small bonus that increases with golem index
                        q_values[action_idx] += late_golem_bonus
                
                # Add special handling for crystal management and strategy
                if total_crystals >= 8:  # Getting close to crystal limit (10)
                    # Boost actions that help manage crystals more effectively
                    
                    # 1. Encourage upgrade actions more (useM2 = 45, useM11 = 54)
                    if valid_actions[45] == 1:  # useM2 (2-level upgrade)
                        q_values[45] += 3.0  # Significant boost when near crystal limit
                    if valid_actions[54] == 1:  # useM11 (3-level upgrade)
                        q_values[54] += 4.0  # Even stronger boost for 3-level upgrade
                    
                    # 2. Boost golem acquisition actions more when near crystal limit
                    for action_idx in range(89, 125):
                        if valid_actions[action_idx] == 1:
                            q_values[action_idx] += 2.0  # Additional boost when crystals are full
                
                # Pink crystal strategy - encourage using cards that give pink crystals
                # These are generally later-added merchant cards (higher indices in useM actions)
                if pink < 1 and mid_game:  # Prioritize pink in mid-game when ready
                    # Boost actions that can gain pink crystals (various trade cards)
                    # Focusing on specific indices from actions 44-88 that relate to pink-gaining cards
                    pink_related_actions = [69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 88]  # Example indices
                    for action_idx in pink_related_actions:
                        if action_idx < len(valid_actions) and valid_actions[action_idx] == 1:
                            q_values[action_idx] += 2.0  # Boost for pink-generating actions
                
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
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.main_network.parameters(), 1.0)
        
        self.optimizer.step()

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
            'cards_per_golem': list(self.cards_per_golem),
            'turns_per_golem': list(self.turns_per_golem),
            'optimal_cards_estimate': self.optimal_cards_estimate
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
        
        # Load model checkpoint with weights_only=False since we trust the source
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
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
        
        # Load card efficiency metrics if available
        self.cards_per_golem = deque(checkpoint.get('cards_per_golem', []))
        self.turns_per_golem = deque(checkpoint.get('turns_per_golem', []))
        self.optimal_cards_estimate = checkpoint.get('optimal_cards_estimate', 9)
        
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
    
    # Track efficiency metrics
    dqn_agent.episode_turns = []
    
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
            
            # Track turns taken in this episode
            turns_taken = 0
            
            # Reset episode-specific trackers
            dqn_agent.episode_card_count = 0

            print(f'\nTraining on EPISODE {ep+1} with epsilon {dqn_agent.epsilon}')
            start = time.time()

            for t in range(num_timesteps):
                time_step += 1

                # Update Target Network every update_rate timesteps
                if time_step % dqn_agent.update_rate == 0:
                    dqn_agent.update_target_network()

                if info['current_player'] == 0:
                    turns_taken += 1
                    action = dqn_agent.pick_epsilon_greedy_action(state, info)
                    
                    # Track merchant card acquisitions (actions 1-43 are getM3-getM45)
                    if action >= 1 and action <= 43:
                        merchant_cards_acquired += 1
                        dqn_agent.episode_card_count += 1
                        
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
                    
                    # Track efficiency metrics
                    dqn_agent.episode_turns.append(turns_taken)
                    if len(dqn_agent.episode_turns) > 10:
                        avg_turns = sum(dqn_agent.episode_turns[-10:]) / 10
                        print(f"Average turns per game (last 10): {avg_turns:.1f}")
                    
                    # If this was a win, update turns-per-golem metric
                    if dqn_total_reward > 0:  # Winning games have positive reward
                        golem_count = next_state[59] if info['current_player'] == 0 else next_state[117]  # Adjust index based on player
                        if golem_count > 0:
                            dqn_agent.turns_per_golem.append(turns_taken / golem_count)
                            if len(dqn_agent.turns_per_golem) > 10:
                                avg_turns_per_golem = sum(list(dqn_agent.turns_per_golem)[-10:]) / min(10, len(dqn_agent.turns_per_golem))
                                print(f"Average turns per golem (last 10 wins): {avg_turns_per_golem:.1f}")
                    
                    break

                # Train the Main NN when ReplayBuffer has enough experiences
                if len(dqn_agent.replay_buffer) > batch_size:
                    dqn_agent.train(batch_size)

            dqn_agent.rewards.append(dqn_total_reward)
            dqn_agent.epsilon_values.append(dqn_agent.epsilon)
            dqn_agent.merchant_card_counts.append(merchant_cards_acquired)
            
            # Print merchant card acquisition stats periodically
            if ep % 10 == 0:
                avg_cards = sum(dqn_agent.merchant_card_counts[-10:]) / min(10, len(dqn_agent.merchant_card_counts[-10:]))
                print(f"Avg merchant cards acquired (last 10 ep): {avg_cards:.2f}")
                
                # Calculate card efficiency ratio (cards per win)
                if sum(1 for i in range(max(0, ep-10), ep) if dqn_agent.rewards[i] > 0) > 0:
                    # Get cards acquired in winning episodes only
                    win_indices = [i for i in range(max(0, ep-10), ep) if dqn_agent.rewards[i] > 0]
                    cards_in_wins = sum(dqn_agent.merchant_card_counts[i] for i in win_indices) / len(win_indices)
                    print(f"Avg cards in winning games (last 10 wins): {cards_in_wins:.2f}")

            # Update Epsilon value with adaptive mechanism
            if ep % 50 == 0 and ep > 0:
                # Calculate win rate over last 50 episodes
                recent_wins = sum(1 for i in range(max(0, ep-50), ep) if dqn_agent.rewards[i] > 0)
                win_rate = recent_wins / min(50, ep)
                
                # Calculate card efficiency metric
                avg_cards = sum(dqn_agent.merchant_card_counts[-50:]) / min(50, len(dqn_agent.merchant_card_counts[-50:]))
                
                # Adaptive epsilon adjustment based on both win rate and card efficiency
                if win_rate < 0.4:
                    # If win rate is low, increase exploration
                    dqn_agent.epsilon = min(0.7, dqn_agent.epsilon + 0.2)
                    print(f"Low win rate detected, increasing epsilon to {dqn_agent.epsilon}")
                elif avg_cards > 12 and win_rate > 0.6:  # Adjusted threshold for more cards in v15
                    # If winning but using too many cards, increase exploration to find more efficient strategies
                    dqn_agent.epsilon = min(0.3, dqn_agent.epsilon + 0.1)
                    print(f"Card efficiency issue detected, increasing epsilon to {dqn_agent.epsilon}")
                # Otherwise, keep decreasing normally
                elif dqn_agent.epsilon > dqn_agent.epsilon_min:
                    dqn_agent.epsilon *= dqn_agent.epsilon_decay

            # Print episode info
            elapsed = time.time() - start
            print(f'Time elapsed during EPISODE {ep+1}: {elapsed} seconds = {round(elapsed/60, 3)} minutes')
            
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
    
    args = parser.parse_args()
    
    # Safety check: prevent accidental training from scratch
    if args.checkpoint is None and not args.reset:
        print("Error: To train from scratch, please use the --reset flag.")
        print("To resume from a checkpoint, use the --checkpoint <path> argument.")
        exit(1)
        
    config_dict = {
        'episodes': args.episodes if args.episodes is not None else 1000,  # Increased to 10000 for more training
        'checkpoint': args.checkpoint if args.checkpoint is not None else None,
        'checkpoint_freq': args.checkpoint_freq if args.checkpoint_freq is not None else 100,
        'model_save_freq': args.model_save_freq if args.model_save_freq is not None else 100,
        'gamma': args.gamma if args.gamma is not None else 0.99,
        'epsilon': args.epsilon if args.epsilon is not None else 1.0,
        'epsilon_decay': args.epsilon_decay if args.epsilon_decay is not None else 0.998,
        'epsilon_min': args.epsilon_min if args.epsilon_min is not None else 0.05,
        'learning_rate': args.learning_rate if args.learning_rate is not None else 0.0003,
        'update_rate': args.update_rate if args.update_rate is not None else 150,
        'batch_size': args.batch_size if args.batch_size is not None else 256,
        'replay_buffer_size': args.replay_buffer_size if args.replay_buffer_size is not None else 100000,
        'num_timesteps': args.num_timesteps if args.num_timesteps is not None else 3000,
    }

    config = DQNConfig(**config_dict)
    train_dqn(config, config_dict['episodes'], config_dict['checkpoint'])