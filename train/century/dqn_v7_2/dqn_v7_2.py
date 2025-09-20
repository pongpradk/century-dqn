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
        self.update_rate = kwargs.get('update_rate', 200)
        self.batch_size = kwargs.get('batch_size', 128)
        self.replay_buffer_size = kwargs.get('replay_buffer_size', 50000)
        self.num_timesteps = kwargs.get('num_timesteps', 2000)
        self.checkpoint_freq = kwargs.get('checkpoint_freq', 100)
        self.model_save_freq = kwargs.get('model_save_freq', 100)


class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 512)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(512, 512)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, action_size)
        
        # Add an auxiliary head to predict estimated turns to victory
        self.aux_fc1 = nn.Linear(256, 128)
        self.aux_fc2 = nn.Linear(128, 1)  # Predicts turns to victory
        
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
        features = F.relu(self.fc3(x))
        
        # Main output - Q-values
        q_values = self.fc4(features)
        
        # Auxiliary output - estimated turns to victory
        aux_out = F.relu(self.aux_fc1(features))
        turns_estimate = self.aux_fc2(aux_out)
        
        return q_values, turns_estimate


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
        
        # Additional hyperparameters
        self.efficiency_weight = 0.01  # Weight for the auxiliary task
        self.max_desired_cards = 8  # Maximum desired merchant cards (including starting cards)
        self.hard_limit_cards = 10  # Hard limit for merchant cards after which severe penalties apply
        
        # Card value assessment weights - higher values are more desirable cards
        self.card_values = {
            # Crystal generation cards (yellow is less valuable, green/blue more valuable)
            3: 1.0,   # M3: 3Y
            4: 1.2,   # M4: 4Y
            5: 1.5,   # M5: 1Y1G
            6: 1.8,   # M6: 2Y1G 
            7: 2.0,   # M7: 2G
            12: 3.0,  # M12: 1B
            13: 3.0,  # M13: 1Y1B
            
            # Trade cards - prioritize those that convert to higher value crystals
            8: 1.3,   # M8: 1G:3Y (medium value - good for yellow accumulation)
            9: 1.8,   # M9: 2Y:2G (high value - converts yellow to green)
            10: 2.0,  # M10: 3Y:3G (high value - mass yellow to green conversion)
            14: 2.5,  # M14: 2Y:1B (high value - yellow to blue)
            15: 2.8,  # M15: 3Y:1G1B (very high value - yellow to green+blue)
            16: 3.0,  # M16: 4Y:2B (very high value - mass yellow to blue)
            17: 3.2,  # M17: 5Y:3B (extremely high value - mass yellow to blue)
            18: 2.5,  # M18: 2G:2B (high value - green to blue)
            19: 2.2,  # M19: 2G:3Y1B (good value - adds diversity)
            20: 2.5,  # M20: 3G:2Y2B (high value - green to yellow+blue mix)
            21: 3.0,  # M21: 3G:3B (very high value - mass green to blue)
            22: 1.8,  # M22: 1B:2G (good value - blue to green, but expensive)
            23: 2.0,  # M23: 1B:4Y1G (high value - blue to yellow+green mix)
            24: 2.0,  # M24: 1B:1Y2G (high value - blue to yellow+green)
            25: 2.5,  # M25: 2B:2Y3G (very high value - blue to mixed crystals)
            
            # Upgrade cards
            11: 2.5,  # M11: 3U (very high value - efficient upgrading)
        }

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
        
        # Track card efficiency metrics
        self.cards_per_golem = deque(maxlen=100)  # Track how many merchant cards were used per golem
        self.turns_per_golem = deque(maxlen=100)  # Track how many turns it took to get each golem
        self.optimal_cards_estimate = 6  # Lower initial estimate of optimal card count
        
        # Track average turns to win for games
        self.turns_to_win = deque(maxlen=100)

    def update_target_network(self):
        """Method to set the Main NN's weights on the Target NN"""
        self.target_network.load_state_dict(self.main_network.state_dict())

    def save_experience(self, state, action, reward, next_state, terminal, turns_taken=None):
        # Track metrics for card efficiency
        merchant_cards = state[14:39]  # Use correct slice for v13
        owned_cards = sum(1 for card_status in merchant_cards if card_status > 0)
        
        # Track progression in the game
        golem_count_before = state[39]  # Index for player1_golem_count
        golem_count_after = next_state[39]  # Index for player1_golem_count after action
        
        # Determine game phase
        early_game = golem_count_before < 1
        mid_game = 1 <= golem_count_before < 3
        late_game = golem_count_before >= 3
        
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
                # Track the number of turns it took to win
                if turns_taken is not None:
                    self.turns_to_win.append(turns_taken)
                
                # Update the optimal card estimate based on this game's performance
                if hasattr(self, 'episode_card_count') and self.episode_card_count > 0:
                    self.cards_per_golem.append(self.episode_card_count / max(1, golem_count_before))
                    # Recalculate optimal card estimate
                    if len(self.cards_per_golem) >= 10:  # Wait until we have enough data
                        # Get the 20th percentile of successful games as our target
                        # This biases toward more efficient games
                        card_counts = sorted(list(self.cards_per_golem))
                        idx = max(0, int(len(card_counts) * 0.2) - 1)
                        self.optimal_cards_estimate = max(4, min(8, card_counts[idx] * 2))  # *2 because we count cards per golem
                        print(f"Updated optimal card estimate: {self.optimal_cards_estimate}")
        
        # High rewards
        if reward > 5.0:
            priority *= 2.0
        
        # Check action type
        if action >= 49 and action <= 62:  # Golem acquisition actions (getG1-getG14)
            # Track merchant card count at time of golem acquisition for efficiency metrics
            if hasattr(self, 'episode_card_count'):
                self.cards_per_golem.append(owned_cards)
            
            # Heavily prioritize golem actions
            priority *= 5.0
            
            # Add bonus reward for efficient golem acquisition
            if owned_cards <= self.optimal_cards_estimate:
                efficiency_bonus = max(0, (self.optimal_cards_estimate - owned_cards)) * 1.0
                reward += efficiency_bonus  # Stronger bonus for acquiring golem with fewer cards
        
        # Apply merchant card hoarding penalty
        # Actions 1-23 correspond to getM3 through getM25
        if action >= 1 and action <= 23:
            # Count how many merchant cards the agent already has
            merchant_cards = state[14:39]
            owned_cards = sum(1 for card_status in merchant_cards if card_status > 0)
            
            # Calculate penalty threshold based on game phase (more forgiving in early game)
            if early_game:
                penalty_threshold = 5  # More lenient in early game
            elif mid_game:
                penalty_threshold = 3  # Tighter in mid game
            else:
                penalty_threshold = 2  # Very strict in late game
            
            # Apply diminishing returns - reduce reward based on cards already owned
            if owned_cards > penalty_threshold:
                # Progressive penalty based on game phase and card count
                base_penalty = 0.4 if early_game else 0.5  # Less severe in early game
                if mid_game:
                    base_penalty = 0.7
                elif late_game:
                    base_penalty = 0.9
                
                # Assess the value of the card being acquired
                card_id = action + 2  # Convert action to card ID (e.g., action 1 -> M3)
                card_value = self.card_values.get(card_id, 1.0)
                
                # Adjust penalty based on card value - reduce penalty for high-value cards
                # Value-based adjustment is stronger in early game
                value_factor = 0.5 if early_game else 0.3
                adjusted_base_penalty = base_penalty * (1.0 - (card_value - 1.0) * value_factor)
                
                # Ensure penalty is still positive even for valuable cards
                adjusted_base_penalty = max(base_penalty * 0.2, adjusted_base_penalty)
                
                # Exponential penalty factor that grows much faster
                excess = owned_cards - penalty_threshold
                penalty_factor = 1.0 - min(0.97, adjusted_base_penalty * (1.5 ** excess))
                
                # Modify the actual experience's reward value
                reward = reward * penalty_factor
                
                # Also reduce the priority of collecting more cards when we already have many
                priority *= penalty_factor
                
                # For severe hoarding, apply a much stronger direct negative reward
                # But consider card value for the penalty
                if owned_cards > penalty_threshold + 2:
                    # Value-adjusted direct penalty
                    direct_penalty_base = -1.0 * (owned_cards - penalty_threshold)
                    value_adjustment = min(0.8, (card_value - 1.0) * 0.4)  # More valuable cards get less penalty
                    direct_penalty = direct_penalty_base * (1.0 - value_adjustment)
                    reward += direct_penalty  # Stronger direct penalty for excessive hoarding
            
            # If we're already above the hard limit, make it extremely punishing
            # But still consider exceptionally valuable cards in early game
            hard_limit = 10 if early_game else 8  # Higher limit in early game
            if owned_cards >= hard_limit:
                # Assess card value
                value_factor = min(0.8, (card_value - 1.0) * 0.2)
                
                # Extremely valuable cards in early game get less severe penalty
                if early_game and card_value >= 2.8:
                    severe_penalty = -3.0 * (1.0 - value_factor)  # Reduced penalty for top-tier cards
                else:
                    severe_penalty = -5.0 * (1.0 - value_factor)  # Very severe penalty
                
                reward += severe_penalty
                priority *= 0.2  # De-prioritize these experiences
        
        # For card usage actions, give stronger bonus for using cards when we have many
        if action >= 24 and action <= 48:  # Card usage actions (useM1-useM25)
            # Bonus for using cards when we have many
            if owned_cards > 5:
                usage_bonus = min(2.0, 0.3 * (owned_cards - 5))
                reward += usage_bonus  # Increased bonus
                priority *= (1.0 + usage_bonus)  # Also boost priority
        
        # Add action count tracking if it doesn't exist
        if not hasattr(self, 'action_counts'):
            self.action_counts = np.ones(self.action_size)
        
        # Update action count
        self.action_counts[action] += 1
        
        # Prioritize rare actions
        rarity_factor = np.sum(self.action_counts) / (self.action_counts[action] * self.action_size)
        priority *= min(3.0, rarity_factor)  # Cap at 3x priority
        
        # Add turns_taken to the experience tuple for auxiliary task learning
        if turns_taken is not None:
            self.replay_buffer.append((state, action, reward, next_state, terminal, priority, turns_taken))
        else:
            self.replay_buffer.append((state, action, reward, next_state, terminal, priority, 0))

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
        
        # Extract turns data for auxiliary task
        turns_batch = torch.FloatTensor(np.array([batch[6] for batch in exp_batch])).to(self.device)

        return state_batch, action_batch, reward_batch, next_state_batch, terminal_batch, turns_batch

    def pick_epsilon_greedy_action(self, state, info):
            """Epsilon-greedy action selection with valid action masking"""
            valid_actions = info["valid_actions"]
            
            # Select random valid action with probability ε
            if random.uniform(0, 1) < self.epsilon:
                # Implement guided exploration - bias against excessive merchant card acquisition
                valid_indices = np.where(valid_actions == 1)[0]
                
                # Count merchant cards in current state
                merchant_cards = state[14:39]
                owned_cards = sum(1 for card_status in merchant_cards if card_status > 0)
                
                # Get the merchant market cards (to assess their value)
                merchant_market = state[:6]  # First 6 values represent the merchant market
                
                # Early game should focus on acquiring good cards, mid/late game should be more restrictive
                player_golem_count = state[39]  # Index for player1_golem_count
                early_game = player_golem_count < 1
                
                # If we already have too many cards, reduce probability of picking merchant card actions
                if (early_game and owned_cards >= self.max_desired_cards) or (not early_game and owned_cards >= self.max_desired_cards - 2):
                    # Get indices for merchant card acquisition actions (1-23)
                    merchant_indices = [idx for idx in valid_indices if 1 <= idx <= 23]
                    
                    # If there are merchant card actions available, possibly avoid them
                    if merchant_indices and len(valid_indices) > len(merchant_indices):
                        avoidance_chance = 0.7 if early_game else 0.9  # More flexibility in early game
                        if random.random() < avoidance_chance:
                            # Filter out merchant card actions
                            valid_indices = [idx for idx in valid_indices if idx not in merchant_indices]
                
                # Handle case where no valid actions exist during random selection (shouldn't happen in Century)
                if len(valid_indices) == 0:
                    # Fallback: maybe rest action if valid, otherwise error or default action
                    if valid_actions[0] == 1: return 0 
                    else: raise ValueError("No valid actions available for random choice!")
                
                return np.random.choice(valid_indices)

            # Pick action with highest Q-Value among valid actions
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values, turns_est = self.main_network(state_tensor)
                q_values = q_values[0].cpu().numpy()
                
                # Apply merchant card acquisition penalty
                # Actions 1-23 correspond to getM3 through getM25
                merchant_cards = state[14:39]  # Use correct slice for v13
                owned_cards = sum(1 for card_status in merchant_cards if card_status > 0)
                
                # Extract player progress indicators to determine game phase
                player_golem_count = state[39]  # Index for player1_golem_count
                player_points = state[40]  # Index for player1_points
                
                # Determine game phase based on golem count and points
                early_game = player_golem_count < 1
                mid_game = 1 <= player_golem_count < 3
                late_game = player_golem_count >= 3
                
                # Get the merchant market cards for value assessment
                merchant_market = state[:6]  # First 6 values represent the merchant market
                
                # Thresholds for card limits - more lenient in early game, stricter later
                if early_game:
                    soft_threshold = 6  # Allow more cards in early game for building engine
                    hard_threshold = 8
                elif mid_game:
                    soft_threshold = 4  # Tighter in mid game
                    hard_threshold = 6
                else:
                    soft_threshold = 3  # Very strict in late game
                    hard_threshold = 5
                
                # If agent has too many cards, reduce Q-values for getting more merchant cards
                if owned_cards >= soft_threshold:
                    # Progressive penalty based on how far above threshold
                    excess = owned_cards - soft_threshold
                    
                    # Exponential penalty that grows much faster than cubic
                    # Less severe in early game, more severe later
                    base_multiplier = 2.0 if early_game else 3.0
                    penalty_base = base_multiplier * (1.5 ** excess)
                    
                    # Scale penalty based on game phase
                    phase_multiplier = 1.0
                    if mid_game:
                        phase_multiplier = 2.0  # Much stronger penalty in mid-game
                    elif late_game:
                        phase_multiplier = 3.0  # Even stronger in late-game
                    
                    penalty = penalty_base * phase_multiplier
                    
                    # Apply penalty to getM3-getM25 actions (indices 1-23), but consider card value
                    for action_idx in range(1, 24):
                        if valid_actions[action_idx] == 1:  # Only penalize valid actions
                            # Card ID is action_idx + 2 (e.g., action 1 -> M3)
                            card_id = action_idx + 2
                            card_value = self.card_values.get(card_id, 1.0)
                            
                            # Adjust penalty based on card value - reduce penalty for high-value cards
                            # Value-based penalty reduction is stronger in early game
                            value_factor = 0.6 if early_game else 0.3
                            adjusted_penalty = penalty * (1.0 - (card_value - 1.0) * value_factor)
                            
                            # Ensure penalty is still positive even for valuable cards
                            adjusted_penalty = max(penalty * 0.2, adjusted_penalty)
                            
                            # Apply the adjusted penalty
                            q_values[action_idx] -= adjusted_penalty
                    
                    # Enhance boost for card usage actions dramatically
                    # Actions 24-48 correspond to playM1 through playM25
                    boost_base = min(15.0, owned_cards * 1.2)  # Much larger boost
                    
                    # Scale boost based on game phase and excess cards
                    phase_boost = 1.0
                    if mid_game:
                        phase_boost = 1.5
                    elif late_game:
                        phase_boost = 2.0
                    
                    # Additional boost based on how many excess cards we have
                    excess_boost = 1.0 + (excess * 0.5)
                    
                    final_boost = boost_base * phase_boost * excess_boost
                    
                    # Apply boost to all playM actions
                    for action_idx in range(24, 49):
                        if valid_actions[action_idx] == 1:  # Only boost valid actions
                            q_values[action_idx] += final_boost
                
                # Special boost for golem acquisition actions when possible
                # Actions 49-62 correspond to getG1-getG14
                golem_boost = 10.0 + player_golem_count * 5.0  # Much stronger boost for golems
                for action_idx in range(49, 63):
                    if valid_actions[action_idx] == 1:  # Only boost valid golem actions
                        q_values[action_idx] += golem_boost
                
                # If we're above the hard threshold, make merchant card acquisition nearly impossible
                # However, still allow extremely valuable cards in early game
                if owned_cards >= hard_threshold:
                    for action_idx in range(1, 24):
                        if valid_actions[action_idx] == 1:
                            # Card ID is action_idx + 2
                            card_id = action_idx + 2
                            card_value = self.card_values.get(card_id, 1.0)
                            
                            # In early game, only extremely valuable cards can be considered
                            if early_game and card_value >= 2.8:  # Only top-tier cards
                                q_values[action_idx] = -10.0  # Still penalized but not impossible
                            else:
                                q_values[action_idx] = -100.0  # Extremely low value
                
                masked_q_values = np.where(valid_actions == 1, q_values, -np.inf)
                # Handle case where all valid actions have -inf Q-value
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
        state_batch, action_batch, reward_batch, next_state_batch, terminal_batch, turns_batch = self.sample_experience_batch(batch_size)

        # Create a mask for valid actions in the next state
        # This would require storing valid_actions in replay buffer
        valid_action_mask = torch.zeros((batch_size, self.action_size), device=self.device)
        
        # Double DQN: Use main network to select actions, target network to evaluate them
        # Get next Q values from main network for action selection
        with torch.no_grad():
            next_q_values_main, _ = self.main_network(next_state_batch)
            # Apply valid action masking for next state
            next_q_values_main[~valid_action_mask.bool()] = -1e9
            # Select best actions from main network (Double DQN)
            best_actions = next_q_values_main.max(1)[1]
            
            # Get action values from target network
            next_q_values_target, _ = self.target_network(next_state_batch)
            # Use the best actions from main network to get Q-values from target network
            max_next_q = next_q_values_target.gather(1, best_actions.unsqueeze(1)).squeeze()
            target_q_values = reward_batch + (1 - terminal_batch) * self.gamma * max_next_q

        # Get current Q values and turn estimates
        current_q_values, turns_estimate = self.main_network(state_batch)
        current_q_values = current_q_values.gather(1, action_batch.unsqueeze(1))

        # Compute TD loss for Q-values
        q_loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # Only use valid turn data (> 0) for the auxiliary task
        valid_turns_mask = turns_batch > 0
        
        # If we have valid turns data, compute auxiliary loss
        if valid_turns_mask.sum() > 0:
            # Compute loss for turn estimation auxiliary task
            turns_loss = F.mse_loss(
                turns_estimate.squeeze()[valid_turns_mask], 
                turns_batch[valid_turns_mask]
            )
            
            # Combine losses with weighting
            total_loss = q_loss + self.efficiency_weight * turns_loss
        else:
            total_loss = q_loss
        
        # Zero gradients, compute gradients, and update weights
        self.optimizer.zero_grad()
        total_loss.backward()
        
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
            'optimal_cards_estimate': self.optimal_cards_estimate,
            'turns_to_win': list(self.turns_to_win),
            'max_desired_cards': self.max_desired_cards,
            'hard_limit_cards': self.hard_limit_cards,
            'efficiency_weight': self.efficiency_weight,
            'card_values': self.card_values,
            # Save action counts for prioritized experience replay
            'action_counts': getattr(self, 'action_counts', None),
            # Save win rate tracking information
            'peak_win_rate': getattr(self, 'peak_win_rate', 0.0),
            'recent_wins': list(self.recent_wins),
            # Save episode-specific trackers if they exist
            'episode_card_count': getattr(self, 'episode_card_count', 0),
            'episode_turns': getattr(self, 'episode_turns', [])
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
        self.cards_per_golem = deque(checkpoint.get('cards_per_golem', []), maxlen=100)
        self.turns_per_golem = deque(checkpoint.get('turns_per_golem', []), maxlen=100)
        self.optimal_cards_estimate = checkpoint.get('optimal_cards_estimate', 6)
        
        # Load turns to win if available
        self.turns_to_win = deque(checkpoint.get('turns_to_win', []), maxlen=100)
        
        # Load card limits if available
        self.max_desired_cards = checkpoint.get('max_desired_cards', 8)
        self.hard_limit_cards = checkpoint.get('hard_limit_cards', 10)
        self.efficiency_weight = checkpoint.get('efficiency_weight', 0.01)
        
        # Load card values if available
        if 'card_values' in checkpoint:
            self.card_values = checkpoint['card_values']
        
        # Load action counts for prioritized experience replay
        if 'action_counts' in checkpoint and checkpoint['action_counts'] is not None:
            self.action_counts = checkpoint['action_counts']
        
        # Load win rate tracking information
        self.peak_win_rate = checkpoint.get('peak_win_rate', 0.0)
        self.recent_wins = deque(checkpoint.get('recent_wins', []), maxlen=100)
        
        # Load episode-specific trackers if they exist
        self.episode_card_count = checkpoint.get('episode_card_count', 0)
        self.episode_turns = checkpoint.get('episode_turns', [])
        
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
    # Create checkpoint directory if it doesn't exist
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    env = gym.make("gymnasium_env/CenturyGolem-v14")
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
                    
                    # Track merchant card acquisitions (actions 1-23 are getM3-getM25)
                    if action >= 1 and action <= 23:
                        merchant_cards_acquired += 1
                        dqn_agent.episode_card_count += 1
                        
                    next_state, reward, terminal, _, info = env.step(action)
                    
                    # Augment reward with efficiency bonus
                    # Subtract a small penalty for each turn taken
                    time_penalty = -0.05 * turns_taken / 10  # Small penalty that increases with game length
                    reward += time_penalty
                    
                    dqn_total_reward += reward
                    
                    if not terminal and info['current_player'] == 1:
                        opponent_action = opponent.pick_action(next_state, info)
                        next_state_after_opponent, opponent_reward, terminal, _, info = env.step(opponent_action)
                        
                        opponent_total_reward += opponent_reward
                        
                        dqn_agent.save_experience(state, action, reward, next_state_after_opponent, terminal, turns_taken if terminal else None)
                        
                        state = next_state_after_opponent
                    else:
                        dqn_agent.save_experience(state, action, reward, next_state, terminal, turns_taken if terminal else None)
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
                        golem_count = next_state[39] if info['current_player'] == 0 else next_state[93]  # Adjust index based on player
                        if golem_count > 0:
                            dqn_agent.turns_per_golem.append(turns_taken / golem_count)
                            if len(dqn_agent.turns_per_golem) > 10:
                                avg_turns_per_golem = sum(list(dqn_agent.turns_per_golem)[-10:]) / min(10, len(dqn_agent.turns_per_golem))
                                print(f"Average turns per golem (last 10 wins): {avg_turns_per_golem:.1f}")
                        
                            # Track merchant card count at end of winning games
                            merchant_cards = next_state[14:39] if info['current_player'] == 0 else next_state[68:93]
                            final_card_count = sum(1 for card in merchant_cards if card > 0)
                            print(f"Final merchant card count: {final_card_count}")
                    
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
                    
                    # Dynamically adjust card limits based on winning performance
                    if len(win_indices) >= 5:  # Only if we have enough winning games
                        # If we're consistently winning with fewer cards, tighten limits
                        if cards_in_wins < dqn_agent.max_desired_cards - 1:
                            dqn_agent.max_desired_cards = max(6, int(cards_in_wins) + 1)
                            dqn_agent.hard_limit_cards = max(8, dqn_agent.max_desired_cards + 2)
                            print(f"Adjusted card limits: soft={dqn_agent.max_desired_cards}, hard={dqn_agent.hard_limit_cards}")

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
                elif avg_cards > 8 and win_rate > 0.6:  # Lower threshold from 10 to 8
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
    
    # Check for existing checkpoint if not provided in arguments but exists in default location
    if args.checkpoint is None and not args.reset:
        # Look for most recent checkpoint
        checkpoint_files = sorted(glob.glob(os.path.join("checkpoints", "checkpoint_ep*.pt")))
        if checkpoint_files:
            latest_checkpoint = checkpoint_files[-1]
            print(f"Found existing checkpoint: {latest_checkpoint}")
            print(f"Use with --checkpoint {latest_checkpoint} to resume training")
            print("Or use --reset to start from scratch")
            exit(1)
        else:
            print("No existing checkpoints found. Use --reset to start from scratch.")
            exit(1)
        
    config_dict = {
        'episodes': args.episodes if args.episodes is not None else 5000, # 10000
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

    # Add the ability to force initialization of certain parameters even when loading a checkpoint
    if args.checkpoint and args.epsilon is not None:
        print(f"Note: Overriding checkpoint epsilon with command line value: {args.epsilon}")
        # This will be handled in the DQNAgent after loading the checkpoint
    
    config = DQNConfig(**config_dict)
    train_dqn(config, config_dict['episodes'], config_dict['checkpoint'])