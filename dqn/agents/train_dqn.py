"""
DQN Agent for Century: Golem Edition YGBP Environment

Trains a DQN agent to play Century: Golem Edition YGBP against a selected opponent agent.
"""

import random
import time
import os
import pickle
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
from random_agent import RandomAgent
from phase_agent import PhaseAgent



CONFIG = {
    'num_episodes': 1000,
    'num_timesteps': 2000,
    'model_save_freq': 50,
    'opponent_agent': 'phase', # Type of opponent ('random' or 'phase')
    
    'gamma': 0.99,                
    'epsilon': 1.0,                
    'epsilon_min': 0.05,
    'epsilon_decay': 0.985,
    'learning_rate': 0.0007,
    'update_rate': 150,
    'batch_size': 256,
    'replay_buffer_size': 100000,
}


class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 512)
        self.dropout1 = nn.Dropout(0.25)
        self.fc2 = nn.Linear(512, 512)
        self.dropout2 = nn.Dropout(0.25)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, action_size)
        
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

        self.gamma = config['gamma']
        self.epsilon = config['epsilon']
        self.epsilon_min = config['epsilon_min']
        self.epsilon_decay = config['epsilon_decay']
        self.learning_rate = config['learning_rate']
        self.update_rate = config['update_rate']
        
        self.replay_buffer = deque(maxlen=config['replay_buffer_size'])
        
        # Metrics tracking
        self.rewards = []
        self.epsilon_values = []
        self.merchant_card_counts = []
        self.training_loss = []
        self.avg_q_values = []
        self.recent_wins = deque(maxlen=100)
        self.action_counts = np.ones(action_size)
        
        self.total_episodes = 1000
        self.epsilon_decay_schedule = {}
        self._initialize_epsilon_schedule()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.main_network = DQN(state_size, action_size).to(self.device)
        self.target_network = DQN(state_size, action_size).to(self.device)
        
        self.target_network.load_state_dict(self.main_network.state_dict())
        
        self.optimizer = optim.Adam(
            self.main_network.parameters(), 
            lr=self.learning_rate, 
            weight_decay=1e-5
        )
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='max',
            factor=0.5,
            patience=30,
            threshold=0.01,
            verbose=True, 
            min_lr=1e-6
        )

    def _initialize_epsilon_schedule(self):
        epsilon_start = 1.0
        epsilon_mid = 0.3
        epsilon_end = 0.05
        
        episodes_start = 0
        episodes_mid = int(self.total_episodes * 0.3)
        episodes_end = self.total_episodes
        
        # Schedule for initial phase (high exploration)
        for ep in range(episodes_start, episodes_mid):
            progress = (ep - episodes_start) / (episodes_mid - episodes_start)
            self.epsilon_decay_schedule[ep] = epsilon_start - progress * (epsilon_start - epsilon_mid)
            
        # Schedule for later phase (increasing exploitation)
        for ep in range(episodes_mid, episodes_end):
            progress = (ep - episodes_mid) / (episodes_end - episodes_mid)
            decay = math.exp(-5 * progress)
            self.epsilon_decay_schedule[ep] = epsilon_mid - (1 - decay) * (epsilon_mid - epsilon_end)

    def update_epsilon(self, episode):
        if episode in self.epsilon_decay_schedule:
            self.epsilon = self.epsilon_decay_schedule[episode]
            return
            
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon_min, self.epsilon)

    def update_target_network(self):
        self.target_network.load_state_dict(self.main_network.state_dict())

    def save_experience(self, state, action, reward, next_state, terminal):
        """
        Experience to replay buffer with priority
        
        Prioritises:
        - Terminal states
        - High rewards
        - Golem acquisition actions
        - Rare action types
        """
        priority = 1.0
        
        # Prioritise terminal states
        if terminal:
            priority *= 3.0
        
        # Prioritise high rewards
        if reward > 5.0:
            priority *= 2.0
        
        # Prioritise golem acquisition actions
        if 89 <= action <= 124:  # getG1-getG36
            priority *= 4.0
        
        # Apply merchant card hoarding penalty
        if 1 <= action <= 43:  # getM3-getM45
            # Count owned merchant cards
            merchant_cards = state[14:59]
            owned_cards = sum(1 for card_status in merchant_cards if card_status > 0)
            
            if owned_cards > 4:
                penalty_factor = 1.0 - min(0.9, 0.25 * (owned_cards - 4))
                reward = reward * penalty_factor
                priority *= penalty_factor
        
        # Prioritise rare actions
        self.action_counts[action] += 1
        rarity_factor = np.sum(self.action_counts) / (self.action_counts[action] * self.action_size)
        priority *= min(3.0, rarity_factor)
        
        # Replay buffer with priority
        self.replay_buffer.append((state, action, reward, next_state, terminal, priority))

    def sample_experience_batch(self, batch_size):
        priorities = np.array([exp[5] for exp in self.replay_buffer])
        probs = priorities / np.sum(priorities)
        
        indices = np.random.choice(len(self.replay_buffer), batch_size, p=probs)
        exp_batch = [self.replay_buffer[idx] for idx in indices]

        state_batch = torch.FloatTensor(np.array([batch[0] for batch in exp_batch])).to(self.device)
        action_batch = torch.LongTensor(np.array([batch[1] for batch in exp_batch])).to(self.device)
        reward_batch = torch.FloatTensor(np.array([batch[2] for batch in exp_batch])).to(self.device)
        next_state_batch = torch.FloatTensor(np.array([batch[3] for batch in exp_batch])).to(self.device)
        terminal_batch = torch.FloatTensor(np.array([batch[4] for batch in exp_batch])).to(self.device)

        return state_batch, action_batch, reward_batch, next_state_batch, terminal_batch

    def pick_epsilon_greedy_action(self, state, info):
        valid_actions = info["valid_actions"]
        valid_indices = np.where(valid_actions == 1)[0]
        
        # Edge case with no valid actions
        if len(valid_indices) == 0:
            if valid_actions[0] == 1:  # Try rest action
                return 0
            raise ValueError("No valid actions available!")
        
        # Random action
        if random.uniform(0, 1) < self.epsilon:
            return np.random.choice(valid_indices)

        # Non-random action
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.main_network(state_tensor)[0].cpu().numpy()
            
            self.avg_q_values.append(np.mean(q_values))
            self._apply_action_penalties(state, q_values, valid_actions)
            
            masked_q_values = np.where(valid_actions == 1, q_values, -np.inf)
            
            # In case all valid actions have -inf Q-value
            if np.all(masked_q_values == -np.inf):
                return np.random.choice(valid_indices)
                
            return np.argmax(masked_q_values)
            
    def _apply_action_penalties(self, state, q_values, valid_actions):
        # Count owned merchant cards
        merchant_cards = state[14:59]
        owned_cards = sum(1 for card_status in merchant_cards if card_status > 0)
        
        if owned_cards > 4:
            # Penalty for acquiring more merchant cards
            penalty = (owned_cards - 4) ** 2 * 0.8
            
            # Apply penalty to getM3-getM45 (indices 1-43)
            for action_idx in range(1, 44):
                if valid_actions[action_idx] == 1:
                    q_values[action_idx] -= penalty
            
            # Boost for card usage actions
            boost = min(5.0, owned_cards * 0.5)
            for action_idx in range(44, 89):  # useM1-useM45
                if valid_actions[action_idx] == 1:
                    q_values[action_idx] += boost

    def train(self, batch_size):
        state_batch, action_batch, reward_batch, next_state_batch, terminal_batch = \
            self.sample_experience_batch(batch_size)

        with torch.no_grad():
            next_q_values_main = self.main_network(next_state_batch)
            best_actions = next_q_values_main.max(1)[1]
            
            next_q_values_target = self.target_network(next_state_batch)
            max_next_q = next_q_values_target.gather(1, best_actions.unsqueeze(1)).squeeze()
            
            target_q_values = reward_batch + (1 - terminal_batch) * self.gamma * max_next_q

        current_q_values = self.main_network(state_batch).gather(1, action_batch.unsqueeze(1))

        loss = F.smooth_l1_loss(current_q_values.squeeze(), target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.main_network.parameters(), 1.0)
        
        self.optimizer.step()
        
        self.training_loss.append(loss.item())
    
    def save_model(self, episode, models_dir="models"):
        """Save the model for inference"""
        os.makedirs(models_dir, exist_ok=True)
        model_path = os.path.join(models_dir, f"trained_model_{episode}.pt")
        torch.save(self.main_network.state_dict(), model_path)
        print(f"Model saved at episode {episode}")
        return model_path


def train_dqn(opponent_type=None):
    if opponent_type is None:
        opponent_type = CONFIG['opponent_agent']
    
    env = gym.make("gymnasium_env/CenturyGolem-YGBP")
    env = FlattenObservation(env)
    state, info = env.reset()

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    dqn_agent = DQNAgent(state_size, action_size, CONFIG)
    
    if opponent_type.lower() == 'phase':
        opponent = PhaseAgent(action_size)
        print("Training against PhaseAgent")
    else:
        opponent = RandomAgent(action_size)
        print("Training against RandomAgent")
    
    time_step = 0
    
    try:
        for ep in range(CONFIG['num_episodes']):
            dqn_total_reward = 0
            opponent_total_reward = 0
            merchant_cards_acquired = 0
            
            state, info = env.reset()
            
            print(f'\nTraining on EPISODE {ep+1} with epsilon {dqn_agent.epsilon:.4f}')
            start = time.time()

            for t in range(CONFIG['num_timesteps']):
                time_step += 1

                if time_step % dqn_agent.update_rate == 0:
                    dqn_agent.update_target_network()

                # DQN agent's turn
                if info['current_player'] == 0:
                    action = dqn_agent.pick_epsilon_greedy_action(state, info)
                    
                    # Track merchant card acquisitions
                    if 1 <= action <= 43:
                        merchant_cards_acquired += 1
                        
                    next_state, reward, terminal, _, info = env.step(action)
                    dqn_total_reward += reward
                    
                    # Opponent's turn
                    if not terminal and info['current_player'] == 1:
                        opponent_action = opponent.pick_action(next_state, info)
                        next_state_after_opponent, opponent_reward, terminal, _, info = env.step(opponent_action)
                        
                        opponent_total_reward += opponent_reward
                        dqn_agent.save_experience(state, action, reward, next_state_after_opponent, terminal)
                        state = next_state_after_opponent
                    else:
                        dqn_agent.save_experience(state, action, reward, next_state, terminal)
                        state = next_state
                
                # Opponent's turn
                elif info['current_player'] == 1:
                    opponent_action = opponent.pick_action(state, info)
                    next_state, opponent_reward, terminal, _, info = env.step(opponent_action)
                    
                    opponent_total_reward += opponent_reward
                    state = next_state

                if terminal:
                    print(f'Episode {ep+1} terminated with reward {dqn_total_reward:.2f}')
                    break

                if len(dqn_agent.replay_buffer) > CONFIG['batch_size']:
                    dqn_agent.train(CONFIG['batch_size'])

            dqn_agent.rewards.append(dqn_total_reward)
            dqn_agent.epsilon_values.append(dqn_agent.epsilon)
            dqn_agent.merchant_card_counts.append(merchant_cards_acquired)
            
            dqn_agent.update_epsilon(ep)
            
            _print_periodic_stats(dqn_agent, ep)
            _update_learning_rate(dqn_agent, ep)

            # Save model
            if ((ep + 1) % CONFIG['model_save_freq'] == 0) or (ep == 0):
                dqn_agent.save_model(ep + 1)
                
            elapsed = time.time() - start
            print(f'Time elapsed during EPISODE {ep+1}: {elapsed:.2f} seconds = {round(elapsed/60, 3):.2f} minutes')

    except KeyboardInterrupt:
        print("\nTraining interrupted manually.")
    
    finally:
        env.close()


def _print_periodic_stats(agent, episode):
    if episode % 10 == 0:
        # Merchant card acquisition stats
        recent_cards = agent.merchant_card_counts[-10:]
        avg_cards = sum(recent_cards) / len(recent_cards) if recent_cards else 0
        print(f"Avg merchant cards acquired (last 10 ep): {avg_cards:.2f}")
        
        # Training loss
        if agent.training_loss:
            recent_loss = agent.training_loss[-100:]
            avg_loss = sum(recent_loss) / len(recent_loss)
            print(f"Avg training loss (last 100 updates): {avg_loss:.4f}")
        
        # Q-values
        if agent.avg_q_values:
            recent_q = agent.avg_q_values[-100:]
            avg_q = sum(recent_q) / len(recent_q)
            print(f"Avg Q-value (last 100 actions): {avg_q:.4f}")


def _update_learning_rate(agent, episode):
    if episode > 0 and episode % 25 == 0:
        # Monitor win rate for dynamic adjustment
        recent_rewards = agent.rewards[max(0, episode-25):episode]
        recent_wins = sum(1 for r in recent_rewards if r > 0)
        win_rate = recent_wins / len(recent_rewards) if recent_rewards else 0
        
        agent.recent_wins.append(1 if agent.rewards[episode-1] > 0 else 0)
        moving_win_rate = sum(agent.recent_wins) / len(agent.recent_wins) if agent.recent_wins else 0
        
        # Update learning rate based on win rate
        agent.scheduler.step(moving_win_rate)
        current_lr = agent.optimizer.param_groups[0]['lr']
        print(f"Current win rate: {moving_win_rate:.2f}, learning rate: {current_lr:.6f}")
        
        # Increase exploration if win rate is low
        if win_rate < 0.3:
            boost = 0.2 + (0.3 - win_rate) * 0.5
            agent.epsilon = min(0.7, agent.epsilon + boost)
            print(f"Low win rate ({win_rate:.2f}) detected, increasing epsilon to {agent.epsilon:.4f}")
        
        # Track peak win rate and increase exploration if drop a lot
        if hasattr(agent, 'peak_win_rate'):
            if agent.peak_win_rate > 0.7 and moving_win_rate < agent.peak_win_rate - 0.15:
                boost = 0.1 + (agent.peak_win_rate - moving_win_rate) * 0.3
                agent.epsilon = min(0.4, agent.epsilon + boost)
                print(f"Win rate dropped from peak {agent.peak_win_rate:.2f} to {moving_win_rate:.2f}, "
                      f"increasing exploration to {agent.epsilon:.4f}")
        
        # Update peak win rate
        if not hasattr(agent, 'peak_win_rate') or moving_win_rate > agent.peak_win_rate:
            agent.peak_win_rate = moving_win_rate


if __name__ == '__main__':
    train_dqn()