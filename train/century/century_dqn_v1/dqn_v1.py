import json
import random
import time
from collections import deque

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F

import gymnasium_env
from gymnasium.wrappers import FlattenObservation

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        # Initialize Replay Buffer
        self.replay_buffer = deque(maxlen=40000)

        # Set algorithm hyperparameters
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.98
        self.learning_rate = 0.001
        self.update_rate = 10

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
        self.replay_buffer.append((state, action, reward, next_state, terminal))

    def sample_experience_batch(self, batch_size):
        # Sample experiences from the Replay Buffer
        exp_batch = random.sample(self.replay_buffer, batch_size)

        # Create tensors for each component
        state_batch = torch.FloatTensor(np.array([batch[0] for batch in exp_batch])).to(self.device)
        action_batch = torch.LongTensor(np.array([batch[1] for batch in exp_batch])).to(self.device)
        reward_batch = torch.FloatTensor(np.array([batch[2] for batch in exp_batch])).to(self.device)
        next_state_batch = torch.FloatTensor(np.array([batch[3] for batch in exp_batch])).to(self.device)
        terminal_batch = torch.FloatTensor(np.array([batch[4] for batch in exp_batch])).to(self.device)

        return state_batch, action_batch, reward_batch, next_state_batch, terminal_batch

    def pick_epsilon_greedy_action(self, state):
        # Pick random action with probability ε
        if random.uniform(0, 1) < self.epsilon:
            return random.randrange(self.action_size)

        # Pick action with highest Q-Value
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.main_network(state)
            return q_values.argmax().item()

    def train(self, batch_size):
        # Sample a batch of experiences
        state_batch, action_batch, reward_batch, next_state_batch, terminal_batch = self.sample_experience_batch(batch_size)

        # Get current Q values
        current_q_values = self.main_network(state_batch).gather(1, action_batch.unsqueeze(1))

        # Get next Q values from target network
        with torch.no_grad():
            next_q_values = self.target_network(next_state_batch).max(1)[0]
            target_q_values = reward_batch + (1 - terminal_batch) * self.gamma * next_q_values

        # Compute loss and update weights
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

def temp():
    env = gym.make("gymnasium_env/CenturyGolem-v9")
    env = FlattenObservation(env)
    state, info = env.reset()
    
    if info['current_player'] == 0:
        if not terminal and info['current_player'] == 1:
            pass
    elif info['current_player'] == 1:
        pass
        

if __name__ == '__main__':
    env = gym.make("gymnasium_env/CenturyGolem-v9")
    env = FlattenObservation(env)
    state, _ = env.reset()

    # Define state and action size
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # Define number of episodes, timesteps per episode and batch size
    num_episodes = 150
    num_timesteps = 500
    batch_size = 64
    dqn_agent = DQNAgent(state_size, action_size)
    time_step = 0
    rewards, epsilon_values = list(), list()

    for ep in range(num_episodes):
        tot_reward = 0
        state, _ = env.reset()

        print(f'\nTraining on EPISODE {ep+1} with epsilon {dqn_agent.epsilon}')
        start = time.time()

        for t in range(num_timesteps):
            time_step += 1

            # Update Target Network every update_rate timesteps
            if time_step % dqn_agent.update_rate == 0:
                dqn_agent.update_target_network()

            action = dqn_agent.pick_epsilon_greedy_action(state)
            next_state, reward, terminal, _, _ = env.step(action)
            dqn_agent.save_experience(state, action, reward, next_state, terminal)

            state = next_state
            tot_reward += reward

            if terminal:
                print('Episode: ', ep+1, ',' ' terminated with Reward ', tot_reward)
                break

            # Train the Main NN when ReplayBuffer has enough experiences
            if len(dqn_agent.replay_buffer) > batch_size:
                dqn_agent.train(batch_size)

        rewards.append(tot_reward)
        epsilon_values.append(dqn_agent.epsilon)

        # Update Epsilon value
        if dqn_agent.epsilon > dqn_agent.epsilon_min:
            dqn_agent.epsilon *= dqn_agent.epsilon_decay

        # Print episode info
        elapsed = time.time() - start
        print(f'Time elapsed during EPISODE {ep+1}: {elapsed} seconds = {round(elapsed/60, 3)} minutes')

    # Save rewards
    with open('rewards_v2.txt', 'w') as f:
        f.write(json.dumps(rewards))
    print("Rewards of the training saved in 'rewards_v2.txt'")

    # Save epsilon values
    with open('epsilon_values_v2.txt', 'w') as f:
        f.write(json.dumps(epsilon_values))
    print("Epsilon values of the training saved in 'epsilon_values_v2.txt'")

    # Save trained model
    torch.save(dqn_agent.main_network.state_dict(), 'trained_agent_v2.pth')
    print("Trained agent saved in 'trained_agent_v2.pth'")
