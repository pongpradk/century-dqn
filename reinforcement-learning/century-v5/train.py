import json
import random
import time
import pickle
from collections import deque

import gymnasium as gym
import gymnasium_env
from gymnasium.wrappers import FlattenObservation
import numpy as np
from keras.layers import Dense, Dropout, BatchNormalization
from keras.models import Sequential
from keras.optimizers import Adam

class PrioritizedReplayBuffer:
    def __init__(self, max_size, alpha=0.6):
        self.max_size = max_size
        self.alpha = alpha
        self.buffer = []
        self.priorities = np.zeros((max_size,), dtype=np.float32)
        self.position = 0

    def __len__(self):
        """ Returns the number of elements in the buffer """
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
        """ Sample batch of experiences based on priority """
        if len(self.buffer) == 0:
            return []

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
        """ Update priorities based on new TD errors """
        self.priorities[indices] = np.abs(td_errors) + 1e-6  # Avoid zero priority

class RandomAgent:
    def __init__(self, action_size):
        self.action_size = action_size

    def pick_random_action(self, valid_actions):
        valid_indices = np.where(valid_actions == 1)[0]
        return np.random.choice(valid_indices)

class DQNAgent:
    def __init__(self, state_size, action_size):

        self.state_size = state_size
        self.action_size = action_size

        # Initialize Prioritized Replay Buffer
        self.replay_buffer = PrioritizedReplayBuffer(max_size=500000)

        # Hyperparameters
        self.gamma = 0.98
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995
        self.learning_rate = 0.0001
        self.update_rate = 500

        # Create networks
        self.main_network = self.create_nn()
        self.target_network = self.create_nn()
        self.target_network.set_weights(self.main_network.get_weights())

    def create_nn(self):
        """ Creates a simple feedforward neural network """
        model = Sequential()
        model.add(Dense(256, activation='relu', input_dim=self.state_size))
        model.add(BatchNormalization())
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def update_target_network(self):
        """ Sync target network with main network """
        self.target_network.set_weights(self.main_network.get_weights())

    def save_experience(self, state, action, reward, next_state, terminal):
        """ Compute TD error and save prioritized experience """
        state = state.reshape((1, self.state_size))
        next_state = next_state.reshape((1, self.state_size))
        
        # Compute TD error: TD_error = |Q(s,a) - target|
        q_values = self.main_network.predict(state, verbose=0)[0]
        next_q_values = self.target_network.predict(next_state, verbose=0)[0]
        
        target = reward if terminal else reward + self.gamma * np.max(next_q_values)
        td_error = abs(q_values[action] - target)

        # Add experience with priority
        self.replay_buffer.add((state, action, reward, next_state, terminal), td_error)

    def sample_experience_batch(self, batch_size):
        """ Sample a batch from replay buffer """
        return self.replay_buffer.sample(batch_size)

    def pick_epsilon_greedy_action(self, state, valid_actions):
        """ Epsilon-greedy action selection with valid action masking """
        if random.uniform(0, 1) < self.epsilon:
            valid_indices = np.where(valid_actions == 1)[0]
            return np.random.choice(valid_indices)

        state = state.reshape((1, self.state_size))
        q_values = self.main_network.predict(state, verbose=0)[0]
        masked_q_values = np.where(valid_actions, q_values, -np.inf)
        return np.argmax(masked_q_values)

    def train(self, batch_size, beta=0.4):
        """ Train the network using prioritized experience replay """
        experiences, indices, weights = self.replay_buffer.sample(batch_size, beta)
        if not experiences:
            return

        state_batch = np.array([e[0] for e in experiences]).reshape(batch_size, self.state_size)
        action_batch = np.array([e[1] for e in experiences])
        reward_batch = np.array([e[2] for e in experiences])
        next_state_batch = np.array([e[3] for e in experiences]).reshape(batch_size, self.state_size)
        terminal_batch = np.array([e[4] for e in experiences])

        # Compute TD targets
        next_q_values = self.target_network.predict(next_state_batch, verbose=0)
        max_next_q = np.amax(next_q_values, axis=1)
        targets = reward_batch + (1 - terminal_batch) * self.gamma * max_next_q

        # Compute TD errors
        q_values = self.main_network.predict(state_batch, verbose=0)
        td_errors = np.abs(q_values[np.arange(batch_size), action_batch] - targets)

        # Update Q-values with importance sampling
        targets = q_values
        targets[np.arange(batch_size), action_batch] = reward_batch + (1 - terminal_batch) * self.gamma * max_next_q
        self.main_network.fit(state_batch, targets, sample_weight=weights, verbose=0)

        # Update priorities in buffer
        self.replay_buffer.update_priorities(indices, td_errors)

    def save_agent(self, filename_prefix):
        """ Save the agent's models and replay buffer """
        self.main_network.save(f'{filename_prefix}_main_model.keras')
        self.target_network.save(f'{filename_prefix}_target_model.keras')
        with open(f'{filename_prefix}_metadata.pkl', 'wb') as f:
            pickle.dump({'epsilon': self.epsilon, 'replay_buffer': self.replay_buffer}, f)
        print(f"[INFO] Agent saved as '{filename_prefix}'.")

    def load_agent(self, filename_prefix):
        """ Load the agent's models and replay buffer """
        self.main_network.load_weights(f'{filename_prefix}_main_model.keras')
        self.target_network.load_weights(f'{filename_prefix}_target_model.keras')
        with open(f'{filename_prefix}_metadata.pkl', 'rb') as f:
            data = pickle.load(f)
            self.epsilon = data['epsilon']
            self.replay_buffer = data['replay_buffer']
        print(f"[INFO] Agent loaded from '{filename_prefix}'.")

def save_training_metadata(filename, metadata):
    with open(filename, 'w') as f:
        json.dump(metadata, f)
    print(f"[INFO] Training metadata saved to '{filename}'.")

def load_training_metadata(filename):
    try:
        with open(filename, 'r') as f:
            metadata = json.load(f)
        print(f"[INFO] Training metadata loaded from '{filename}'.")
        return metadata
    except FileNotFoundError:
        print(f"[INFO] No previous training metadata found. Starting fresh.")
        return {'episode': 0, 'time_step': 0, 'rewards': [], 'epsilon_values': []}

if __name__ == '__main__':

    env = gym.make("gymnasium_env/CenturyGolem-v5")
    env = FlattenObservation(env)

    # Define state and action size
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    dqn_agent = DQNAgent(state_size, action_size)
    random_agent = RandomAgent(action_size)
    
    # Load Agent and Metadata if continuing training
    continue_training = True  # Set to False if starting fresh
    if not continue_training:
        input("[INFO] Please confirm that you do not want to continue.")
    if continue_training:
        dqn_agent.load_agent('checkpoint')  # Load saved agent
        metadata = load_training_metadata('training_metadata.json')
    else:
        metadata = {'episode': 0, 'time_step': 0, 'rewards': [], 'epsilon_values': []}
    
    # Define number of episodes, timesteps per episode and batch size
    num_episodes = 10000
    num_timesteps = 200
    batch_size = 256
    time_step = metadata['time_step']
    rewards = metadata['rewards']
    epsilon_values = metadata['epsilon_values']

    try:
        for ep in range(metadata['episode'], num_episodes):

            tot_reward = 0

            # state, _ = env.reset()
            state, info = env.reset()
                
            if ep % 500 == 0:
                dqn_agent.epsilon = max(0.5, dqn_agent.epsilon * (1 + ep / num_episodes))

            print(f'\nTraining on EPISODE {ep+1} with epsilon {dqn_agent.epsilon}')
            start = time.time()

            for t in range(num_timesteps):

                time_step += 1

                # Update Target Network every {dqn_agent.update_rate} timesteps
                if time_step % dqn_agent.update_rate == 0:
                    dqn_agent.update_target_network()
                
                # if opponent starts first
                if info['current_player'] == 1:
                    opponent_action = random_agent.pick_random_action(info["valid_actions"])
                    next_state, _, terminal, _, info = env.step(opponent_action)
                    state = next_state
                
                if info["current_player"] == 0:
                    # Select action using valid actions
                    action = dqn_agent.pick_epsilon_greedy_action(state, info["valid_actions"]) # Select action with ε-greedy policy
                    next_state, reward, terminal, _, info = env.step(action)  # Perform action on environment
                    
                    if not terminal and info["current_player"] == 1:
                        opponent_action = random_agent.pick_random_action(info["valid_actions"])
                        next_state_after_opponent, _, terminal, _, info = env.step(opponent_action) # ignore reward from opponent's turn
                        
                        dqn_agent.save_experience(state, action, reward, next_state_after_opponent, terminal) # Save experience in Replay Buffer
                        
                        # Update state for the next DQN turn
                        state = next_state_after_opponent
                    else:
                        dqn_agent.save_experience(state, action, reward, next_state, terminal)
                    
                    tot_reward += reward  # Only DQNAgent's reward is accumulated
                    
                    # Train the Main NN when ReplayBuffer has enough experiences to fill a batch
                    if len(dqn_agent.replay_buffer) > batch_size:
                        dqn_agent.train(batch_size)

                if terminal:
                    print('Episode: ', ep+1, ',' ' terminated with Reward ', tot_reward)
                    break

            rewards.append(tot_reward)
            epsilon_values.append(dqn_agent.epsilon)

            # Everytime an episode ends, update Epsilon value to a lower value
            if dqn_agent.epsilon > dqn_agent.epsilon_min:
                dqn_agent.epsilon *= dqn_agent.epsilon_decay

            # Print info about the episode performed
            elapsed = time.time() - start
            print(f'Time elapsed during EPISODE {ep+1}: {elapsed} seconds = {round(elapsed/60, 3)} minutes')

            # If the agent got a reward >499 in each of the last 10 episodes, the training is terminated
            if sum(rewards[-10:]) > 4990:
                print('Training stopped because agent has performed a perfect episode in the last 10 episodes')
                break
            
    except KeyboardInterrupt:
        print("\n[INFO] Training interrupted manually. Saving checkpoint...")

        # Save Agent and Metadata upon interruption
        dqn_agent.save_agent('checkpoint')
        save_training_metadata('training_metadata.json', {
            'episode': ep + 1,
            'time_step': time_step,
            'rewards': rewards,
            'epsilon_values': epsilon_values
        })
        print("[INFO] Checkpoint saved successfully. You can resume training later.")
        
    finally:
        env.close()
        print("[INFO] Environment closed.")
        
# Final save after training stops
# dqn_agent.save_agent('final_agent')
# save_training_metadata('final_training_metadata.json', {
#     'episode': ep + 1,
#     'time_step': time_step,
#     'rewards': rewards,
#     'epsilon_values': epsilon_values
# })
# print("[INFO] Final model and metadata saved.")

    # # Save rewards on 'rewards.txt' file
    # with open('rewards.txt', 'w') as f:
    #     f.write(json.dumps(rewards))
    # print("Rewards of the training saved in 'rewards.txt'")

    # # Save epsilon values
    # with open('epsilon_values.txt', 'w') as f:
    #     f.write(json.dumps(epsilon_values))
    # print("Epsilon values of the training saved in 'epsilon_values.txt'")

    # # Save trained model
    # dqn_agent.main_network.save('trained_agent.keras')
    # print("Trained agent saved in 'trained_agent.keras'")

