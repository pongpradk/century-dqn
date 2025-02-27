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

class RandomAgent:
    def __init__(self, action_size):
        self.action_size = action_size

class DQNAgent:
    def __init__(self, state_size, action_size):

        self.state_size = state_size
        self.action_size = action_size

        # Initialize Replay Buffer as python deque
        self.replay_buffer = deque(maxlen=50000)

        # Set algorithm hyperparameters
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.learning_rate = 0.00025
        self.update_rate = 50

        # Create both Main and Target Neural Networks
        self.main_network = self.create_nn()
        self.target_network = self.create_nn()

        # Initialize Target Network with Main Network's weights
        self.target_network.set_weights(self.main_network.get_weights())

    def create_nn(self):
        model = Sequential()
        
        model.add(Dense(128, activation='relu', input_dim=self.state_size))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        
        return model

    def update_target_network(self):
        """ Sync target network with main network """
        self.target_network.set_weights(self.main_network.get_weights())
        
    def save_experience(self, state, action, reward, next_state, terminal):
        self.replay_buffer.append((state, action, reward, next_state, terminal))

    def sample_experience_batch(self, batch_size):
        # Sample {batchsize} experiences from the Replay Buffer
        exp_batch = random.sample(self.replay_buffer, batch_size)

        # Create an array with the {batchsize} elements for s, a, r, s' and terminal information
        state_batch = np.array([batch[0] for batch in exp_batch]).reshape(batch_size, self.state_size)
        action_batch = np.array([batch[1] for batch in exp_batch])
        reward_batch = [batch[2] for batch in exp_batch]
        next_state_batch = np.array([batch[3] for batch in exp_batch]).reshape(batch_size, self.state_size)
        terminal_batch = [batch[4] for batch in exp_batch]

        # Return a tuple, where each item corresponds to each array/batch created above
        return state_batch, action_batch, reward_batch, next_state_batch, terminal_batch

    def pick_epsilon_greedy_action(self, state):
        # Pick random action with probability ε
        if random.uniform(0, 1) < self.epsilon:
            return np.random.randint(self.action_size)

        # Pick action with highest Q-Value (item with highest value for Main NN's output)
        state = state.reshape((1, self.state_size))
        q_values = self.main_network.predict(state, verbose=0)
        return np.argmax(q_values[0])

    def train(self, batch_size):

        # Sample a batch of experiences
        state_batch, action_batch, reward_batch, next_state_batch, terminal_batch = self.sample_experience_batch(batch_size)

        # Get the actions with highest Q-Value for the batch of next states
        next_q = self.target_network.predict(next_state_batch, verbose=0)
        max_next_q = np.amax(next_q, axis=1)
        # Get the Q-Values of each state in the batch of states
        q_values = self.main_network.predict(state_batch, verbose=0)

        # Update the Q-Value corresponding to the current action with the Target Value
        for i in range(batch_size):
            q_values[i][action_batch[i]] = reward_batch[i] if terminal_batch[i] else reward_batch[i] + self.gamma * max_next_q[i]

        # Fit the Neural Network
        self.main_network.fit(state_batch, q_values, verbose=0)

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

    env = gym.make("gymnasium_env/CenturyGolem-v6")
    env = FlattenObservation(env)

    # Define state and action size
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    dqn_agent = DQNAgent(state_size, action_size)
    
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
    num_timesteps = 400
    batch_size = 128
    time_step = metadata['time_step']
    rewards = metadata['rewards']
    epsilon_values = metadata['epsilon_values']

    try:
        for ep in range(metadata['episode'], num_episodes):

            tot_reward = 0

            state, info = env.reset()

            print(f'\nTraining on EPISODE {ep+1} with epsilon {dqn_agent.epsilon}')
            start = time.time()

            for t in range(num_timesteps):

                time_step += 1

                # Update Target Network every {dqn_agent.update_rate} timesteps
                if time_step % dqn_agent.update_rate == 0:
                    dqn_agent.update_target_network()
                
                # if opponent starts first
                if info['current_player'] == 1:                    
                    opponent_action = env.action_space.sample()
                    next_state, _, terminal, _, info = env.step(opponent_action)
                    state = next_state
                
                if info["current_player"] == 0:
                    # Select action using valid actions
                    action = dqn_agent.pick_epsilon_greedy_action(state) # Select action with ε-greedy policy
                    next_state, reward, terminal, _, info = env.step(action)  # Perform action on environment
                    
                    if not terminal and info["current_player"] == 1:
                        opponent_action = env.action_space.sample()
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
            if sum(rewards[-10:]) > 3000:
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
        
        # ✅ Append new rewards to 'rewards.txt' (keeping previous records)
        try:
            with open('rewards.txt', 'r') as f:
                prev_rewards = json.loads(f.read())  # Load existing rewards
        except (FileNotFoundError, json.JSONDecodeError):
            prev_rewards = []  # If file doesn't exist or is corrupted, start fresh

        # Combine previous rewards with new ones
        updated_rewards = prev_rewards + rewards

        with open('rewards.txt', 'w') as f:  # Overwrite with full updated list
            f.write(json.dumps(updated_rewards))
        print("Rewards of the training saved in 'rewards.txt'")

        # ✅ Append new epsilon values to 'epsilon_values.txt' (keeping previous records)
        try:
            with open('epsilon_values.txt', 'r') as f:
                prev_epsilon_values = json.loads(f.read())  # Load existing epsilon values
        except (FileNotFoundError, json.JSONDecodeError):
            prev_epsilon_values = []  # If file doesn't exist or is corrupted, start fresh

        # Combine previous epsilon values with new ones
        updated_epsilon_values = prev_epsilon_values + epsilon_values

        with open('epsilon_values.txt', 'w') as f:  # Overwrite with full updated list
            f.write(json.dumps(updated_epsilon_values))
        print("Epsilon values of the training saved in 'epsilon_values.txt'")
        
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

    # Save rewards on 'rewards.txt' file
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
    
