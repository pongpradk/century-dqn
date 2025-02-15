import numpy as np
import random
import gymnasium as gym
import gymnasium_env
import time
import matplotlib.pyplot as plt
from IPython import display

def initialize_Q_table(env):
    return np.zeros([env.observation_space.n, env.action_space.n])

def get_epsilon_greedy_action(Q, state):
    
    if np.random.uniform(0, 1) < epsilon: # epsilon defined in the cell below
        return env.action_space.sample()
    
    return np.argmax(Q[state])

def execute_episode(Q, max_timesteps):
    
    # Initialize Rewards historic
    rewards = list()
    
    # Reset environment
    state, _ = env.reset()
    
    for _ in range(max_timesteps):
        
        # Pick action following epsilon-greedy policy
        action = get_epsilon_greedy_action(Q, state)
        
        # Perform action and receive new state S' and reward R
        new_state, reward, done, info, _ = env.step(action)
        rewards.append(reward)
        
        # Update Q-values 
        old_value = Q[state, action]
        next_max = np.max(Q[new_state])
        new_value = (1 - step_size)*old_value + step_size*(reward + discount_factor * next_max) # step_size defined in the cell below
        Q[state, action] = new_value
        
        state = new_state

        if done:
            break
            
    return Q, sum(rewards)

def train_q_learning(epsilon, env):
    
    Q = initialize_Q_table(env)

    for episode in range(n_episodes):
        print(f'Training on Episode {episode+1}... Epsilon: {epsilon}', end="\r")

        Q, reward = execute_episode(Q, n_steps)
        
        rewards_history.append(reward)
    
    return Q, rewards_history

def plot(n_episodes, rewards_history):
    plt.plot(range(n_episodes), rewards_history)
    plt.show()

if __name__ == '__main__':

    step_size = 0.7
    discount_factor = 0.8
    epsilon = 0.1
    n_episodes = 150
    n_steps = 200

    env = gym.make('gymnasium_env/CenturyGolem-v0', render_mode='text')
    rewards_history = list()
    
    trained_Q, rewards_history = train_q_learning(epsilon, env)