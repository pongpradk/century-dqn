import gymnasium as gym
import gymnasium_env
from gymnasium.wrappers import FlattenObservation
import keras
import numpy as np
import matplotlib.pyplot as plt
import json


def load_pretrained_model(path):
    """Load a pretrained model from the path provided as parameter"""
    return keras.models.load_model(path, compile=False)

def select_trained_agent_action(state, trained_model):
    """Uses the trained model to predict the action with highest Q-Value"""
    # q_values = trained_model.predict(state, verbose=0)
    # return np.argmax(q_values[0])
    q_values = trained_agent.predict(state, verbose=0)[0]
    print("full q_values: ", q_values)
    # Mask invalid actions
    masked_q_values = np.where(info['valid_actions'] == 1, q_values, -np.inf)
    return np.argmax(masked_q_values)  # Select highest Q-value among valid actions

class RandomAgent:
    def __init__(self, action_size):
        self.action_size = action_size

    def pick_random_action(self, valid_actions):
        valid_indices = np.where(valid_actions == 1)[0]
        return np.random.choice(valid_indices)

def plot_epsilon_values():
    """Plots the different values for Epsilon during the training"""
    with open('epsilon_valuesHighE.txt', 'r') as f:
        file = f.read()
        epsilon_values = json.loads(file)

    plt.plot(range(len(epsilon_values)), epsilon_values)
    plt.show()


def plot_rewards(calculate_mean=1):
    """Plots the rewards obtained during the training. The total number of episodes must be
    a multiple of 'calculate_mean' parameter"""
    with open('rewardsHighE.txt', 'r') as f:
        file = f.read()
        rewards = json.loads(file)

    if len(rewards) % calculate_mean != 0:
        raise ValueError

    # Calculate mean of rewards
    mean_rewards = list()
    for i in range(round(len(rewards)/calculate_mean)):
        reward_range = rewards[i*calculate_mean:i*calculate_mean+calculate_mean]
        reward_mean = round(sum(reward_range)/len(reward_range))
        mean_rewards.append(reward_mean)

    plt.plot(range(len(mean_rewards)), mean_rewards)
    plt.show()


if __name__ == '__main__':

    env = gym.make('gymnasium_env/CenturyGolem-v6', render_mode='text')
    env = FlattenObservation(env)
    state, info = env.reset()
    
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    terminal = False
    trained_agent = load_pretrained_model('checkpoint_main_model.keras')
    random_agent = RandomAgent(action_size)

    total_reward = 0
    max_timesteps = 200
    
    for t in range(max_timesteps):
        # Check which agent's turn it is
        current_player = info.get("current_player", 0)  # 0 = DQNAgent, 1 = RandomAgent

        # === DQNAgent's Turn ===
        if current_player == 0:
            print(info['valid_actions'])
            # print Q-values for each action
            state_input = state.reshape((1, state_size))  # Reshape for NN input
            action = select_trained_agent_action(state_input, trained_agent)
            next_state, reward, terminal, _, info = env.step(action)
            
            total_reward += reward
        # === RandomAgent's Turn ===
        else:
            print(info['valid_actions'])
            valid_mask = info["valid_actions"]
            action = random_agent.pick_random_action(valid_mask)
            next_state, reward, terminal, _, info = env.step(action)
            
            if terminal:
                total_reward += reward

        # Update state
        state = next_state

        # Check for terminal condition
        if terminal:
            print(f"Episode ended after {t+1} timesteps with Total Reward: {total_reward}")
            break

    env.close()
        
    print(f"Total Reward: {total_reward}")
    plot_rewards(1)
    plot_epsilon_values()
