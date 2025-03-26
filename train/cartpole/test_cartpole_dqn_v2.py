import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import json
import torch
from cartpole_dqn_v2 import DQN


def load_pretrained_model(path, state_size, action_size):
    """Load a pretrained PyTorch model from the path provided as parameter"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DQN(state_size, action_size).to(device)
    model.load_state_dict(torch.load(path))
    model.eval()  # Set the model to evaluation mode
    return model


def select_trained_agent_action(state, trained_model):
    """Uses the trained model to predict the action with highest Q-Value"""
    device = next(trained_model.parameters()).device
    with torch.no_grad():
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        q_values = trained_model(state)
        return q_values.argmax().item()


def plot_epsilon_values():
    """Plots the different values for Epsilon during the training"""
    with open('epsilon_values_v2.txt', 'r') as f:
        file = f.read()
        epsilon_values = json.loads(file)

    plt.plot(range(len(epsilon_values)), epsilon_values)
    plt.title('Epsilon Values During Training')
    plt.xlabel('Episode')
    plt.ylabel('Epsilon')
    plt.show()


def plot_rewards(calculate_mean=1):
    """Plots the rewards obtained during the training. The total number of episodes must be
    a multiple of 'calculate_mean' parameter"""
    with open('rewards_v2.txt', 'r') as f:
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
    plt.title('Mean Rewards During Training')
    plt.xlabel('Episode Group')
    plt.ylabel('Mean Reward')
    plt.show()


if __name__ == '__main__':
    # Initialize environment with rendering
    env = gym.make('CartPole-v1', render_mode='human')
    state, _ = env.reset()
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    terminal = False
    
    # Load the trained model
    trained_agent = load_pretrained_model('trained_agent_v2.pth', state_size, action_size)

    total_reward = 0
    max_timesteps = 500

    # Execute Episode
    for t in range(max_timesteps):
        env.render()
        action = select_trained_agent_action(state, trained_agent)
        next_state, reward, terminal, _, _ = env.step(action)
        total_reward += reward
        state = next_state

        if terminal:
            print(f'Episode terminated with total reward: {total_reward}')
            break

    print(f'Final total reward: {total_reward}')
    
    # Plot training results
    plot_rewards(1)
    plot_epsilon_values() 