import gymnasium as gym
import gymnasium_env
from gymnasium_env.envs.century_v9.enums import Actions
import torch
import numpy as np
import matplotlib.pyplot as plt
import json
from century_dqn6.dqn6 import DQN
from gymnasium.wrappers import FlattenObservation
from random_agent import RandomAgent


def load_pretrained_model(path):
    """Load a pretrained DQN model from the path provided as parameter"""
    # Get state and action size from environment
    env = gym.make('gymnasium_env/CenturyGolem-v11')
    env = FlattenObservation(env)
    state, _ = env.reset()
    state_size = len(state)
    action_size = env.action_space.n
    env.close()
    
    # Create and load the model
    model = DQN(state_size, action_size)
    model.load_state_dict(torch.load(path))
    model.eval()  # Set to evaluation mode
    return model


def select_trained_agent_action(state, trained_model, info):
    """Uses the trained model to predict the action with highest Q-Value among valid actions"""
    with torch.no_grad():
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = trained_model(state_tensor)[0].numpy()
        valid_actions = info['valid_actions']
        masked_q_values = np.where(valid_actions == 1, q_values, -np.inf)
        return np.argmax(masked_q_values)


def display_valid_actions(info):
    valid_actions = info['valid_actions']
    valid_indices = np.where(valid_actions == 1)[0]  # Get indices of valid actions
    valid_actions = [f"{index}: {Actions(index).name}" for index in valid_indices]
    print(valid_actions)


if __name__ == '__main__':
    # Create environment
    env = gym.make('gymnasium_env/CenturyGolem-v11', render_mode='text')
    env = FlattenObservation(env)
    state, info = env.reset()
    
    # Load the trained model
    trained_agent = load_pretrained_model('century_dqn6/models/trained_model_3000.pt')  # Adjust path as needed
    
    total_reward = 0
    max_timesteps = 2000
    opponent = RandomAgent(env.action_space.n)  # Create random opponent

    # Execute Episode
    for t in range(max_timesteps):
        # env.render()
        
        if info['current_player'] == 0:
            # DQN agent's turn
            display_valid_actions(info)
            action = select_trained_agent_action(state, trained_agent, info)
            next_state, reward, terminal, _, info = env.step(action)
            total_reward += reward
            
            if not terminal and info['current_player'] == 1:
                # Opponent's turn
                display_valid_actions(info)
                opponent_action = opponent.pick_action(next_state, info)
                next_state, opponent_reward, terminal, _, info = env.step(opponent_action)
                state = next_state
            else:
                state = next_state
        
        elif info['current_player'] == 1:
            # Opponent's turn
            display_valid_actions(info)
            opponent_action = opponent.pick_action(state, info)
            next_state, opponent_reward, terminal, _, info = env.step(opponent_action)
            state = next_state

        if terminal:
            print(f'Episode terminated with Reward: {total_reward}')
            break