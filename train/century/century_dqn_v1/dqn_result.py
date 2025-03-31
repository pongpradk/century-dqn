import gymnasium as gym
import gymnasium_env
from gymnasium_env.envs.century_v9.enums import Actions
import torch
import numpy as np
import matplotlib.pyplot as plt
import json
from dqn_v1 import DQN
from gymnasium.wrappers import FlattenObservation
from random_agent import RandomAgent


def load_pretrained_model(path, state_size, action_size):
    
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
    # Initialize statistics
    dqn_wins = 0
    random_wins = 0
    ties = 0
    
    # Create environment for first game with render mode
    env = gym.make('gymnasium_env/CenturyGolem-v9', render_mode='text')
    env = FlattenObservation(env)
    state, info = env.reset()
    
    state_size = len(state)
    action_size = env.action_space.n
    
    # Load the trained model
    trained_agent = load_pretrained_model('models/trained_model_1000.pt', state_size, action_size)
    
    # Run 100 games
    for game_num in range(100):
        if game_num > 0:
            # For games 2-100, create new environment without render mode
            env.close()
            env = gym.make('gymnasium_env/CenturyGolem-v9', render_mode=None)
            env = FlattenObservation(env)
            state, info = env.reset()
        
        opponent = RandomAgent(env.action_space.n)
        max_timesteps = 2000
        
        # Execute Episode
        for t in range(max_timesteps):
            if info['current_player'] == 0:
                # DQN agent's turn
                if game_num == 0:
                    display_valid_actions(info)
                action = select_trained_agent_action(state, trained_agent, info)
                next_state, reward, terminal, _, info = env.step(action)
                
                if not terminal and info['current_player'] == 1:
                    # Opponent's turn
                    if game_num == 0:
                        display_valid_actions(info)
                    opponent_action = opponent.pick_action(next_state, info)
                    next_state, opponent_reward, terminal, _, info = env.step(opponent_action)
                    state = next_state
                else:
                    state = next_state
            
            elif info['current_player'] == 1:
                # Opponent's turn
                if game_num == 0:
                    display_valid_actions(info)
                opponent_action = opponent.pick_action(state, info)
                next_state, opponent_reward, terminal, _, info = env.step(opponent_action)
                state = next_state

            if terminal:
                # Update statistics based on winner
                if info['winner'] == "P1":
                    dqn_wins += 1
                elif info['winner'] == "P2":
                    random_wins += 1
                else:  # P0
                    ties += 1
                
                if game_num == 0:
                    print(f'Game {game_num + 1} terminated')
                    print(f'Winner: {info["winner"]}')
                break
    
    # Close the final environment
    env.close()
    
    # Print final statistics
    print("\nEvaluation Results:")
    print(f"Total Games: 100")
    print(f"DQN Agent Wins: {dqn_wins}")
    print(f"Random Agent Wins: {random_wins}")
    print(f"Ties: {ties}")
    print(f"Win Rate: {(dqn_wins / 100) * 100:.2f}%")