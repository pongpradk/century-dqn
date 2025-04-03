import gymnasium as gym
import gymnasium_env
from gymnasium_env.envs.century_v9.enums import Actions
import torch
import numpy as np
from century_dqn5.dqn5 import DQN
from gymnasium.wrappers import FlattenObservation
from random_agent import RandomAgent

NUM_GAMES = 1000

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
    
    total_action = 0
    action_count = {
        0: 0,
        1: 0,
        2: 0,
        3: 0,
        4: 0,
        5: 0,
        6: 0,
        7: 0,
        8: 0,
        9: 0,
        10: 0,
        11: 0,
        12: 0,
        13: 0,
        14: 0,
        15: 0,
        16: 0,
        17: 0,
        18: 0,
        19: 0,
        20: 0,
        21: 0,
        22: 0,
        23: 0,
    }
    
    final_yellow_crystals = 0
    final_green_crystals = 0
    
    # Create environment for first game with render mode
    env = gym.make('gymnasium_env/CenturyGolem-v11', render_mode=None)
    env = FlattenObservation(env)
    state, info = env.reset()
    
    state_size = len(state)
    action_size = env.action_space.n
    
    # Load the trained model
    trained_agent = load_pretrained_model('century_dqn5/models/trained_model_400.pt', state_size, action_size)
    
    for game_num in range(NUM_GAMES):
        
        opponent = RandomAgent(env.action_space.n)
        max_timesteps = 2000
        
        # Execute Episode
        for t in range(max_timesteps):
            if info['current_player'] == 0:
                # DQN agent's turn
                action = select_trained_agent_action(state, trained_agent, info)
                total_action += 1
                action_count[action] += 1
                next_state, reward, terminal, _, info = env.step(action)
                
                if not terminal and info['current_player'] == 1:
                    # Opponent's turn
                    opponent_action = opponent.pick_action(next_state, info)
                    next_state, opponent_reward, terminal, _, info = env.step(opponent_action)
                    state = next_state
                else:
                    state = next_state
            
            elif info['current_player'] == 1:
                # Opponent's turn
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
                
                final_yellow_crystals += info['final_yellow_crystals']
                final_green_crystals += info['final_green_crystals']
                
                break
    
    # Close the final environment
    env.close()
    
    # Print final statistics
    print("\nEvaluation Results:")
    print(f"Total Games: {NUM_GAMES}")
    print(f"DQN Agent Wins: {dqn_wins}")
    print(f"Random Agent Wins: {random_wins}")
    print(f"Ties: {ties}")
    print(f"Win Rate: {(dqn_wins / NUM_GAMES) * 100:.2f}%")
    
    print("\nAction Distribution (DQN Agent):")
    for action in action_count:
        action_count[action] = (action_count[action] / total_action) * 100
        print(f"{Actions(int(action)).name}: {action_count[action]:.2f}%")
        
    final_yellow_crystals = final_yellow_crystals / NUM_GAMES
    final_green_crystals = final_green_crystals / NUM_GAMES
    print(f"\nAverage Final Yellow Crystals: {final_yellow_crystals}")
    print(f"Average Final Green Crystals: {final_green_crystals}")