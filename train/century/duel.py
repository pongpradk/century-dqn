import gymnasium as gym
import gymnasium_env
from gymnasium.wrappers import FlattenObservation
import torch
import numpy as np
import importlib

# Agent type configuration
# Options: "DQN", "Random", "Strategic"
AGENT_TYPE_P1 = "DQN"
AGENT_TYPE_P2 = "DQN"

# DQN configuration (only used if agent type is "DQN")
DQN_VERSION_P1 = "v9_2"
MODEL_VERSION_P1 = "1000"
DQN_VERSION_P2 = "v9_1_1"
MODEL_VERSION_P2 = "1000"

# Game configuration
RENDER_TEXT = False
NUM_GAMES = 1000  # Number of games to play
CALCULATE_WIN_RATE = True  # Whether to calculate and display win rate
MAX_TIMESTEPS = 2000  # Maximum timesteps per game

# Mapping of DQN versions to environment versions
DQN_ENV_MAPPING = {
    "v4": "gymnasium_env/CenturyGolem-v10",
    "v6": "gymnasium_env/CenturyGolem-v12",
    "v7": "gymnasium_env/CenturyGolem-v14",
    "v9": "gymnasium_env/CenturyGolem-v16",
    "v9_1": "gymnasium_env/CenturyGolem-v16",
    "v9_1_1": "gymnasium_env/CenturyGolem-v16",
    "v9_2": "gymnasium_env/CenturyGolem-v16",
}

# Use P1's DQN version if it's a DQN agent, otherwise default to v9 for the environment
ENV_VERSION = DQN_ENV_MAPPING[DQN_VERSION_P1] if AGENT_TYPE_P1 == "DQN" else "gymnasium_env/CenturyGolem-v16"

# Dynamically set model paths and import modules for DQN agents
P1_DQN_MODULE = f"dqn_{DQN_VERSION_P1}.dqn_{DQN_VERSION_P1}" if AGENT_TYPE_P1 == "DQN" else None
P1_MODEL_PATH = f"dqn_{DQN_VERSION_P1}/models/trained_model_{MODEL_VERSION_P1}.pt" if AGENT_TYPE_P1 == "DQN" else None

P2_DQN_MODULE = f"dqn_{DQN_VERSION_P2}.dqn_{DQN_VERSION_P2}" if AGENT_TYPE_P2 == "DQN" else None
P2_MODEL_PATH = f"dqn_{DQN_VERSION_P2}/models/trained_model_{MODEL_VERSION_P2}.pt" if AGENT_TYPE_P2 == "DQN" else None

# Import the Actions class for the chosen environment
ACTIONS_MODULE = f"gymnasium_env.envs.century_{ENV_VERSION.split('-')[-1].lower()}.enums"
Actions = __import__(ACTIONS_MODULE, fromlist=["Actions"]).Actions

def load_model(dqn_module, model_path, state_size, action_size):
    """Load a pretrained DQN model"""
    # Import the DQN class dynamically
    DQN = __import__(dqn_module, fromlist=["DQN"]).DQN
    model = DQN(state_size, action_size)
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set to evaluation mode
    return model

def load_agent(agent_type, dqn_module=None, model_path=None, state_size=None, action_size=None):
    """Load an agent based on the specified type"""
    if agent_type == "DQN":
        return load_model(dqn_module, model_path, state_size, action_size)
    elif agent_type == "Random":
        from random_agent import RandomAgent
        return RandomAgent(action_size)
    elif agent_type == "Strategic":
        from phase_agent import StrategicAgent
        return StrategicAgent(action_size)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

def select_agent_action(state, agent, agent_type, info):
    """Select an action using the appropriate method for the agent type"""
    if agent_type == "DQN":
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = agent(state_tensor)[0].numpy()
            valid_actions = info['valid_actions']
            masked_q_values = np.where(valid_actions == 1, q_values, -np.inf)
            return np.argmax(masked_q_values)
    else:
        # For Random and Strategic agents, use their pick_action method
        return agent.pick_action(state, info)

def display_valid_actions(info):
    valid_actions = info['valid_actions']
    valid_indices = np.where(valid_actions == 1)[0]  # Get indices of valid actions
    valid_actions = [f"{index}: {Actions(index).name}" for index in valid_indices]
    print(valid_actions)

def get_agent_display_name(agent_type, dqn_version=None, model_version=None):
    """Get a display name for the agent for logging purposes"""
    if agent_type == "DQN":
        return f"DQN {dqn_version}_m{model_version}"
    else:
        return agent_type

if __name__ == '__main__':
    # Create environment
    env = gym.make(ENV_VERSION, render_mode="text" if RENDER_TEXT else None)
    env = FlattenObservation(env)
    state, info = env.reset()
    
    # Get state and action sizes from environment
    state_size = len(state)
    action_size = env.action_space.n
    
    # Load the agents
    p1_agent = load_agent(
        AGENT_TYPE_P1, 
        P1_DQN_MODULE, 
        P1_MODEL_PATH, 
        state_size, 
        action_size
    )
    
    p2_agent = load_agent(
        AGENT_TYPE_P2, 
        P2_DQN_MODULE, 
        P2_MODEL_PATH, 
        state_size, 
        action_size
    )
    
    # Get display names for agents
    p1_display = get_agent_display_name(AGENT_TYPE_P1, DQN_VERSION_P1, MODEL_VERSION_P1)
    p2_display = get_agent_display_name(AGENT_TYPE_P2, DQN_VERSION_P2, MODEL_VERSION_P2)
    
    # Statistics
    p1_wins = 0
    p2_wins = 0
    ties = 0
    
    for game in range(NUM_GAMES):
        if RENDER_TEXT:
            print(f"\n===== GAME {game+1}/{NUM_GAMES} =====")
            
        state, info = env.reset()
        terminal = False
        timestep = 0
        
        # Execute Game
        while not terminal and timestep < MAX_TIMESTEPS:
            timestep += 1
            
            if info['current_player'] == 0:  # Player 1's turn
                if RENDER_TEXT:
                    print(f"\nPlayer 1's turn ({p1_display})")
                    display_valid_actions(info)
                    
                action = select_agent_action(state, p1_agent, AGENT_TYPE_P1, info)
                next_state, _, terminal, _, info = env.step(action)
                
                if not terminal and info['current_player'] == 1:
                    # Player 2's turn
                    if RENDER_TEXT:
                        print(f"\nPlayer 2's turn ({p2_display})")
                        display_valid_actions(info)
                        
                    p2_action = select_agent_action(next_state, p2_agent, AGENT_TYPE_P2, info)
                    next_state, _, terminal, _, info = env.step(p2_action)
                
                state = next_state
                
            elif info['current_player'] == 1:  # Player 2's turn (if they start)
                if RENDER_TEXT:
                    print(f"\nPlayer 2's turn ({p2_display})")
                    display_valid_actions(info)
                    
                p2_action = select_agent_action(state, p2_agent, AGENT_TYPE_P2, info)
                next_state, _, terminal, _, info = env.step(p2_action)
                state = next_state
        
        # Record game result
        if info.get('winner') == 'P1':
            p1_wins += 1
            if RENDER_TEXT:
                print(f"\nGame {game+1} Result: Player 1 wins!")
        elif info.get('winner') == 'P2':
            p2_wins += 1
            if RENDER_TEXT:
                print(f"\nGame {game+1} Result: Player 2 wins!")
        elif info.get('winner') == 'P0':
            ties += 1
            if RENDER_TEXT:
                print(f"\nGame {game+1} Result: Tie!")
    
    # Display final statistics
    if CALCULATE_WIN_RATE:
        p1_win_rate = p1_wins / NUM_GAMES
        p2_win_rate = p2_wins / NUM_GAMES
        tie_rate = ties / NUM_GAMES
        
        print("\n===== FINAL RESULTS =====")
        print(f"Player 1 ({p1_display}): {p1_wins} wins ({p1_win_rate:.2%})")
        print(f"Player 2 ({p2_display}): {p2_wins} wins ({p2_win_rate:.2%})")
        print(f"Ties: {ties} ({tie_rate:.2%})")
    
    env.close()
