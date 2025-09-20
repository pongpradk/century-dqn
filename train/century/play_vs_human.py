import gymnasium as gym
import gymnasium_env
from gymnasium.wrappers import FlattenObservation
import torch
import numpy as np

# Configuration: Specify DQN version and model version here
DQN_VERSION = "v7"
MODEL_VERSION = "4000"

# Mapping of DQN versions to environment versions
DQN_ENV_MAPPING = {
    "v1": "gymnasium_env/CenturyGolem-v9",
    "v3": "gymnasium_env/CenturyGolem-v10",
    "v4": "gymnasium_env/CenturyGolem-v10",
    "v5_1": "gymnasium_env/CenturyGolem-v11",
    "v6": "gymnasium_env/CenturyGolem-v12",
    "v6_1": "gymnasium_env/CenturyGolem-v13",
    "v7": "gymnasium_env/CenturyGolem-v14",
    "v7_1": "gymnasium_env/CenturyGolem-v14",
}

# Dynamically set environment, DQN imports, and Actions import based on DQN version
ENV_VERSION = DQN_ENV_MAPPING[DQN_VERSION]
# DQN_MODULE = f"century_dqn_{DQN_VERSION}.dqn_{DQN_VERSION.replace('_', '')}"
# MODEL_PATH = f"century_dqn_{DQN_VERSION}/models/trained_model_{MODEL_VERSION}.pt"
DQN_MODULE = f"dqn_{DQN_VERSION}.dqn_{DQN_VERSION}"
MODEL_PATH = f"dqn_{DQN_VERSION}/models/trained_model_{MODEL_VERSION}.pt"
ACTIONS_MODULE = f"gymnasium_env.envs.century_{ENV_VERSION.split('-')[-1].lower()}.enums"

# Import the correct DQN class dynamically
DQN = __import__(DQN_MODULE, fromlist=["DQN"]).DQN

# Import the correct Actions class dynamically
Actions = __import__(ACTIONS_MODULE, fromlist=["Actions"]).Actions

def load_pretrained_model(path):
    """Load a pretrained DQN model from the path provided as parameter"""
    # Get state and action size from environment
    env = gym.make(ENV_VERSION)
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
    env = gym.make(ENV_VERSION, render_mode='text')
    env = FlattenObservation(env)
    state, info = env.reset()
    
    # Load the trained model
    trained_agent = load_pretrained_model(MODEL_PATH)
    
    total_reward = 0
    max_timesteps = 2000

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
                while True:
                    opponent_action = input("Enter opponent action: ")
                    opponent_action = int(opponent_action)
                    if info['valid_actions'][opponent_action] == 1:
                        break
                    else:
                        print("Invalid action. Please enter a valid action number.")
                next_state, opponent_reward, terminal, _, info = env.step(opponent_action)
                state = next_state
            else:
                state = next_state
        
        elif info['current_player'] == 1:
            # Opponent's turn
            display_valid_actions(info)
            opponent_action = int(input("Enter opponent action: "))
            next_state, opponent_reward, terminal, _, info = env.step(opponent_action)
            state = next_state

        if terminal:
            print(f'Episode terminated with Reward: {total_reward}')
            break