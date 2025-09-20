import gymnasium as gym
import gymnasium_env
from gymnasium.wrappers import FlattenObservation
import torch
import numpy as np
import matplotlib.pyplot as plt
from random_agent import RandomAgent
import json
import os
import importlib
import math

# === Configuration for Action Distribution Plotting ===
# Set these variables to configure the plotting
DQN_VERSION = "v9"        # The DQN version to plot
MODEL_VERSION = "1000"          # The model number to plot
EPISODES_FOR_ACTION_COUNT = 1000  # Number of games to evaluate
MAX_TIMESTEPS = 2000           # Maximum number of timesteps per game
OVERWRITE_ACTION_COUNT = True  # Whether to overwrite existing data
ACTION_COUNT_DIR = "action_count"  # Directory to store action count data

# DQN version to environment version mapping
DQN_ENV_MAPPING = {
    "v4": "gymnasium_env/CenturyGolem-v10",
    "v6": "gymnasium_env/CenturyGolem-v12",
    "v7": "gymnasium_env/CenturyGolem-v14",
    "v9": "gymnasium_env/CenturyGolem-v16",
    "v9_1": "gymnasium_env/CenturyGolem-v16",
    "v9_1_1": "gymnasium_env/CenturyGolem-v16",
}

# Dynamically set environment and imports based on DQN version
ENV_VERSION = DQN_ENV_MAPPING[DQN_VERSION]
DQN_MODULE = f"dqn_{DQN_VERSION}.dqn_{DQN_VERSION}"
MODEL_PATH = f"dqn_{DQN_VERSION}/models/trained_model_{MODEL_VERSION}.pt"
ACTIONS_MODULE = f"gymnasium_env.envs.century_{ENV_VERSION.split('-')[-1].lower()}.enums"

# Load DQN class and Actions enum
DQN = __import__(DQN_MODULE, fromlist=["DQN"]).DQN
Actions = __import__(ACTIONS_MODULE, fromlist=["Actions"]).Actions

def load_model(path, state_size, action_size, DQNClass):
    """Load a trained DQN model"""
    model = DQNClass(state_size, action_size)
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

def evaluate_action_distribution(model, env, opponent, num_episodes, action_size, max_timesteps):
    """Evaluate and count action usage over multiple episodes"""
    action_counts = np.zeros(action_size, dtype=int)
    
    def select_action(state, model, info):
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = model(state_tensor)[0].numpy()
            valid_actions = info['valid_actions']
            masked_q_values = np.where(valid_actions == 1, q_values, -np.inf)
            return int(np.argmax(masked_q_values))
    
    for _ in range(num_episodes):
        state, info = env.reset()
        done = False
        
        for _ in range(max_timesteps):
            if info['current_player'] == 0:
                action = select_action(state, model, info)
                action_counts[action] += 1
                next_state, _, done, _, info = env.step(action)
                
                if not done and info['current_player'] == 1:
                    opponent_action = opponent.pick_action(next_state, info)
                    next_state, _, done, _, info = env.step(opponent_action)
                
                state = next_state
            elif info['current_player'] == 1:
                opponent_action = opponent.pick_action(state, info)
                next_state, _, done, _, info = env.step(opponent_action)
                state = next_state
                
            if done:
                break
                
    return action_counts / num_episodes

def plot_action_distribution():
    """Plot the action distribution as three separate vertical bar charts"""
    os.makedirs(ACTION_COUNT_DIR, exist_ok=True)
    action_data_filename = f"{DQN_VERSION}_ep{MODEL_VERSION}.json"
    action_data_path = os.path.join(ACTION_COUNT_DIR, action_data_filename)

    # Setup environment and model parameters
    env = gym.make(ENV_VERSION)
    env = FlattenObservation(env)
    state, _ = env.reset()
    state_size = len(state)
    action_size = env.action_space.n
    opponent = RandomAgent(action_size)

    if not OVERWRITE_ACTION_COUNT and os.path.exists(action_data_path):
        print(f"Loading existing action data from {action_data_path}")
        with open(action_data_path, "r") as f:
            avg_action_counts = np.array(json.load(f))
    else:
        print(f"Collecting action distribution data over {EPISODES_FOR_ACTION_COUNT} games...")
        model = load_model(MODEL_PATH, state_size, action_size, DQN)
        avg_action_counts = evaluate_action_distribution(
            model, env, opponent, EPISODES_FOR_ACTION_COUNT, action_size, MAX_TIMESTEPS
        )
        
        with open(action_data_path, "w") as f:
            json.dump(avg_action_counts.tolist(), f, indent=4)
        print(f"Action data saved to {action_data_path}")

    # Custom order for the actions: rest, useM, getG, getM
    custom_order = [0] + list(range(44, 89)) + list(range(89, 125)) + list(range(1, 44))
    action_labels = [Actions(i).name for i in custom_order]
    avg_action_counts = avg_action_counts[custom_order]

    # Set colors for different action groups
    bar_colors = []
    for i in custom_order:
        if i == 0:  # rest
            bar_colors.append("tab:blue")
        elif 44 <= i <= 88:  # useM (use merchant card)
            bar_colors.append("tab:orange")
        elif 89 <= i <= 124:  # getG (get golem card)
            bar_colors.append("tab:green")
        elif 1 <= i <= 43:  # getM (get merchant card)
            bar_colors.append("tab:red")
        else:
            bar_colors.append("gray")
    
    # Create legend patches
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='tab:blue', label='Rest'),
        Patch(facecolor='tab:orange', label='Use Merchant Card'),
        Patch(facecolor='tab:green', label='Get Golem Card'),
        Patch(facecolor='tab:red', label='Get Merchant Card')
    ]
    
    # Split the actions into 3 equal parts
    total_actions = len(action_labels)
    actions_per_chart = math.ceil(total_actions / 3)
    
    # Create the results directory if it doesn't exist
    os.makedirs(f"dqn_{DQN_VERSION}/results", exist_ok=True)
    
    # Calculate the maximum y-value across all sections for consistent y-axis
    max_y_value = max(avg_action_counts) * 1.1  # Add 10% padding
    
    # Create a single figure with 3 vertically stacked subplots
    fig, axes = plt.subplots(3, 1, figsize=(20, 24))
    # fig.suptitle(f"Action Distribution (Model {MODEL_VERSION}, {EPISODES_FOR_ACTION_COUNT} games)", fontsize=24, y=0.95)
    
    # Create 3 subplots for each section
    for i in range(3):
        start_idx = i * actions_per_chart
        end_idx = min((i + 1) * actions_per_chart, total_actions)
        
        if start_idx >= total_actions:
            break
            
        section_labels = action_labels[start_idx:end_idx]
        section_counts = avg_action_counts[start_idx:end_idx]
        section_colors = bar_colors[start_idx:end_idx]
        
        # Plot vertical bars
        axes[i].bar(section_labels, section_counts, color=section_colors)
        axes[i].set_xlabel("Action", fontsize=20)
        axes[i].set_ylabel("Average Count per Game", fontsize=20)
        
        # Set consistent y-axis limit for all plots
        axes[i].set_ylim(0, max_y_value)
        
        # Set chart title for each section
        if i == 0:
            # axes[i].set_title("Part 1: rest, useM1-41", fontsize=22)
            # Add legend
            # axes[i].legend(handles=legend_elements, loc='upper right', fontsize=20)
            pass
        if i == 1:
            # axes[i].set_title("Part 2: useM42-45, getG1-36, getM3-4", fontsize=22)
            # Add legend
            axes[i].legend(handles=legend_elements, loc='upper right', fontsize=20)
        if i == 2:
            # axes[i].set_title("Part 3: getM3-41", fontsize=22)
            # Add legend
            # axes[i].legend(handles=legend_elements, loc='upper right', fontsize=20)
            pass
        
        # Rotate x-tick labels for readability
        axes[i].tick_params(axis='x', rotation=70, labelsize=20)
        axes[i].tick_params(axis='y', labelsize=20)
    
    plt.tight_layout()
    
    # Save the combined figure as a single PNG file
    output_path = f"dqn_{DQN_VERSION}/results/{DQN_VERSION}_ep{MODEL_VERSION}_ad.png"
    plt.savefig(output_path)
    print(f"Combined plot saved as {output_path}")
    plt.close()  # Close the figure to free memory

if __name__ == "__main__":
    plot_action_distribution()
