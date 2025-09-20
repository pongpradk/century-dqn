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

# Configuration: Specify DQN version and model version here
DQN_VERSION = "v9_2"
MODEL_VERSION = "100"

# Add a flag to control win rate calculation
auto_calculate_winrate = True

# === Additional configuration for action distribution plotting ===
plot_action_distribution = False
episodes_for_action_count = 100
# Configuration variable for toggling bar chart orientation
horizontal_bar_chart = True  # Set to False for vertical bar chart

# Configuration for round counting
calculate_rounds = False  # Toggle for round counting feature
games_for_round_count = 1000  # Number of games to calculate average rounds

overwrite_action_count = True
action_count_dir = "action_count"

# Mapping of DQN versions to environment versions
DQN_ENV_MAPPING = {
    "v1": "gymnasium_env/CenturyGolem-v9",
    "v3": "gymnasium_env/CenturyGolem-v10",
    "v4": "gymnasium_env/CenturyGolem-v10",
    "v6": "gymnasium_env/CenturyGolem-v12",
    "v6_1": "gymnasium_env/CenturyGolem-v13",
    "v6_2": "gymnasium_env/CenturyGolem-v14",
    "v7": "gymnasium_env/CenturyGolem-v14",
    "v7_1": "gymnasium_env/CenturyGolem-v14",
    "v7_2": "gymnasium_env/CenturyGolem-v14",
    "v7_2": "gymnasium_env/CenturyGolem-v14",
    "v8": "gymnasium_env/CenturyGolem-v15",
    "v8_1": "gymnasium_env/CenturyGolem-v15",
    "v8_2": "gymnasium_env/CenturyGolem-v15",
    "v9": "gymnasium_env/CenturyGolem-v16",
    "v9_1": "gymnasium_env/CenturyGolem-v16",
    "v9_1_1": "gymnasium_env/CenturyGolem-v16",
    "v9_2": "gymnasium_env/CenturyGolem-v16",
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

# === Configuration ===
model_dir = os.path.join(DQN_MODULE.split('.')[0], "models")
max_episodes = 19000
episodes_per_eval = 100 # number of games
model_filename_format = "trained_model_{}.pt"
overwrite_existing = False
results_file = f"winrate_log_{DQN_VERSION}.json"

# === Utility to dynamically import DQN class ===
def import_dqn_class(dqn_import_path):
    dqn_module = importlib.import_module(dqn_import_path)
    return dqn_module.DQN

# === Load a single DQN model for evaluation ===
def load_model(path, state_size, action_size, DQNClass):
    model = DQNClass(state_size, action_size)
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

# === Select action from trained agent ===
def select_trained_agent_action(state, model, info):
    with torch.no_grad():
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = model(state_tensor)[0].numpy()
        valid_actions = info['valid_actions']
        masked_q_values = np.where(valid_actions == 1, q_values, -np.inf)
        return int(np.argmax(masked_q_values))

# === Evaluate a model by playing episodes ===
def evaluate_model(model, env, opponent, num_episodes):
    def select_action(state, model, info):
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = model(state_tensor)[0].numpy()
            valid_actions = info['valid_actions']
            masked_q_values = np.where(valid_actions == 1, q_values, -np.inf)
            return int(np.argmax(masked_q_values))

    wins = 0
    total_rounds = 0
    for _ in range(num_episodes):
        state, info = env.reset()
        done = False
        total_reward = 0
        rounds = 0

        for _ in range(2000):  # safety limit on steps
            rounds += 1
            if info['current_player'] == 0:
                action = select_action(state, model, info)
                next_state, reward, done, _, info = env.step(action)
                total_reward += reward

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

        if info.get('winner') == 'P1':
            wins += 1
        total_rounds += rounds

    return wins / num_episodes, total_rounds / num_episodes

# === Main Evaluation Loop ===
def main():
    if not auto_calculate_winrate:
        print("Win rate calculation is disabled.")
        return

    # Create evaluation environment
    env = gym.make(ENV_VERSION)
    env = FlattenObservation(env)
    state, _ = env.reset()
    state_size = len(state)
    action_size = env.action_space.n
    opponent = RandomAgent(action_size)

    # Load existing results if available
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            results = json.load(f)
    else:
        results = {}

    # Automatically detect models in the directory
    available_models = [
        int(f.split('_')[-1].split('.')[0])
        for f in os.listdir(model_dir)
        if f.startswith("trained_model_") and f.endswith(".pt")
    ]
    available_models = [ep for ep in available_models if ep <= max_episodes]
    available_models.sort()

    for episode in available_models:
        if not overwrite_existing and str(episode) in results:
            # print(f"Skipping episode {episode}, result already exists.")
            continue

        model_path = os.path.join(model_dir, model_filename_format.format(episode))
        if not os.path.exists(model_path):
            print(f"Model not found: {model_path}")
            continue

        model = load_model(model_path, state_size, action_size, DQN)
        win_rate, avg_rounds = evaluate_model(model, env, opponent, episodes_per_eval)

        print(f"Episode {episode}: Win rate = {win_rate:.2%}")
        if calculate_rounds:
            print(f"Average rounds per game = {avg_rounds:.2f}")
        results[str(episode)] = win_rate

        # Save results after each evaluation
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=4)

    env.close()

    # Plot win rate using results file data
    if auto_calculate_winrate:
        episode_counts = sorted([int(k) for k in results.keys()])
        win_rates = [results[str(ep)] for ep in episode_counts]

        plt.figure(figsize=(10, 5))
        plt.plot(episode_counts, win_rates, label="DQN")
        plt.xlabel("Training Episodes", fontsize=16)
        plt.ylabel("Win Rate of DQN Agent", fontsize=16)
        plt.ylim(0, 1)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"dqn_{DQN_VERSION}/results/dqn_{DQN_VERSION}_wr.png")
        plt.show()

    # Calculate and display average rounds if enabled
    if calculate_rounds:
        model = load_model(MODEL_PATH, state_size, action_size, DQN)
        _, avg_rounds = evaluate_model(model, env, opponent, games_for_round_count)
        print(f"\nAverage rounds per game over {games_for_round_count} games: {avg_rounds:.2f}")

if __name__ == "__main__":
    main()


# === Restore action distribution plotting ===
import sys
if plot_action_distribution:
    os.makedirs(action_count_dir, exist_ok=True)
    action_data_filename = f"{DQN_VERSION}_ep{MODEL_VERSION}.json"
    action_data_path = os.path.join(action_count_dir, action_data_filename)

    # Setup environment and model parameters
    env = gym.make(ENV_VERSION)
    env = FlattenObservation(env)
    state, _ = env.reset()
    state_size = len(state)
    action_size = env.action_space.n
    opponent = RandomAgent(action_size)

    if not overwrite_action_count and os.path.exists(action_data_path):
        with open(action_data_path, "r") as f:
            avg_action_counts = np.array(json.load(f))
    else:
        def evaluate_action_distribution(model, env, opponent, num_episodes, action_size):
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
                for _ in range(2000):
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

        model = load_model(MODEL_PATH, state_size, action_size, DQN)
        avg_action_counts = evaluate_action_distribution(model, env, opponent, episodes_for_action_count, action_size)
        with open(action_data_path, "w") as f:
            json.dump(avg_action_counts.tolist(), f, indent=4)

    # custom_order = [0] + list(range(9, 24)) + list(range(1, 9))
    # custom_order = [0] + list(range(24, 63)) + list(range(1, 24))
    custom_order = [0] + list(range(44, 125)) + list(range(1, 44))
    action_labels = [Actions(i).name for i in custom_order]
    avg_action_counts = avg_action_counts[custom_order]

    bar_colors = []
    # for i in custom_order:
    #     if i == 0:
    #         bar_colors.append("tab:blue")
    #     elif 9 <= i <= 18:
    #         bar_colors.append("tab:orange")
    #     elif 19 <= i <= 23:
    #         bar_colors.append("tab:green")
    #     elif 1 <= i <= 8:
    #         bar_colors.append("tab:red")
    #     else:
    #         bar_colors.append("gray")

    # for i in custom_order:
    #     if i == 0:
    #         bar_colors.append("tab:blue")
    #     elif 24 <= i <= 48:
    #         bar_colors.append("tab:orange")
    #     elif 49 <= i <= 62:
    #         bar_colors.append("tab:green")
    #     elif 1 <= i <= 23:
    #         bar_colors.append("tab:red")
    #     else:
    #         bar_colors.append("gray")

    for i in custom_order:
        if i == 0:
            bar_colors.append("tab:blue")
        elif 44 <= i <= 88:
            bar_colors.append("tab:orange")
        elif 89 <= i <= 124:
            bar_colors.append("tab:green")
        elif 1 <= i <= 43:
            bar_colors.append("tab:red")
        else:
            bar_colors.append("gray")

    # plt.figure(figsize=(12, 6))
    # plt.figure(figsize=(18, 6))
    plt.figure(figsize=(6, 30))
    if horizontal_bar_chart:
        plt.barh(action_labels, avg_action_counts, color=bar_colors)
        plt.xlabel("Average Count per Game", fontsize=18)
        plt.ylabel("Action", fontsize=18)
        plt.yticks(fontsize=16)
        plt.xticks(fontsize=16)
    else:
        plt.bar(action_labels, avg_action_counts, color=bar_colors)
        plt.ylabel("Average Count per Game", fontsize=18)
        plt.xlabel("Action", fontsize=18)
        plt.xticks(rotation=70, fontsize=16)
        plt.yticks(fontsize=16)
    # plt.title(f"Action Distribution (Model {MODEL_VERSION}, {episodes_for_action_count} games)", fontsize=16)
    plt.tight_layout()
    plt.savefig(f"dqn_{DQN_VERSION}_ep{MODEL_VERSION}_ad.png")
    plt.show()