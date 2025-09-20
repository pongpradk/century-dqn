import gymnasium as gym
import gymnasium_env
from gymnasium.wrappers import FlattenObservation
import torch
import numpy as np
from random_agent import RandomAgent

# === Configuration for Data Collection ===
# Set these variables to configure the data collection
DQN_VERSION = "v9"           # The DQN version to evaluate
MODEL_VERSION = "1000"       # The model number to evaluate
EPISODES_TO_EVALUATE = 1000   # Number of games to evaluate
MAX_TIMESTEPS = 2000         # Maximum number of timesteps per game
RENDER = False

# DQN version to environment version mapping
DQN_ENV_MAPPING = {
    "v4": "gymnasium_env/CenturyGolem-v10",
    "v6": "gymnasium_env/CenturyGolem-v12",
    "v7": "gymnasium_env/CenturyGolem-v14",
    "v9": "gymnasium_env/CenturyGolem-v16",
    "v9_1": "gymnasium_env/CenturyGolem-v16",
    "v9_1_1": "gymnasium_env/CenturyGolem-v16",
}

def load_model(path, state_size, action_size, DQNClass):
    """Load a trained DQN model"""
    model = DQNClass(state_size, action_size)
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

def evaluate_model(model, env, opponent, num_episodes, max_timesteps):
    """Evaluate the model and collect statistics"""
    merchant_cards_per_game = []
    rounds_per_win = []
    wins = 0
    
    def select_action(state, model, info):
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = model(state_tensor)[0].numpy()
            valid_actions = info['valid_actions']
            masked_q_values = np.where(valid_actions == 1, q_values, -np.inf)
            return int(np.argmax(masked_q_values))
    
    for episode in range(num_episodes):
        state, info = env.reset()
        done = False
        total_reward = 0
        merchant_cards_acquired = 0
        
        for _ in range(max_timesteps):
            if info['current_player'] == 0:  # DQN agent's turn
                action = select_action(state, model, info)
                
                # Track merchant card acquisition (actions 1-43 are getM3-getM45)
                if 1 <= action <= 43:
                    merchant_cards_acquired += 1
                
                next_state, reward, done, _, info = env.step(action)
                total_reward += reward
                
                if not done and info['current_player'] == 1:
                    opponent_action = opponent.pick_action(next_state, info)
                    next_state, _, done, _, info = env.step(opponent_action)
                
                state = next_state
            elif info['current_player'] == 1:  # Opponent's turn
                opponent_action = opponent.pick_action(state, info)
                next_state, _, done, _, info = env.step(opponent_action)
                state = next_state
                
            if done:
                # Add merchant card count for this game
                merchant_cards_per_game.append(merchant_cards_acquired)
                
                # Check if agent won and record rounds if so
                if total_reward > 0:
                    wins += 1
                    rounds_per_win.append(info['round_number'])
                
                if RENDER:
                    print(f"Episode {episode+1}/{num_episodes} - Merchant Cards: {merchant_cards_acquired}, " + 
                      f"Rounds: {info['round_number']}, Won: {total_reward > 0}")
                
                break
    
    # Calculate statistics
    avg_merchant_cards = np.mean(merchant_cards_per_game) if merchant_cards_per_game else 0
    avg_rounds_per_win = np.mean(rounds_per_win) if rounds_per_win else 0
    win_rate = wins / num_episodes
    
    return {
        'avg_merchant_cards': avg_merchant_cards,
        'avg_rounds_per_win': avg_rounds_per_win,
        'win_rate': win_rate,
        'total_episodes': num_episodes,
        'wins': wins
    }

def main():
    # Dynamically set environment and imports based on DQN version
    ENV_VERSION = DQN_ENV_MAPPING[DQN_VERSION]
    DQN_MODULE = f"dqn_{DQN_VERSION}.dqn_{DQN_VERSION}"
    MODEL_PATH = f"dqn_{DQN_VERSION}/models/trained_model_{MODEL_VERSION}.pt"
    
    # Dynamically import DQN class
    DQN = __import__(DQN_MODULE, fromlist=["DQN"]).DQN
    
    # Setup environment
    env = gym.make(ENV_VERSION)
    env = FlattenObservation(env)
    state, _ = env.reset()
    state_size = len(state)
    action_size = env.action_space.n
    
    # Create opponent
    opponent = RandomAgent(action_size)
    
    # Load the model
    model = load_model(MODEL_PATH, state_size, action_size, DQN)
    
    print(f"Evaluating DQN {DQN_VERSION} model {MODEL_VERSION} over {EPISODES_TO_EVALUATE} episodes...")
    stats = evaluate_model(model, env, opponent, EPISODES_TO_EVALUATE, MAX_TIMESTEPS)
    
    # Print results
    print("\n=== Evaluation Results ===")
    print(f"Model: DQN {DQN_VERSION}, Model Version: {MODEL_VERSION}")
    print(f"Episodes Evaluated: {stats['total_episodes']}")
    print(f"Win Rate: {stats['win_rate']:.2f} ({stats['wins']}/{stats['total_episodes']})")
    print(f"Average Merchant Cards Acquired Per Game: {stats['avg_merchant_cards']:.2f}")
    print(f"Average Rounds Per Win: {stats['avg_rounds_per_win']:.2f}")

if __name__ == "__main__":
    main()
