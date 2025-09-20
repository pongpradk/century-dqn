import gymnasium as gym
import gymnasium_env
from gymnasium.wrappers import FlattenObservation
import numpy as np
import torch
import time
import argparse
import os

from strategic_agent import StrategicAgent
from random_agent import RandomAgent

def load_dqn_model(model_path, state_size, action_size):
    """Load a trained DQN model from a saved file"""
    from dqn_v9.dqn_v9 import DQN
    
    model = DQN(state_size, action_size)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()  # Set to evaluation mode
    
    return model

def evaluate_agent(model, opponent, num_games=100, render=False):
    """Evaluate a trained DQN model against a specific opponent"""
    env = gym.make("gymnasium_env/CenturyGolem-v16", render_mode="text" if render else None)
    env = FlattenObservation(env)
    
    # Track performance metrics
    wins = 0
    losses = 0
    ties = 0
    dqn_points = []
    opponent_points = []
    game_lengths = []
    
    for game in range(num_games):
        if render:
            print(f"\n=== Game {game+1}/{num_games} ===")
        
        state, info = env.reset()
        
        terminated = False
        turn_count = 0
        
        while not terminated:
            turn_count += 1
            
            if info['current_player'] == 0:  # DQN Agent's turn
                # Pick action using the model
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0)
                    q_values = model(state_tensor)[0].cpu().numpy()
                    valid_actions = info['valid_actions']
                    masked_q_values = np.where(valid_actions == 1, q_values, -np.inf)
                    action = int(np.argmax(masked_q_values))
                
                next_state, _, terminated, _, info = env.step(action)
                state = next_state
            else:  # Opponent's turn
                action = opponent.pick_action(state, info)
                next_state, _, terminated, _, info = env.step(action)
                state = next_state
            
            if render:
                time.sleep(0.5)  # Brief pause to make it easier to follow
        
        # Get final scores
        if 'winner' in info and info['winner'] is not None:
            # We need to extract the actual scores from the environment
            # In v16 env, the final scores are stored as player1_final_points and player2_final_points
            if hasattr(env.unwrapped, 'player1_final_points') and hasattr(env.unwrapped, 'player2_final_points'):
                p1_points = env.unwrapped.player1_final_points
                p2_points = env.unwrapped.player2_final_points
            else:
                # Fallback - extract from state if available
                p1_points = None
                p2_points = None
            
            dqn_points.append(p1_points)
            opponent_points.append(p2_points)
            
            if info['winner'] == 'P1':
                wins += 1
                if render:
                    print(f"DQN Agent wins! ({p1_points} vs {p2_points})")
            elif info['winner'] == 'P2':
                losses += 1
                if render:
                    print(f"Opponent wins! ({p1_points} vs {p2_points})")
            else:  # Tie
                ties += 1
                if render:
                    print(f"Game ended in a tie! ({p1_points} vs {p2_points})")
        
        game_lengths.append(turn_count)
        
        if render:
            print(f"Turns taken: {turn_count}")
    
    # Calculate metrics
    win_rate = wins / num_games
    avg_game_length = sum(game_lengths) / len(game_lengths)
    
    # Calculate average points if available
    avg_dqn_points = sum(dqn_points) / len(dqn_points) if dqn_points and all(p is not None for p in dqn_points) else None
    avg_opponent_points = sum(opponent_points) / len(opponent_points) if opponent_points and all(p is not None for p in opponent_points) else None
    
    env.close()
    
    return {
        'win_rate': win_rate,
        'wins': wins,
        'losses': losses,
        'ties': ties,
        'avg_game_length': avg_game_length,
        'avg_dqn_points': avg_dqn_points,
        'avg_opponent_points': avg_opponent_points
    }

def run_comparison(random_model_path, strategic_model_path, num_games=100, render=False):
    """Compare the performance of two DQN models against different opponents"""
    env = gym.make("gymnasium_env/CenturyGolem-v16")
    env = FlattenObservation(env)
    state, _ = env.reset()
    
    # Get state and action sizes
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    # Load models
    random_trained_model = load_dqn_model(random_model_path, state_size, action_size)
    strategic_trained_model = load_dqn_model(strategic_model_path, state_size, action_size)
    
    # Create opponents
    random_opponent = RandomAgent(action_size)
    strategic_opponent = StrategicAgent(action_size)
    
    print("\n===== Evaluating Random-Trained DQN vs Random Opponent =====")
    random_vs_random = evaluate_agent(random_trained_model, random_opponent, num_games, render)
    
    print("\n===== Evaluating Random-Trained DQN vs Strategic Opponent =====")
    random_vs_strategic = evaluate_agent(random_trained_model, strategic_opponent, num_games, render)
    
    print("\n===== Evaluating Strategic-Trained DQN vs Random Opponent =====")
    strategic_vs_random = evaluate_agent(strategic_trained_model, random_opponent, num_games, render)
    
    print("\n===== Evaluating Strategic-Trained DQN vs Strategic Opponent =====")
    strategic_vs_strategic = evaluate_agent(strategic_trained_model, strategic_opponent, num_games, render)
    
    # Print comparison results
    print("\n===== COMPARISON RESULTS =====")
    print(f"Games per matchup: {num_games}")
    
    print("\nRandom-Trained DQN Performance:")
    print(f"  vs Random: Win Rate = {random_vs_random['win_rate']:.2f} ({random_vs_random['wins']}/{num_games})")
    print(f"  vs Strategic: Win Rate = {random_vs_strategic['win_rate']:.2f} ({random_vs_strategic['wins']}/{num_games})")
    
    print("\nStrategic-Trained DQN Performance:")
    print(f"  vs Random: Win Rate = {strategic_vs_random['win_rate']:.2f} ({strategic_vs_random['wins']}/{num_games})")
    print(f"  vs Strategic: Win Rate = {strategic_vs_strategic['win_rate']:.2f} ({strategic_vs_strategic['wins']}/{num_games})")
    
    print("\nDetailed Metrics:")
    print("                        | Avg Game Length | Avg DQN Points | Avg Opponent Points")
    print("------------------------+----------------+----------------+-------------------")
    print(f"Random DQN vs Random    | {random_vs_random['avg_game_length']:.1f}           | {random_vs_random.get('avg_dqn_points', 'N/A')}           | {random_vs_random.get('avg_opponent_points', 'N/A')}")
    print(f"Random DQN vs Strategic | {random_vs_strategic['avg_game_length']:.1f}           | {random_vs_strategic.get('avg_dqn_points', 'N/A')}           | {random_vs_strategic.get('avg_opponent_points', 'N/A')}")
    print(f"Strategic DQN vs Random | {strategic_vs_random['avg_game_length']:.1f}           | {strategic_vs_random.get('avg_dqn_points', 'N/A')}           | {strategic_vs_random.get('avg_opponent_points', 'N/A')}")
    print(f"Strategic DQN vs Strat  | {strategic_vs_strategic['avg_game_length']:.1f}           | {strategic_vs_strategic.get('avg_dqn_points', 'N/A')}           | {strategic_vs_strategic.get('avg_opponent_points', 'N/A')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare DQN agents trained against different opponents')
    parser.add_argument('--random-model', type=str, default='dqn_v9/models/trained_model_1000.pt', 
                        help='Path to the DQN model trained against random opponent')
    parser.add_argument('--strategic-model', type=str, default='dqn_v9/models_strategic/trained_model_1000.pt', 
                        help='Path to the DQN model trained against strategic opponent')
    parser.add_argument('--games', type=int, default=20, 
                        help='Number of games to play for each evaluation')
    parser.add_argument('--render', action='store_true', 
                        help='Render the games (text mode)')
    
    args = parser.parse_args()
    
    # Verify model files exist
    if not os.path.exists(args.random_model):
        print(f"Error: Random model file not found at {args.random_model}")
        exit(1)
    
    if not os.path.exists(args.strategic_model):
        print(f"Error: Strategic model file not found at {args.strategic_model}")
        exit(1)
    
    run_comparison(args.random_model, args.strategic_model, args.games, args.render) 