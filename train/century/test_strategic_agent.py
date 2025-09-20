import gymnasium as gym
import gymnasium_env
from gymnasium.wrappers import FlattenObservation
import numpy as np
import time

from strategic_agent import StrategicAgent
from random_agent import RandomAgent

def run_test_games(num_games=10, num_timesteps=500, render=True):
    """
    Run test games between the StrategicAgent and RandomAgent
    to evaluate performance.
    """
    env = gym.make("gymnasium_env/CenturyGolem-v16", render_mode="text" if render else None)
    env = FlattenObservation(env)
    
    # Initialize agents
    action_size = env.action_space.n
    strategic_agent = StrategicAgent(action_size)
    random_agent = RandomAgent(action_size)
    
    # Track wins and points
    strategic_wins = 0
    random_wins = 0
    ties = 0
    strategic_total_points = 0
    random_total_points = 0
    
    for game in range(num_games):
        print(f"\n=== Game {game+1}/{num_games} ===")
        state, info = env.reset()
        
        terminated = False
        turn_count = 0
        
        for t in range(num_timesteps):
            turn_count += 1
            
            if info['current_player'] == 0:  # Strategic Agent's turn
                action = strategic_agent.pick_action(state, info)
                next_state, _, terminated, _, info = env.step(action)
                state = next_state
            else:  # Random Agent's turn
                action = random_agent.pick_action(state, info)
                next_state, _, terminated, _, info = env.step(action)
                state = next_state
                
            # if render:
            #     time.sleep(0.5)
            if terminated:
                break
        
        # Get final scores from info
        if 'winner' in info and info['winner'] is not None:
            if info['winner'] == 'P1':
                strategic_wins += 1
                print(f"Strategic Agent wins!")
            elif info['winner'] == 'P2':
                random_wins += 1
                print(f"Random Agent wins!")
            else:  # Tie
                ties += 1
                print(f"Game ended in a tie!")
        
        print(f"Turns taken: {turn_count}")
    
    # Print overall results
    print("\n=== Final Results ===")
    print(f"Games played: {num_games}")
    print(f"Strategic Agent wins: {strategic_wins} ({strategic_wins/num_games*100:.1f}%)")
    print(f"Random Agent wins: {random_wins} ({random_wins/num_games*100:.1f}%)")
    print(f"Ties: {ties} ({ties/num_games*100:.1f}%)")
    
    env.close()
    return strategic_wins, random_wins, ties

if __name__ == "__main__":
    # Run 5 test games with rendering
    run_test_games(num_games=100, render=None) 