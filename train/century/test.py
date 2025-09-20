import gymnasium as gym
import gymnasium_env
from gymnasium.wrappers import FlattenObservation
import numpy as np
from random_agent import RandomAgent
from greedy_agent import GreedyAgent

def play_with_random_agent(env_version="gymnasium_env/CenturyGolem-v14", max_timesteps=2000):
    """
    Play a game using a RandomAgent against another RandomAgent.
    
    Args:
        env_version (str): The version of the environment to use
        max_timesteps (int): Maximum number of timesteps to play
    
    Returns:
        tuple: (total_reward, timesteps_played)
    """
    # Create environment
    env = gym.make(env_version, render_mode='text')
    env = FlattenObservation(env)
    state, info = env.reset()
    
    # Create random agents for both players
    agent1 = GreedyAgent(env.action_space.n)
    agent2 = RandomAgent(env.action_space.n)
    
    total_reward = 0
    timesteps_played = 0

    # Execute Episode
    for t in range(max_timesteps):
        if info['current_player'] == 0:
            # Agent 1's turn
            action = agent1.pick_action(state, info)
            next_state, reward, terminal, _, info = env.step(action)
            total_reward += reward
            
            if not terminal and info['current_player'] == 1:
                # Agent 2's turn
                action = agent2.pick_action(next_state, info)
                next_state, _, terminal, _, info = env.step(action)
                state = next_state
            else:
                state = next_state
        
        elif info['current_player'] == 1:
            # Agent 2's turn
            action = agent2.pick_action(state, info)
            next_state, _, terminal, _, info = env.step(action)
            state = next_state

        timesteps_played += 1
        if terminal:
            break
    
    env.close()
    return total_reward, timesteps_played

if __name__ == '__main__':
    # Example usage
    reward, timesteps = play_with_random_agent()
    print(f'Game finished with reward: {reward} after {timesteps} timesteps')
