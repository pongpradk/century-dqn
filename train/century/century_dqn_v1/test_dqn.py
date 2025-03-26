import gymnasium as gym
import gymnasium_env
from gymnasium.wrappers import FlattenObservation
import numpy as np
import torch
import matplotlib.pyplot as plt
import json
import random
import time
from collections import deque

from dqn_trainer import DQN, DQNAgent, RandomAgent


def load_trained_model(model_name):
    """Load a trained DQN model for evaluation"""
    # Initialize environment to get state and action sizes
    env = gym.make("gymnasium_env/CenturyGolem-v9")
    env = FlattenObservation(env)
    
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    # Create agent and load trained model
    agent = DQNAgent(state_size, action_size)
    agent.load_agent(model_name)
    
    # Set epsilon to minimum for evaluation (mostly greedy actions)
    agent.epsilon = agent.epsilon_min
    
    return agent, env


def evaluate_against_random(agent, env, num_episodes=100, render_mode=None):
    """Evaluate the trained agent against a random opponent"""
    opponent = RandomAgent(env.action_space.n)
    
    wins = 0
    draws = 0
    total_rewards = []
    opponent_rewards = []
    
    for ep in range(num_episodes):
        # Reset environment
        if render_mode and ep < 5:  # Only render first 5 episodes to save time
            env.render_mode = render_mode
        else:
            env.render_mode = None
            
        state, info = env.reset()
        
        dqn_total_reward = 0
        opponent_total_reward = 0
        
        done = False
        while not done:
            # DQN agent's turn (player 1)
            if info['current_player'] == 0:
                action = agent.pick_action(state, info)
                next_state, reward, terminal, _, info = env.step(action)
                
                dqn_total_reward += reward
                state = next_state
                done = terminal
            
            # Random agent's turn (player 2)
            elif info['current_player'] == 1 and not done:
                action = opponent.pick_action(state, info)
                next_state, reward, terminal, _, info = env.step(action)
                
                opponent_total_reward += reward
                state = next_state
                done = terminal
        
        # Record results
        total_rewards.append(dqn_total_reward)
        opponent_rewards.append(opponent_total_reward)
        
        if dqn_total_reward > opponent_total_reward:
            wins += 1
        elif dqn_total_reward == opponent_total_reward:
            draws += 1
            
        if (ep + 1) % 10 == 0:
            print(f"Episode {ep+1}/{num_episodes} - DQN: {dqn_total_reward:.2f}, Random: {opponent_total_reward:.2f}")
            
    # Calculate statistics
    win_rate = wins / num_episodes
    draw_rate = draws / num_episodes
    loss_rate = 1 - win_rate - draw_rate
    avg_reward = sum(total_rewards) / num_episodes
    
    print(f"\nEvaluation Results against Random Agent:")
    print(f"Win Rate: {win_rate:.2f} ({wins}/{num_episodes})")
    print(f"Draw Rate: {draw_rate:.2f} ({draws}/{num_episodes})")
    print(f"Loss Rate: {loss_rate:.2f} ({num_episodes - wins - draws}/{num_episodes})")
    print(f"Average DQN Reward: {avg_reward:.2f}")
    
    return {
        'win_rate': win_rate,
        'draw_rate': draw_rate,
        'loss_rate': loss_rate,
        'avg_reward': avg_reward,
        'rewards': total_rewards,
        'opponent_rewards': opponent_rewards
    }


def evaluate_against_model(agent, opponent_model, env, num_episodes=100, render_mode=None):
    """Evaluate the agent against another trained model"""
    # Load opponent model
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    opponent = DQNAgent(state_size, action_size)
    opponent.load_agent(opponent_model)
    opponent.epsilon = opponent.epsilon_min  # Set to evaluation mode
    
    wins = 0
    draws = 0
    total_rewards = []
    opponent_rewards = []
    
    for ep in range(num_episodes):
        # Reset environment
        if render_mode and ep < 5:  # Only render first 5 episodes
            env.render_mode = render_mode
        else:
            env.render_mode = None
            
        state, info = env.reset()
        
        dqn_total_reward = 0
        opponent_total_reward = 0
        
        done = False
        while not done:
            # DQN agent's turn (player 1)
            if info['current_player'] == 0:
                action = agent.pick_action(state, info)
                next_state, reward, terminal, _, info = env.step(action)
                
                dqn_total_reward += reward
                state = next_state
                done = terminal
            
            # Opponent model's turn (player 2)
            elif info['current_player'] == 1 and not done:
                action = opponent.pick_action(state, info)
                next_state, reward, terminal, _, info = env.step(action)
                
                opponent_total_reward += reward
                state = next_state
                done = terminal
        
        # Record results
        total_rewards.append(dqn_total_reward)
        opponent_rewards.append(opponent_total_reward)
        
        if dqn_total_reward > opponent_total_reward:
            wins += 1
        elif dqn_total_reward == opponent_total_reward:
            draws += 1
            
        if (ep + 1) % 10 == 0:
            print(f"Episode {ep+1}/{num_episodes} - Main DQN: {dqn_total_reward:.2f}, Opponent DQN: {opponent_total_reward:.2f}")
            
    # Calculate statistics
    win_rate = wins / num_episodes
    draw_rate = draws / num_episodes
    loss_rate = 1 - win_rate - draw_rate
    avg_reward = sum(total_rewards) / num_episodes
    
    print(f"\nEvaluation Results against Model {opponent_model}:")
    print(f"Win Rate: {win_rate:.2f} ({wins}/{num_episodes})")
    print(f"Draw Rate: {draw_rate:.2f} ({draws}/{num_episodes})")
    print(f"Loss Rate: {loss_rate:.2f} ({num_episodes - wins - draws}/{num_episodes})")
    print(f"Average DQN Reward: {avg_reward:.2f}")
    
    return {
        'win_rate': win_rate,
        'draw_rate': draw_rate,
        'loss_rate': loss_rate,
        'avg_reward': avg_reward,
        'rewards': total_rewards,
        'opponent_rewards': opponent_rewards
    }


def plot_training_progress(model_name):
    """Plot training progress from saved files"""
    # Load data
    try:
        with open(f'{model_name}_rewards.txt', 'r') as f:
            rewards = json.load(f)
        
        with open(f'{model_name}_epsilon.txt', 'r') as f:
            epsilon_values = json.load(f)
            
        with open(f'{model_name}_opponent_rewards.txt', 'r') as f:
            opponent_rewards = json.load(f)
    except FileNotFoundError:
        print("Training data files not found.")
        return
    
    # Create figure and subplots
    fig, axs = plt.subplots(2, 1, figsize=(10, 12))
    
    # Plot rewards
    axs[0].plot(rewards, label='DQN Rewards')
    axs[0].plot(opponent_rewards, label='Opponent Rewards', alpha=0.7)
    axs[0].set_title('Rewards During Training')
    axs[0].set_xlabel('Episode')
    axs[0].set_ylabel('Reward')
    axs[0].legend()
    axs[0].grid(True, alpha=0.3)
    
    # Plot epsilon values
    axs[1].plot(epsilon_values)
    axs[1].set_title('Epsilon Values During Training')
    axs[1].set_xlabel('Episode')
    axs[1].set_ylabel('Epsilon')
    axs[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{model_name}_training_progress.png')
    plt.show()


def plot_evaluation_results(results, title):
    """Plot evaluation results"""
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot rewards distribution
    axs[0].hist(results['rewards'], bins=20, alpha=0.7, label='DQN')
    axs[0].hist(results['opponent_rewards'], bins=20, alpha=0.5, label='Opponent')
    axs[0].set_title('Reward Distribution')
    axs[0].set_xlabel('Reward')
    axs[0].set_ylabel('Frequency')
    axs[0].legend()
    
    # Plot win/draw/loss rates
    labels = ['Win', 'Draw', 'Loss']
    values = [results['win_rate'], results['draw_rate'], results['loss_rate']]
    colors = ['green', 'yellow', 'red']
    
    axs[1].bar(labels, values, color=colors)
    axs[1].set_ylim(0, 1.0)
    axs[1].set_title('Game Outcomes')
    axs[1].set_ylabel('Rate')
    
    for i, v in enumerate(values):
        axs[1].text(i, v + 0.02, f'{v:.2f}', ha='center')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_').lower()}.png")
    plt.show()


if __name__ == "__main__":
    model_name = 'century_dqn_v1'  # Replace with your model name
    
    # Load trained model
    agent, env = load_trained_model(model_name)
    
    # Show training progress
    plot_training_progress(model_name)
    
    # Evaluate against random agent
    results_random = evaluate_against_random(agent, env, num_episodes=100, render_mode='text')
    plot_evaluation_results(results_random, f"Evaluation Against Random Agent")
    
    # Evaluate against another model (if available)
    # opponent_model = 'another_model_name'
    # results_model = evaluate_against_model(agent, opponent_model, env, num_episodes=100)
    # plot_evaluation_results(results_model, f"Evaluation Against {opponent_model}")
    
    env.close() 