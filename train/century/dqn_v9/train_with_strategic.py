import sys
import os
import argparse

# Add parent directory to path to find modules
sys.path.append('..')

from dqn_v9 import train_dqn, DQNConfig
from strategic_agent import StrategicAgent

if __name__ == '__main__':
    # Allow for command line arguments
    parser = argparse.ArgumentParser(description='Train DQN agent with Strategic opponent for Century game')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of episodes to train')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint file to resume training')
    parser.add_argument('--reset', action='store_true', help='Flag to explicitly train from scratch (overwrites existing if no checkpoint)')
    parser.add_argument('--checkpoint-freq', type=int, default=50, help='Frequency to save checkpoints')
    parser.add_argument('--model-save-freq', type=int, default=50, help='Frequency to save model versions')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor for future rewards')
    parser.add_argument('--epsilon', type=float, default=1.0, help='Initial epsilon for exploration')
    parser.add_argument('--epsilon-decay', type=float, default=0.985, help='Decay rate for epsilon')
    parser.add_argument('--epsilon-min', type=float, default=0.05, help='Minimum epsilon value')
    parser.add_argument('--learning-rate', type=float, default=0.0007, help='Learning rate for optimizer')
    parser.add_argument('--update-rate', type=int, default=150, help='Update rate for target network')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size for training')
    parser.add_argument('--replay-buffer-size', type=int, default=100000, help='Size of the replay buffer')
    parser.add_argument('--num-timesteps', type=int, default=2000, help='Number of timesteps per episode')
    
    args = parser.parse_args()
    
    # Safety check: prevent accidental training from scratch
    if args.checkpoint is None and not args.reset:
        print("Error: To train from scratch, please use the --reset flag.")
        print("To resume from a checkpoint, use the --checkpoint <path> argument.")
        exit(1)
        
    config_dict = {
        'episodes': args.episodes, 
        'checkpoint': args.checkpoint,
        'checkpoint_freq': args.checkpoint_freq,
        'model_save_freq': args.model_save_freq,
        'gamma': args.gamma,
        'epsilon': args.epsilon,
        'epsilon_decay': args.epsilon_decay,
        'epsilon_min': args.epsilon_min,
        'learning_rate': args.learning_rate,
        'update_rate': args.update_rate,
        'batch_size': args.batch_size,
        'replay_buffer_size': args.replay_buffer_size,
        'num_timesteps': args.num_timesteps,
    }

    config = DQNConfig(**config_dict)
    
    # Modify the train_dqn function to use StrategicAgent
    # We do this by monkey-patching the original function
    original_train_dqn = train_dqn
    
    def train_with_strategic(config, num_episodes, checkpoint_path=None):
        """Modified version of train_dqn that uses StrategicAgent instead of RandomAgent"""
        import gymnasium as gym
        import numpy as np
        from gymnasium.wrappers import FlattenObservation
        import torch
        import time
        
        env = gym.make("gymnasium_env/CenturyGolem-v16")
        env = FlattenObservation(env)
        state, info = env.reset()

        # Define state and action size
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.n

        # Define number of timesteps per episode and batch size
        num_timesteps = config.num_timesteps
        batch_size = config.batch_size
        time_step = 0
        
        # Initialize agent and strategic opponent (instead of random)
        from dqn_v9 import DQNAgent
        dqn_agent = DQNAgent(state_size, action_size, config) 
        opponent = StrategicAgent(action_size)  # Use StrategicAgent
        
        # Initialize rewards and epsilon_values lists for tracking metrics
        dqn_agent.rewards = []
        dqn_agent.epsilon_values = []
        
        # Track merchant card hoarding
        dqn_agent.merchant_card_counts = []
        
        # Start episode counter
        start_episode = 0
        
        # Load checkpoint if provided
        if checkpoint_path:
            buffer_path = checkpoint_path.replace("checkpoint", "replay_buffer")
            start_episode = dqn_agent.load_checkpoint(checkpoint_path, buffer_path)
            print(f"Resuming training from episode {start_episode}")

        try:
            for ep in range(start_episode, num_episodes):
                dqn_total_reward = 0
                opponent_total_reward = 0
                
                state, info = env.reset()
                
                # Track merchant card acquisitions in this episode
                merchant_cards_acquired = 0

                print(f'\nTraining on EPISODE {ep+1} with epsilon {dqn_agent.epsilon:.4f}')
                start = time.time()

                for t in range(num_timesteps):
                    time_step += 1

                    # Update Target Network every update_rate timesteps
                    if time_step % dqn_agent.update_rate == 0:
                        dqn_agent.update_target_network()

                    if info['current_player'] == 0:
                        action = dqn_agent.pick_epsilon_greedy_action(state, info)
                        
                        # Track merchant card acquisitions (actions 1-43 are getM3-getM45)
                        if action >= 1 and action <= 43:
                            merchant_cards_acquired += 1
                            
                        next_state, reward, terminal, _, info = env.step(action)
                        
                        dqn_total_reward += reward
                        
                        if not terminal and info['current_player'] == 1:
                            # Use the strategic agent for opponent actions
                            opponent_action = opponent.pick_action(next_state, info)
                            next_state_after_opponent, opponent_reward, terminal, _, info = env.step(opponent_action)
                            
                            opponent_total_reward += opponent_reward
                            
                            dqn_agent.save_experience(state, action, reward, next_state_after_opponent, terminal)
                            
                            state = next_state_after_opponent
                        else:
                            dqn_agent.save_experience(state, action, reward, next_state, terminal)
                            state = next_state
                    
                    elif info['current_player'] == 1:
                        # Use the strategic agent for opponent actions
                        opponent_action = opponent.pick_action(state, info)
                        next_state, opponent_reward, terminal, _, info = env.step(opponent_action)
                        
                        opponent_total_reward += opponent_reward
                        state = next_state

                    if terminal:
                        print('Episode: ', ep+1, ',' ' terminated with Reward ', dqn_total_reward)
                        break

                    # Train the Main NN when ReplayBuffer has enough experiences
                    if len(dqn_agent.replay_buffer) > batch_size:
                        dqn_agent.train(batch_size)

                dqn_agent.rewards.append(dqn_total_reward)
                dqn_agent.epsilon_values.append(dqn_agent.epsilon)
                dqn_agent.merchant_card_counts.append(merchant_cards_acquired)
                
                # Update epsilon using the adaptive schedule
                dqn_agent.update_epsilon(ep)
                
                # Print merchant card acquisition stats periodically
                if ep % 10 == 0:
                    avg_cards = sum(dqn_agent.merchant_card_counts[-10:]) / min(10, len(dqn_agent.merchant_card_counts[-10:]))
                    print(f"Avg merchant cards acquired (last 10 ep): {avg_cards:.2f}")
                    
                    # Print training metrics
                    if hasattr(dqn_agent, 'training_loss') and len(dqn_agent.training_loss) > 0:
                        avg_loss = sum(dqn_agent.training_loss[-100:]) / min(100, len(dqn_agent.training_loss))
                        print(f"Avg training loss (last 100 updates): {avg_loss:.4f}")
                    
                    if hasattr(dqn_agent, 'avg_q_values') and len(dqn_agent.avg_q_values) > 0:
                        avg_q = sum(dqn_agent.avg_q_values[-100:]) / min(100, len(dqn_agent.avg_q_values))
                        print(f"Avg Q-value (last 100 actions): {avg_q:.4f}")

                # Update Epsilon value
                if ep % 25 == 0 and ep > 0:  # Check more frequently (every 25 episodes)
                    # Calculate win rate over last 25 episodes
                    recent_wins = sum(1 for i in range(max(0, ep-25), ep) if dqn_agent.rewards[i] > 0)
                    win_rate = recent_wins / min(25, ep)
                    
                    # If win rate is low, increase exploration
                    if win_rate < 0.3:  # Lower threshold
                        # Calculate a boost based on how low the win rate is
                        boost = 0.2 + (0.3 - win_rate) * 0.5  # More aggressive boost for very low win rates
                        dqn_agent.epsilon = min(0.7, dqn_agent.epsilon + boost)
                        print(f"Low win rate ({win_rate:.2f}) detected, increasing epsilon to {dqn_agent.epsilon:.4f}")

                # Print episode info
                elapsed = time.time() - start
                print(f'Time elapsed during EPISODE {ep+1}: {elapsed} seconds = {round(elapsed/60, 3)} minutes')
                
                # Update learning rate based on performance
                if ep > 0 and ep % 25 == 0:  # Check more frequently
                    # Track win status for each episode (1 for win, 0 for loss)
                    dqn_agent.recent_wins.append(1 if dqn_agent.rewards[ep-1] > 0 else 0)
                    
                    # Calculate moving win rate for more stable metric
                    if len(dqn_agent.recent_wins) > 0:
                        moving_win_rate = sum(dqn_agent.recent_wins) / len(dqn_agent.recent_wins)
                        
                        # Pass win rate to scheduler - higher win rates will delay learning rate reduction
                        dqn_agent.scheduler.step(moving_win_rate)
                        
                        current_lr = dqn_agent.optimizer.param_groups[0]['lr']
                        print(f"Current win rate: {moving_win_rate:.2f}, learning rate: {current_lr:.6f}")
                        
                        # If win rate drops significantly from peak, take action
                        if hasattr(dqn_agent, 'peak_win_rate'):
                            if dqn_agent.peak_win_rate > 0.7 and moving_win_rate < dqn_agent.peak_win_rate - 0.15:
                                # Lower threshold for responsiveness
                                boost = 0.1 + (dqn_agent.peak_win_rate - moving_win_rate) * 0.3
                                dqn_agent.epsilon = min(0.4, dqn_agent.epsilon + boost)
                                print(f"Win rate dropped from peak {dqn_agent.peak_win_rate:.2f} to {moving_win_rate:.2f}, increasing exploration to {dqn_agent.epsilon:.4f}")
                        
                        # Update peak win rate
                        if not hasattr(dqn_agent, 'peak_win_rate') or moving_win_rate > dqn_agent.peak_win_rate:
                            dqn_agent.peak_win_rate = moving_win_rate

                # Save checkpoint every checkpoint_freq episodes
                if (ep + 1) % config.checkpoint_freq == 0:
                    # Use a different directory for strategic checkpoints
                    checkpoint_dir = "checkpoints_strategic"
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    dqn_agent.save_checkpoint(ep + 1, checkpoint_dir=checkpoint_dir)
                
                # Save model every model_save_freq episodes
                if ((ep + 1) % config.model_save_freq == 0) or (ep == 0):
                    # Use a different directory for strategic models
                    models_dir = "models_strategic"
                    os.makedirs(models_dir, exist_ok=True)
                    dqn_agent.save_model(ep + 1, models_dir=models_dir)

        except KeyboardInterrupt:
            print("\nTraining interrupted manually.")
        
        finally:
            env.close()
    
    # Replace the original train_dqn with our modified version
    train_dqn = train_with_strategic
    
    # Start training with the strategic opponent
    train_dqn(config, config_dict['episodes'], config_dict['checkpoint']) 