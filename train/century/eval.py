import gymnasium as gym
import gymnasium_env
import torch
import numpy as np
from typing import Tuple, Dict, Any
from dataclasses import dataclass
from century_dqn5.dqn5 import DQN
from gymnasium.wrappers import FlattenObservation
from random_agent import RandomAgent


@dataclass
class GameStats:
    """Class to track game statistics."""
    wins: int = 0
    losses: int = 0
    draws: int = 0
    
    @property
    def total_games(self) -> int:
        return self.wins + self.losses + self.draws
    
    @property
    def win_rate(self) -> float:
        return self.wins / self.total_games if self.total_games > 0 else 0.0
    
    def update(self, result: str) -> None:
        """Update statistics based on game result."""
        if result == "win":
            self.wins += 1
        elif result == "loss":
            self.losses += 1
        else:  # draw
            self.draws += 1
    
    def display(self) -> None:
        """Display current statistics."""
        print("\n=== Game Statistics ===")
        print(f"Total Games: {self.total_games}")
        print(f"Wins: {self.wins}")
        print(f"Losses: {self.losses}")
        print(f"Draws: {self.draws}")
        print(f"Win Rate: {self.win_rate:.2%}")
        print("=====================\n")


def load_pretrained_model(model_path: str) -> DQN:
    """
    Load a pretrained DQN model from the specified path.
    
    Args:
        model_path: Path to the pretrained model file
        
    Returns:
        Loaded DQN model in evaluation mode
    """
    # Initialize environment to get state and action dimensions
    env = gym.make('gymnasium_env/CenturyGolem-v11')
    env = FlattenObservation(env)
    state, _ = env.reset()
    state_size = len(state)
    action_size = env.action_space.n
    env.close()
    
    # Create and load the model
    model = DQN(state_size, action_size)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def select_trained_agent_action(
    state: np.ndarray,
    trained_model: DQN,
    info: Dict[str, Any]
) -> int:
    """
    Select the best action using the trained model among valid actions.
    
    Args:
        state: Current state of the environment
        trained_model: Trained DQN model
        info: Environment info containing valid actions
        
    Returns:
        Selected action index
    """
    with torch.no_grad():
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = trained_model(state_tensor)[0].numpy()
        valid_actions = info['valid_actions']
        masked_q_values = np.where(valid_actions == 1, q_values, -np.inf)
        return np.argmax(masked_q_values)


def play_turn(
    env: gym.Env,
    state: np.ndarray,
    info: Dict[str, Any],
    agent: Any,
    is_trained_agent: bool = True
) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
    """
    Execute a single turn in the game.
    
    Args:
        env: The game environment
        state: Current state
        info: Environment info
        agent: The agent to play (trained model or random agent)
        is_trained_agent: Whether the agent is the trained DQN model
        
    Returns:
        Tuple of (next_state, reward, terminal, info)
    """
    if is_trained_agent:
        action = select_trained_agent_action(state, agent, info)
    else:
        action = agent.pick_action(state, info)
        
    return env.step(action)


def determine_game_result(info: Dict[str, Any], total_reward: float) -> str:
    """
    Determine the result of the game.
    
    Args:
        info: Final environment info
        total_reward: Total reward achieved by the trained agent
        
    Returns:
        String indicating the result: "win", "loss", or "draw"
    """
    if info.get('winner') is not None:
        if info['winner'] == "P1":
            return "win"
        elif info['winner'] == "P2":
            return "loss"
        else:  # P0 means tie
            return "draw"
    return "draw"


def run_episode(
    env: gym.Env,
    trained_agent: DQN,
    opponent: RandomAgent,
    max_timesteps: int = 2000
) -> Tuple[float, str]:
    """
    Run a single episode of the game.
    
    Args:
        env: The game environment
        trained_agent: The trained DQN model
        opponent: The random opponent agent
        max_timesteps: Maximum number of timesteps per episode
        
    Returns:
        Tuple of (total_reward, game_result)
    """
    state, info = env.reset()
    total_reward = 0
    
    for _ in range(max_timesteps):
        # Trained agent's turn
        if info['current_player'] == 0:
            next_state, reward, terminal, _, info = play_turn(
                env, state, info, trained_agent, is_trained_agent=True
            )
            total_reward += reward
            
            if not terminal and info['current_player'] == 1:
                # Opponent's turn
                next_state, _, terminal, _, info = play_turn(
                    env, next_state, info, opponent, is_trained_agent=False
                )
                state = next_state
            else:
                state = next_state
                
        # Opponent's turn
        elif info['current_player'] == 1:
            next_state, _, terminal, _, info = play_turn(
                env, state, info, opponent, is_trained_agent=False
            )
            state = next_state
            
        if terminal:
            break
            
    game_result = determine_game_result(info, total_reward)
    return total_reward, game_result


def run_multiple_episodes(
    env: gym.Env,
    trained_agent: DQN,
    opponent: RandomAgent,
    num_episodes: int = 1000
) -> GameStats:
    """
    Run multiple episodes and track statistics.
    
    Args:
        env: The game environment
        trained_agent: The trained DQN model
        opponent: The random opponent agent
        num_episodes: Number of episodes to run
        
    Returns:
        GameStats object containing the statistics
    """
    stats = GameStats()
    
    for episode in range(num_episodes):
        total_reward, result = run_episode(env, trained_agent, opponent)
        stats.update(result)
    
    return stats


def main(num_episodes: int = 1000):
    """
    Main function to run the evaluation.
    
    Args:
        num_episodes: Number of episodes to run
    """
    # Create environment
    env = gym.make('gymnasium_env/CenturyGolem-v11', render_mode=None)
    env = FlattenObservation(env)
    
    # Load the trained model and create opponent
    trained_agent = load_pretrained_model('century_dqn5/models/trained_model_1500.pt')
    opponent = RandomAgent(env.action_space.n)
    
    # Run multiple episodes and get statistics
    stats = run_multiple_episodes(env, trained_agent, opponent, num_episodes)
    
    # Display final statistics
    stats.display()


if __name__ == '__main__':
    main(1000)  # Run 1000 episodes by default