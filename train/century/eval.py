import gymnasium as gym
import torch
import numpy as np
from typing import Tuple, Dict, Any, List
from dataclasses import dataclass
from dqn_v6.dqn_v6 import DQN
from gymnasium.wrappers import FlattenObservation
from random_agent import RandomAgent
from gymnasium_env.envs.century_v12.enums import Actions


@dataclass
class GameStats:
    """Class to track game statistics."""
    wins: int = 0
    losses: int = 0
    draws: int = 0
    action_counts: np.ndarray = None
    action_sequences: List[List[str]] = None
    track_actions: bool = False
    track_wins: bool = False
    track_sequences: bool = False
    
    def __post_init__(self):
        if self.track_actions:
            self.action_counts = np.zeros(len(Actions), dtype=int)
        if self.track_sequences:
            self.action_sequences = []
    
    @property
    def total_games(self) -> int:
        return self.wins + self.losses + self.draws if self.track_wins else 0
    
    @property
    def win_rate(self) -> float:
        return self.wins / self.total_games if self.total_games > 0 else 0.0
    
    def update(self, result: str) -> None:
        """Update statistics based on game result."""
        if self.track_wins:
            if result == "win":
                self.wins += 1
            elif result == "loss":
                self.losses += 1
            else:  # draw
                self.draws += 1
    
    def update_action_count(self, action: int) -> None:
        """Update action count if tracking is enabled."""
        if self.track_actions:
            self.action_counts[action] += 1
    
    def start_new_sequence(self) -> None:
        """Start a new action sequence if tracking is enabled."""
        if self.track_sequences:
            self.action_sequences.append([])
    
    def add_to_sequence(self, action: int) -> None:
        """Add an action to the current sequence if tracking is enabled."""
        if self.track_sequences and self.action_sequences:
            self.action_sequences[-1].append(Actions(action).name)
    
    def display(self, num_episodes: int) -> None:
        """Display current statistics."""
        if self.track_wins:
            print("\n=== Game Statistics ===")
            print(f"Total Games: {self.total_games}")
            print(f"Wins: {self.wins}")
            print(f"Losses: {self.losses}")
            print(f"Draws: {self.draws}")
            print(f"Win Rate: {self.win_rate:.2%}")
            print("=====================\n")
        
        if self.track_actions:
            self.display_action_stats(num_episodes)
            
        if self.track_sequences:
            self.display_action_sequences()
    
    def display_action_stats(self, num_episodes: int) -> None:
        """Display action distribution statistics."""
        if not self.track_actions or num_episodes == 0:
            return
            
        print("\n=== Action Distribution ===")
        total_actions = np.sum(self.action_counts)
        
        # Calculate averages and percentages
        action_averages = self.action_counts / num_episodes
        action_percentages = self.action_counts / total_actions * 100 if total_actions > 0 else np.zeros_like(self.action_counts)
        
        # Sort actions by frequency (descending)
        sorted_indices = np.argsort(-self.action_counts)
        
        for idx in sorted_indices:
            action_name = Actions(idx).name
            avg_count = action_averages[idx]
            percentage = action_percentages[idx]
            print(f"{action_name}: {avg_count:.2f} per game ({percentage:.1f}%)")
        
        print("==========================\n")
    
    def display_action_sequences(self) -> None:
        """Display action sequences for all games."""
        if not self.track_sequences or not self.action_sequences:
            return
            
        print("\n=== Action Sequences ===")
        for game_idx, sequence in enumerate(self.action_sequences, 1):
            print(f"\nGame {game_idx}:")
            # Print actions in groups of 5 for better readability
            for i in range(0, len(sequence), 5):
                print("  " + ", ".join(sequence[i:i+5]))
        print("\n=======================\n")


def load_pretrained_model(model_path: str) -> DQN:
    """
    Load a pretrained DQN model from the specified path.
    
    Args:
        model_path: Path to the pretrained model file
        
    Returns:
        Loaded DQN model in evaluation mode
    """
    # Initialize environment to get state and action dimensions
    env = gym.make('gymnasium_env/CenturyGolem-v12')
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
    is_trained_agent: bool = True,
    stats: GameStats = None
) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
    """
    Execute a single turn in the game.
    
    Args:
        env: The game environment
        state: Current state
        info: Environment info
        agent: The agent to play (trained model or random agent)
        is_trained_agent: Whether the agent is the trained DQN model
        stats: GameStats object for tracking action counts
        
    Returns:
        Tuple of (next_state, reward, terminal, info)
    """
    if is_trained_agent:
        action = select_trained_agent_action(state, agent, info)
        if stats is not None:
            stats.update_action_count(action)
            stats.add_to_sequence(action)
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
    max_timesteps: int = 2000,
    stats: GameStats = None
) -> Tuple[float, str]:
    """
    Run a single episode of the game.
    
    Args:
        env: The game environment
        trained_agent: The trained DQN model
        opponent: The random opponent agent
        max_timesteps: Maximum number of timesteps per episode
        stats: GameStats object for tracking action counts
        
    Returns:
        Tuple of (total_reward, game_result)
    """
    state, info = env.reset()
    total_reward = 0
    
    if stats is not None:
        stats.start_new_sequence()
    
    for _ in range(max_timesteps):
        # Trained agent's turn
        if info['current_player'] == 0:
            next_state, reward, terminal, _, info = play_turn(
                env, state, info, trained_agent, is_trained_agent=True, stats=stats
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
    num_episodes: int = 1000,
    track_actions: bool = False,
    track_wins: bool = False,
    track_sequences: bool = False
) -> GameStats:
    """
    Run multiple episodes and track statistics.
    
    Args:
        env: The game environment
        trained_agent: The trained DQN model
        opponent: The random opponent agent
        num_episodes: Number of episodes to run
        track_actions: Whether to track action distribution
        track_wins: Whether to track win/loss statistics
        track_sequences: Whether to track action sequences
        
    Returns:
        GameStats object containing the statistics
    """
    stats = GameStats(
        track_actions=track_actions,
        track_wins=track_wins,
        track_sequences=track_sequences
    )
    
    for episode in range(num_episodes):
        total_reward, result = run_episode(env, trained_agent, opponent, stats=stats)
        stats.update(result)
    
    return stats


def main(num_episodes: int = 1000, track_actions: bool = False, track_wins: bool = False, track_sequences: bool = False):
    """
    Main function to run the evaluation.
    
    Args:
        num_episodes: Number of episodes to run
        track_actions: Whether to track action distribution
        track_wins: Whether to track win/loss statistics
        track_sequences: Whether to track action sequences
    """
    # Create environment
    env = gym.make('gymnasium_env/CenturyGolem-v12', render_mode=None)
    env = FlattenObservation(env)
    
    # Load the trained model and create opponent
    trained_agent = load_pretrained_model('dqn_v6/models/trained_model_6900.pt')
    opponent = RandomAgent(env.action_space.n)
    
    # Run multiple episodes and get statistics
    stats = run_multiple_episodes(
        env, trained_agent, opponent, num_episodes,
        track_actions, track_wins, track_sequences
    )
    
    # Display final statistics
    stats.display(num_episodes)


if __name__ == '__main__':
    main(1000, track_wins=True, track_actions=True, track_sequences=False)