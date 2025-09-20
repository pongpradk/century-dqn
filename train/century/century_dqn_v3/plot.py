import pickle
import matplotlib.pyplot as plt

def rewards():
    # Load saved rewards
    with open("rewards_log.pkl", "rb") as f:
        rewards = pickle.load(f)

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(rewards, label="DQN Total Reward per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("DQN Training Reward per Episode")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("dqn_training_rewards.png")  # Optional
    plt.show()

def result():
    # Load episode results
    with open("results_log.pkl", "rb") as f:
        results = pickle.load(f)

    # Plotting the outcome per episode
    plt.figure(figsize=(14, 5))
    plt.plot(results, label="Game Outcome", marker='o', markersize=2, linestyle='None', alpha=0.6)
    plt.axhline(0, color='gray', linestyle='--')
    plt.yticks([-1, 0, 1], ['Loss', 'Tie', 'Win'])
    plt.xlabel("Episode")
    plt.ylabel("Result")
    plt.title("Game Result per Episode")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("dqn_game_outcomes.png")  # Optional: Save the plot
    plt.show()

result()