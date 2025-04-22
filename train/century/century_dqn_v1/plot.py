import pickle
import matplotlib.pyplot as plt

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