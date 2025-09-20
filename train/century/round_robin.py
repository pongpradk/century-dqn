import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

# === Manually specify win rate data here (example values) ===
# Format: win_rates[player1][player2] 
# For example: win_rates[0][1] is win rate of DQNRandom vs DQNPhase
win_rates = np.array([
    [0.45, 0.84, 0.75],
    [0.10, 0.46, 0.20],
    [0.23, 0.70, 0.52]
])

# === Labels for the agents ===
agents = ["DQNRandom", "DQNPhase", "DQNSelfplay"]

# === Create the directory if it doesn't exist ===
output_dir = "round_robin"
os.makedirs(output_dir, exist_ok=True)

# === Create the heatmap ===
plt.figure(figsize=(7, 5))

# Colormap similar to example (yellow -> red -> blue). "jet" is very close.
sns.heatmap(win_rates, annot=True, fmt=".2f", cmap="jet", 
            xticklabels=agents, yticklabels=agents, 
            vmin=0.3, vmax=0.7, cbar=True,
            annot_kws={'size': 18}, cbar_kws={'shrink': 0.8})

# Axis labels and title
plt.ylabel("Player 1", fontsize=18)
plt.xlabel("Player 2", fontsize=18)
# plt.title("DQN Round Robin Win Rates", fontsize=14)

# Rotate x-axis labels to avoid overlap
plt.xticks(rotation=20, ha='center', fontsize=18)
plt.yticks(rotation=0, fontsize=18)

# Adjust colorbar tick labels font size
cbar = plt.gcf().axes[-1]
cbar.tick_params(labelsize=16)

# Save the figure
output_path = os.path.join(output_dir, "dqn_vs_dqn_rr.png")
plt.tight_layout()
plt.savefig(output_path, dpi=300)
plt.close()

print(f"Heatmap saved to {output_path}")