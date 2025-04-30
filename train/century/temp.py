import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches

# Example action names
actions = ['rest', 'useM1', 'useM2', 'useM7', 'useM8', 'getG1', 'getG2', 'getG3', 'getM3', 'getM4', 'getM10']

# Made-up average action counts (you can replace this with your real data)
avg_counts = [5, 30, 25, 50, 20, 8, 12, 3, 15, 18, 10]

# Assign colors based on action type
colors = []
for action in actions:
    if action.startswith('useM'):
        colors.append('blue')     # Use Merchant Card
    elif action.startswith('getG'):
        colors.append('green')    # Claim Golem Card
    elif action.startswith('getM'):
        colors.append('orange')   # Acquire Merchant Card
    else:
        colors.append('grey')     # Rest

# Create bar plot
plt.figure(figsize=(12,6))
bars = plt.bar(actions, avg_counts, color=colors)

# Labels and title
plt.xlabel('Action')
plt.ylabel('Average Count per Game')
plt.title('Action Distribution of DQNv4 Agent After 800 Episodes')

# Create custom legend
legend_patches = [
    mpatches.Patch(color='blue', label='Use Merchant Card'),
    mpatches.Patch(color='green', label='Claim Golem Card'),
    mpatches.Patch(color='orange', label='Acquire Merchant Card'),
    mpatches.Patch(color='grey', label='Rest')
]
plt.legend(handles=legend_patches)

# Show plot
plt.tight_layout()
plt.show()