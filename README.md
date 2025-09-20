## Investigating Deep Reinforcement Learning for Century: A Modern Engine-Building Board Game

This repository contains the code and experiments for my undergraduate dissertation, where I investigate how **Double Deep Q-Network (DDQN)** agents learn to play the strategy board game *Century: Golem Edition*.

---

### Project Overview
- Built a **custom Gymnasium environment** simulating the *Century: Golem Edition* board game, supporting multiplayer gameplay.  
- Developed and trained **DDQN agents** under three training schemes:
  1. Versus random opponents
  2. Versus heuristic opponents
  3. Self-play (agent vs. itself)  
- Achieved a **58% win rate against human players**, demonstrating DDQN’s ability to adapt and perform in strategic decision-making scenarios.  
- Analysed trade-offs of different training approaches and enhancement techniques.  

---

### Tech Stack
- **Python 3.11**  
- [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) (custom environment integration)  
- **PyTorch** (DDQN implementation)  
- **NumPy, Pandas** (data handling)  
- **Matplotlib** (visualisation)  

---

### Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/yourusername/century-ddqn.git
cd century-ddqn
