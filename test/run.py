import gymnasium
import gymnasium_env
from gymnasium.wrappers import FlattenObservation
env = gymnasium.make('gymnasium_env/CenturyGolem-v9', render_mode='text')
# env = FlattenObservation(env)
import numpy as np

def random(n_timesteps=10):
    state, info = env.reset()
    
    for _ in range(n_timesteps):
        # print(info["valid_actions"])
        
        action = env.action_space.sample()
        state, reward, terminal, _, info = env.step(action)
        
        if terminal:
            break
    
    env.close()
    
def valid_random(n_timesteps=10):
    state, info = env.reset()
    
    for _ in range(n_timesteps):
        valid_mask = info["valid_actions"]
        valid_indices = np.where(valid_mask == 1)[0]  # Get indices of valid actions
        print(valid_indices)
        action = np.random.choice(valid_indices)  # Choose a valid action randomly
        state, reward, terminal, _, info = env.step(action)
        # print(reward)
        
        if terminal:
            break
    
    env.close()

def manual():
    state, info = env.reset()
    
    while 1:
        valid_mask = info["valid_actions"]
        valid_indices = np.where(valid_mask == 1)[0]  # Get indices of valid actions
        print(valid_indices)
        action = int(input("Enter action: "))
        state, reward, terminal, _, info = env.step(action)
        
        if terminal:
            break

def custom_actions(action_sequence):
    tot_reward = 0
    for a in action_sequence:
        state, reward, terminal, _, __ = env.step(a)
        # print(state)
        tot_reward += reward

        if terminal:
            break
    
    print(f"Reward: {tot_reward}")
    env.close()

def play_against_random(turns=5):
    state, info = env.reset()
    tot_reward = 0
    
    for _ in range(turns):
        # if state['current_player'] == 0:
        if info['current_player'] == 0:
            print(info['valid_actions'])
            action = int(input("Enter action: "))
            state, reward, terminal, _, info = env.step(action)
            tot_reward += reward
        # if state['current_player'] == 1:
        if info['current_player'] == 1:
            print(info['valid_actions'])
            valid_mask = info["valid_actions"]
            valid_indices = np.where(valid_mask == 1)[0]  # Get indices of valid actions
            action = np.random.choice(valid_indices)
            # action = env.action_space.sample()
            state, reward, terminal, _, info = env.step(action)
            if terminal:
                tot_reward += reward
        
        if terminal:
            break
    
    print(f"Reward: {tot_reward}")
    env.close()
    
def dqn_vs_random(turns=5):
    info = env.reset()
    tot_reward = 0
    
    for _ in range(turns):
        if info['current_player'] == 0:
            action = int(input("Enter action: "))
            state, reward, terminal, _, info = env.step(action)
            tot_reward += reward
        if info['current_player'] == 1:
            valid_mask = info["valid_actions"]
            valid_indices = np.where(valid_mask == 1)[0]  # Get indices of valid actions
            action = np.random.choice(valid_indices)
            # action = env.action_space.sample()
            state, reward, terminal, _, info = env.step(action)
            if terminal:
                tot_reward += reward
        
        if terminal:
            break
    
    print(f"Reward: {tot_reward}")
    env.close()
    

manual()