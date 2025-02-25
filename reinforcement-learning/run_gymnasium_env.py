import gymnasium
import gymnasium_env
from gymnasium.wrappers import FlattenObservation
# from gymnasium_env.envs.century_v4 import Actions
# print(f"==== {Actions(int(a)).name} ====")
env = gymnasium.make('gymnasium_env/CenturyGolem-v5', render_mode='text')
# env = FlattenObservation(env)

# state, _ = env.reset()
# done = False


def random(n_timesteps=10):
    for _ in range(n_timesteps):
        # Pick random action
        action = env.action_space.sample()
        state, reward, terminal, _, __ = env.step(action)
        
        if terminal:
            break
    
    env.close()

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

def input_action():
    env.reset()
    tot_reward = 0
    try:
        while 1:
            action = int(input("Enter action: "))
            state, reward, terminal, _, __ = env.step(action)
            tot_reward += reward
            
            if terminal:
                break
    except KeyboardInterrupt:
        pass
    
    print(f"Reward: {tot_reward}")
    env.close()

def turn_based(turns=10):
    for _ in range(turns):
        state, reward, terminal, _, __ = env.step(action)
        
        if terminal:
            break

def play_against_random(turns=5):
    state, _ = env.reset()
    tot_reward = 0
    
    for _ in range(turns):
        if state['current_player'] == 0:
            action = int(input("Enter action: "))
            state, reward, terminal, _, __ = env.step(action)
            tot_reward += reward
        if state['current_player'] == 1:
            action = env.action_space.sample()
            state, reward, terminal, _, __ = env.step(action)
            if terminal:
                tot_reward += reward
        
        if terminal:
            break
    
    print(f"Reward: {tot_reward}")
    env.close()
    

# custom_actions([1,2,3,4,5,6,7,8,9,10,12]) # get golem
# custom_actions([1,2,3,4,5]) # get all merchant cards
# custom_actions([5,0,11,0,12]) # get one golem
# input_action()
# random(4)
play_against_random(50)