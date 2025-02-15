import gymnasium
import gymnasium_env
from gymnasium.wrappers import FlattenObservation
from gymnasium_env.envs.century_golem_v1 import Actions
import sys
sys.path.append("/Users/pongpradk/Documents/Codes/century/gymnasium_env/envs/century_golem_v2.py")
env = gymnasium.make('gymnasium_env/CenturyGolem-v2', render_mode='text')
env = FlattenObservation(env)

state, _ = env.reset()
done = False


def random(n_timesteps=10):
    for _ in range(n_timesteps):
        # Pick random action
        action = env.action_space.sample()
        print("Action:", Actions(int(action)).name)
        state, reward, terminal, _, __ = env.step(action)
        print("")
        
        if terminal:
            break
    
    env.close()
    
def best_strategy(action_sequence):
    tot_reward = 0
    for a in action_sequence:
        state, reward, terminal, _, __ = env.step(a)
        print(state)
        tot_reward += reward

        if terminal:
            break
    
    print(tot_reward)
        
    env.close()

# best_strategy([2,1,3,0,1,3])
random()