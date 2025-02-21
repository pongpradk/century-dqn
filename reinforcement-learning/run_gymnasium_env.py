import gymnasium
import gymnasium_env
from gymnasium.wrappers import FlattenObservation
from gymnasium_env.envs.century_v4 import Actions
env = gymnasium.make('gymnasium_env/CenturyGolem-v4', render_mode='text')
# env = FlattenObservation(env)

state, _ = env.reset()
done = False


def random(n_timesteps=10):
    for _ in range(n_timesteps):
        # Pick random action
        action = env.action_space.sample()
        print("ACTION -", Actions(int(action)).name)
        state, reward, terminal, _, __ = env.step(action)
        
        if terminal:
            break
    
    env.close()
    
def best_strategy(action_sequence):
    tot_reward = 0
    for a in action_sequence:
        print(f"==== {Actions(int(a)).name} ====")
        state, reward, terminal, _, __ = env.step(a)
        # print(state)
        tot_reward += reward

        if terminal:
            break
    
    print(f"Reward: {tot_reward}")
    env.close()

# best_strategy([1,2,3,4,5,6,7,8,9,10,11])
# best_strategy([2, 4, 6, 1, 3, 5, 7, 0, 1, 3, 5, 7, 0])
best_strategy([5,6,11,12])
# random()