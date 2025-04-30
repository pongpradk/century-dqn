import pickle

# Load timestep data from a saved file
with open("models/timestep_data_100.pkl", "rb") as f:
    timestep_data = pickle.load(f)
    
print(f"Total timesteps: {timestep_data['total_timesteps']}")
print(f"Cumulative timesteps per episode: {timestep_data['cumulative_timesteps']}")

# Or use the mapping function to get episode-to-timestep dictionary
episode_to_timestep = {i: timestep_data['cumulative_timesteps'][i] 
                      for i in range(len(timestep_data['cumulative_timesteps']))}