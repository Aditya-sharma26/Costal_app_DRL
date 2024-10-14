import matplotlib.pyplot as plt
import numpy as np
from env_2_regions_4_dikes import Environment
from values_slr import slr
from values_surge import surge
# Assuming the environment and necessary dependencies are already defined and imported.
# We will simulate 10^4 realizations using the environment.

# Assuming your environment and necessary functions (slr, surge) are already defined and imported
env = Environment(env_name="coastal_city", climate_model="245")

# Number of realizations and years
n_realizations = 10 ** 3
years = 40  # As defined in your environment

# Initialize array to store total heights for all realizations over time
total_heights = np.zeros((n_realizations, years))

# Run the simulations
for i in range(n_realizations):
    state = env.reset()
    for year in range(years):
        # Randomly choose a valid action
        valid_actions = env.get_valid_actions(env.system)
        action = np.random.choice(valid_actions)

        # Take a step in the environment
        next_state, _, done, _ = env.step(action)

        # Calculate the total height (SLR + storm surge) for this year
        slr_value = slr(next_state[0]) / 100  # Convert SLR state to meters
        surge_value = surge(next_state[1]) / 100  # Convert surge state to meters
        total_heights[i, year] = slr_value + surge_value

        if done:
            break

# Plot all individual realizations
plt.figure(figsize=(12, 8))

# Plot each realization with low opacity for visual clarity
for i in range(n_realizations):
    plt.plot(total_heights[i], color='gray', alpha=0.01)

# Overlay mean total height
mean_heights = np.mean(total_heights, axis=0)
plt.plot(mean_heights, color='blue', label='Mean Total Height', linewidth=2)

# Overlay 5th to 95th percentile range
plt.fill_between(range(years), np.percentile(total_heights, 2, axis=0), np.percentile(total_heights, 98, axis=0),
                 color='blue', alpha=0.3, label="2th-98th Percentile")

# Add plot details
plt.title('Total Height (SLR + Storm Surge)',fontsize='20')
plt.xlabel('Years', fontsize='20')
plt.ylabel('Total Height (meters)', fontsize='20')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

plt.legend(fontsize=16)
plt.grid(True)

plt.show()