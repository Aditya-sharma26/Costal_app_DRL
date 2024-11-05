import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
import gym
import torch as T
import agent_per as Agents
from utils2 import make_env
from components import components
from values_slr import slr
from values_surge import surge

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate DQN Policy')

    # Environment parameters
    parser.add_argument('-b1', type=float, default=2.5, help='Base height for Region 1 lower dike')
    parser.add_argument('-b2', type=float, default=5, help='Base height for Region 1 higher dike')
    parser.add_argument('-r_1_h0', type=float, default=0.0, help='Base elevation for Region 1')

    # Evaluation parameters
    parser.add_argument('-n_episodes', type=int, default=10, help='Number of evaluation episodes')
    parser.add_argument('-path', type=str, default='models/', help='Path to the saved models')
    parser.add_argument('-algo', type=str, default='DDQNAgent', help='Algorithm used for training')
    parser.add_argument('-climate_ssp', type=str, default='245', help='Climate model scenario')
    parser.add_argument('-gamma', type=float, default=0.90,
                        help='Discount factor for update equation.')
    parser.add_argument('-env', type=str, default='environment_2_regions_4_dikes_elevated', help='Name of the file defining the environment')

    args = parser.parse_args()

    # Set up environment
    env_variant = f"Env_b1_{args.b1}_b2_{args.b2}_r1h0_{args.r_1_h0}"
    results_folder = f"results/{env_variant}"
    os.makedirs(results_folder, exist_ok=True)

    env = make_env(env_name=args.env, climate_model=args.climate_ssp,
                   b1=args.b1, b2=args.b2, r_1_h0=args.r_1_h0)

    # Initialize agent
    agent_ = getattr(Agents, args.algo)
    agent = agent_(args.gamma,  # Use same gamma as in training
                   epsilon=0.0,  # No exploration during evaluation
                   lr=0.0001,  # Learning rate (not used during evaluation)
                   input_dims=env.observation_space.n+1+4,
                   n_actions=len(env.action_space),
                   mem_size=500000,
                   eps_min=0.0,
                   batch_size=128,
                   replace=20,
                   eps_dec=1e-5,
                   chkpt_dir=args.path,
                   algo=args.algo,
                   env_name=env_variant)


    def plot_policy(env, years, slr_values, surge_values, total_heights, actions_taken, episode, results_folder):
        import matplotlib.pyplot as plt
        import os
        from matplotlib.lines import Line2D

        # Define y-axis limits
        water_height_ymin = 0
        water_height_ymax = 5  # Adjust based on maximum expected water height
        action_ymin = -0.5
        action_ymax = 3.5  # Number of dikes (4) minus 0.5 for padding

        fig, ax1 = plt.subplots(figsize=(12, 6))

        # Plot SLR, Storm Surge, and Total Water Height on the left y-axis
        ax1.set_xlabel('Year')
        ax1.set_ylabel('Water Height (m)', color='tab:blue')
        ax1.plot(years, slr_values, label='Sea Level Rise', color='tab:blue', linestyle='--')
        ax1.plot(years, surge_values, label='Storm Surge', color='tab:cyan', linestyle='-.')
        ax1.plot(years, total_heights, label='Total Water Height', color='tab:blue')
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        ax1.set_ylim(water_height_ymin, water_height_ymax)  # Set consistent y-axis limits

        # Create a twin axis for plotting actions
        ax2 = ax1.twinx()
        ax2.set_ylabel('Dikes', color='tab:red')
        ax2.set_ylim(action_ymin, action_ymax)  # Set consistent y-axis limits

        # Define dike labels and colors in the desired order
        dike_labels = ['D_lower-R1', 'D_lower-R2', 'D_higher-R1', 'D_higher-R2']
        dike_colors = ['tab:red', 'tab:orange', 'tab:green', 'tab:purple']
        dike_indices = [0, 1, 2, 3]  # Indices for y-axis positions

        # Action order mapping: map action tuple indices to plotting indices
        # Original action tuple order: [D_lower-R1, D_higher-R1, D_lower-R2, D_higher-R2]
        # Desired plotting order: [D_lower-R1, D_lower-R2, D_higher-R1, D_higher-R2]
        action_order = [0, 2, 1, 3]  # Mapping from action tuple indices to plotting indices

        # Initialize action data for plotting
        action_data = {idx: {'years': [], 'actions': [], 'markers': [], 'colors': []} for idx in dike_indices}

        # Process actions for each time step
        for i, action_idx in enumerate(actions_taken):
            action_tuple = env.action_space[action_idx]
            # For each dike, check if an action was taken
            for action_idx_in_tuple, dike_plot_idx in enumerate(action_order):
                dike_action = action_tuple[action_idx_in_tuple]
                if dike_action != 0:
                    # Determine the marker based on the action
                    if dike_action == 0.75:
                        marker_style = 'x'
                    elif dike_action == 'elevate':
                        marker_style = '^'
                    elif dike_action == 1.5:
                        marker_style = 's'  # Square marker
                    else:
                        marker_style = 'o'  # Default marker for any other action
                    # Use the corresponding line color
                    marker_color = dike_colors[dike_plot_idx]
                    # Append data for plotting
                    action_data[dike_plot_idx]['years'].append(years[i])
                    action_data[dike_plot_idx]['actions'].append(dike_plot_idx)
                    action_data[dike_plot_idx]['markers'].append(marker_style)
                    action_data[dike_plot_idx]['colors'].append(marker_color)

        # Plot actions as markers with vertical lines
        for dike_idx in dike_indices:
            # Plot invisible scatter to set the color
            ax2.scatter([], [], marker='o', color=dike_colors[dike_idx], s=0)
            for x, y, m, c in zip(action_data[dike_idx]['years'], action_data[dike_idx]['actions'],
                                  action_data[dike_idx]['markers'], action_data[dike_idx]['colors']):
                # Plot vertical line
                ax2.vlines(x, action_ymin, y, colors=c, linestyles='dotted', alpha=0.7)
                # Plot marker
                ax2.plot(x, y, marker=m, color=c, linestyle='None', markersize=10)

        ax2.set_yticks(dike_indices)
        ax2.set_yticklabels(dike_labels)
        ax2.tick_params(axis='y', labelcolor='tab:red')

        # Create custom legend entries for the action markers
        action_markers = [
            Line2D([0], [0], marker='x', color='w', label='Construct 0.75m',
                   markerfacecolor='k', markeredgecolor='k', markersize=10),
            Line2D([0], [0], marker='s', color='w', label='Construct 1.5m',
                   markerfacecolor='k', markeredgecolor='k', markersize=10),
            Line2D([0], [0], marker='^', color='w', label='Elevate to 1.5m',
                   markerfacecolor='k', markeredgecolor='k', markersize=10)
        ]

        # Legends and titles
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        # Combine all legend entries
        all_lines = lines_1 + action_markers
        all_labels = labels_1 + [marker.get_label() for marker in action_markers]
        ax1.legend(all_lines, all_labels, loc='upper right')

        plt.title(f'Policy Evaluation - Episode {episode}')
        plt.tight_layout()
        # Save the plot
        plot_filename = os.path.join(results_folder, f'policy_plot_episode_{episode}.png')
        plt.savefig(plot_filename)
        plt.close()
    # Evaluation code
agent.load_models()
agent.epsilon = 0.0  # No random exploration

test_episodes = 100
scores = []

for episode in range(test_episodes):
    observation = env.reset()
    done = False
    rewards = 0
    steps = 0
    year = 0

    # Lists to store data for plotting
    years = []
    slr_values = []
    surge_values = []
    total_heights = []
    actions_taken = []  # To store the action indices

    while not done:
        steps += 1
        slr_value = slr(observation[0])
        surge_value = surge(observation[1])
        total_height = slr_value + surge_value

        print(f'year: {year}, combined state: {observation}, slr: {slr_value} cm, surge: {surge_value} cm')
        state_vector = env.get_state_vector(observation, year)
        action = agent.choose_action(env, state_vector)

        new_observation, reward, done, info = env.step(action)
        rewards += reward

        # Store data for plotting
        years.append(year)
        slr_values.append(slr_value / 100)       # Convert cm to meters
        surge_values.append(surge_value / 100)   # Convert cm to meters
        total_heights.append(total_height / 100) # Convert cm to meters
        actions_taken.append(action)             # Store the action index

        observation = new_observation
        year += 1  # Increment year after observation update

        if done:
            print('Episode {} ends with total rewards {}'.format(episode, rewards))
            # Plot the policy for this episode
            plot_policy(env, years, slr_values, surge_values, total_heights, actions_taken, episode, results_folder)
            break

    scores.append(rewards)

mean_score = np.mean(scores)
std_score = np.std(scores)
print(f'Average reward over {test_episodes} episodes: {mean_score} ± {std_score}')