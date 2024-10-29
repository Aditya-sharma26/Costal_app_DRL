from gym import Env
from gym.spaces import Box, Discrete
import random
import numpy as np
import mat73
from values_slr import slr
from values_surge import surge

class Environment(Env):
    def __init__(self, env_name, climate_model, b1, b2, r_1_h0):

        # Initialize environment name and climate model
        self.env_name = env_name
        self.climate_model = climate_model
        self.b1 = b1
        self.b2 = b2
        self.r_1_h0 = r_1_h0
        
        # Define initial year and horizon
        self.year = 0
        self.horizon = 39
        self.discount_reward = 0.97

        # Define the state space for SLR and storm surge
        self.n_states_slr = 77
        self.n_states_surge = 72
        self.n_states_total = self.n_states_slr + self.n_states_surge

        # Define the action space and state space for the environment
        self.action_space = self.define_action_space()
        self.state_space = self.define_state_space()
        self.observation_space = Discrete(self.n_states_total) # slr + surge

        # Initialize dynamic action space and transition mapping
        self.dynamic_action_space = {}
        self.transition_mapping = {}
        self.generate_dynamic_actions_and_transitions()

        # Define the initial state of the environment (SLR, storm_surge)
        self.initial_state = np.array([4, 1])  # Example initial state
        self.state = self.initial_state

        # Define the initial system configuration (D1, D2, D3, D4)
        self.initial_system = (0, 0, 0, 0)  # All dikes at 0m initially
        self.system = self.initial_system
        self.previous_system = self.initial_system

        # Get transition matrices for SLR and storm surge
        self.trans = self.get_transition(ssp=self.climate_model)
        self.trans_slr = self.trans[0]
        self.trans_surge = self.trans[1]

        #self.terminal_rewards = self.get_terminal_reward(env_model=self.model)

    def define_action_space(self):
        """
        Define all possible unique actions in the environment.
        Each action is a tuple of four elements representing the action at D1, D2, D3, and D4.
        Possible actions at each dike are:
        - D1 and D2:
          - 0: Do nothing
          - 0.75: Construct to 0.75 meters
          - 1.5: Construct directly to 1.5 meters
          - 1.5 (elevate): Elevate from 0.75 to 1.5 meters
        - D3 and D4:
          - 0: Do nothing
          - 0.75: Construct to 0.75 meters
        """
        actions = []

        # Define possible actions at each dike location
        d1_d2_actions = [0, 0.75, 1.5, 'elevate']  # Possible actions at D1 and D2
        d3_d4_actions = [0, 0.75]  # Possible actions at D3 and D4 (Region 2)

        # Generate all combinations of actions for D1, D2, D3, and D4
        for d1_action in d1_d2_actions:
            for d2_action in d1_d2_actions:
                for d3_action in d3_d4_actions:
                    for d4_action in d3_d4_actions:
                        actions.append((d1_action, d2_action, d3_action, d4_action))

        return actions

    def get_action_space(self):
        """
        Return the full action space.
        """
        return self.action_space

    def define_state_space(self):
        """
        Define all possible system configurations (states) in the environment.
        Each state is a tuple of four elements representing the height at D1, D2, D3, and D4.
        Possible states at D1 and D2 are 0 m, 0.75 m, 1.5 m.
        Possible states at D3 and D4 are 0 m, 0.75 m.
        """
        states = []

        d1_d2_states = [0, 0.75, 1.5]  # Possible states at D1 and D2
        d3_d4_states = [0, 0.75]  # Possible states at D3 and D4 (Region 2)

        for d1_state in d1_d2_states:
            for d2_state in d1_d2_states:
                for d3_state in d3_d4_states:
                    for d4_state in d3_d4_states:
                        states.append((d1_state, d2_state, d3_state, d4_state))

        return states

    def get_state_space(self):
        """
        Return the full state space.
        """
        return self.state_space

    def generate_dynamic_actions_and_transitions(self):
        """
        Generate the dynamic action space and transition mapping for each system configuration.
        """
        for state in self.state_space:
            self.dynamic_action_space[state] = self.get_valid_actions(state)
            self.generate_transitions_for_state(state)

    def get_valid_actions(self, state):
        """
        Get the list of valid actions based on the current state.
        """
        d1, d2, d3, d4 = state
        valid_actions = []

        for idx, action in enumerate(self.action_space):
            # Check valid actions for D1
            valid_d1 = (
                    (d1 == 0 and action[0] in [0, 0.75, 1.5]) or
                    (d1 == 0.75 and action[0] in [0, 'elevate']) or
                    (d1 == 1.5 and action[0] == 0)
            )
            # Check valid actions for D2
            valid_d2 = (
                    (d2 == 0 and action[1] in [0, 0.75, 1.5]) or
                    (d2 == 0.75 and action[1] in [0, 'elevate']) or
                    (d2 == 1.5 and action[1] == 0)
            )
            # Check valid actions for D3
            valid_d3 = (
                    (d3 == 0 and action[2] in [0, 0.75]) or
                    (d3 == 0.75 and action[2] == 0)
            )
            # Check valid actions for D4
            valid_d4 = (
                    (d4 == 0 and action[3] in [0, 0.75]) or
                    (d4 == 0.75 and action[3] == 0)
            )

            # Add the action to valid actions if all conditions are met
            if valid_d1 and valid_d2 and valid_d3 and valid_d4:
                valid_actions.append(idx)

        return valid_actions

    def generate_transitions_for_state(self, state):
        """
        Generate the transition mapping for a given state.
        """
        for action_idx in self.dynamic_action_space[state]:
            action = self.action_space[action_idx]
            next_state = (
                max(state[0], action[0] if action[0] != 'elevate' else 1.5),  # D1 height after action
                max(state[1], action[1] if action[1] != 'elevate' else 1.5),  # D2 height after action
                max(state[2], action[2]),  # D3 height after action
                max(state[3], action[3])  # D4 height after action
            )
            self.transition_mapping[(state, action_idx)] = next_state

    def step(self, action):

        # print(f"Year: {self.year}")

        # Transition in sea level rise (SLR) based on the current year and state
        trans_slr = self.trans_slr[self.year]  # 77x77 transition matrix for that year
        trans_slr_state = trans_slr[int(self.state[0])]  # 1x77 probability vector based on the current SLR state (state[0])
        next_slr = np.random.choice(np.array(range(0, self.n_states_slr)), 1,
                                    p=trans_slr_state / np.sum(trans_slr_state))  # Choose next SLR state based on the probability
        # print(f"Next SLR state: {next_slr}")

        # Transition in storm surge based on the current state
        trans_surge_state = self.trans_surge[0][
            int(self.state[1])]  # 1x72 probability vector based on the current storm surge state (state[1])
        next_surge = np.random.choice(np.array(range(0, self.n_states_surge)), 1, p=trans_surge_state)
        print(f"Year: {self.year}, Next SLR state: {next_slr}, Next Surge state: {next_surge}")

        # Combine SLR and storm surge to determine the next water height state
        next_state = np.array([next_slr, next_surge])

        # Retrieve current system configuration
        current_system = self.system

        # Update system configuration based on the action taken
        next_system = self.transition_mapping[(current_system, action)]
        self.previous_system = self.system
        self.system = next_system

        # Calculate the reward based on the next state, action taken, and the updated system
        reward_ = self.immediate_cost(next_state, action, current_system, next_system, self.year)
        # print(f"Immediate reward: {reward_}")

        # Discount and scale the reward
        reward = reward_ / 1e6
        print(f"Immediate reward: {reward_}. Scaled reward: {reward}")

        # Check if the episode is done (i.e., if the time horizon has been exceeded)
        if self.year >= self.horizon:
            done = True
            reward = 0  # You may decide to include terminal rewards here if needed
        else:
            done = False

        # Update the environment's year and state
        self.year += 1
        self.state = next_state

        # Return the next state, reward, done status, and any additional info
        info = {}
        return next_state, reward, done, info

    def render(self):
        raise NotImplemented

    def reset(self):
        # reset your environment
        self.year = 0
        self.state = self.initial_state
        self.system = self.initial_system
        return self.state

    def get_state_vector(self, observation, time):
        horizon = 40 #number of years
        slr = np.zeros(self.n_states_slr)
        slr[int(observation[0])] = 1
        # print(s.shape)
        surge = np.zeros(self.n_states_surge)
        surge[int(observation[1])] = 1
        # print(s[int(state)])
        state_combined = np.concatenate(((np.array([time / horizon])), slr, surge), axis=None)
        # state_combined = np.concatenate(((np.array([time / horizon])), slr), axis=None)
        # print(state_combined.shape)
        state_combined = np.concatenate((state_combined, np.array(self.system).T), axis=None)
        # print(state_combined.shape)
        return state_combined

    def get_transition(self, ssp):
        if ssp == '119':
            transitions_slr = mat73.loadmat('t_slr_245.mat')
            trans_slr = np.array(transitions_slr['t_slr_avg'])
            transitions_surge = mat73.loadmat('t_surge.mat')
            trans_surge = np.array(transitions_surge['t_surge'])
        elif ssp == '245':
            transitions_slr = mat73.loadmat('t_slr_245.mat')     # t_slr_245.mat is a 1 x 131 cell where each cell is a (77x77) matrix defining the transition probabilites between each of the 77 states for each of the 131 year
            trans_slr = np.array(transitions_slr['t_slr_avg'])   # Array (year, (77x77))
            transitions_surge = mat73.loadmat('t_surge.mat')     # t_surge is 1x1 cell defining the storm surge probability distribution
            trans_surge = np.array(transitions_surge['t_surge'])
        elif ssp == '585':
            transitions_slr = mat73.loadmat('t_slr_245.mat')
            trans_slr = np.array(transitions_slr['t_slr_avg'])
            transitions_surge = mat73.loadmat('t_surge.mat')
            trans_surge = np.array(transitions_surge['t_surge'])
        else:
            print(f'not valid ssp')

        return trans_slr, trans_surge

    # def get_rewards(self, env_model):
    #     if env_model == 'TwoDikes':
    #         rewards = mat73.loadmat('rewards_two_dikes_scc_tc_245.mat')
    #         self.rewards_sys = np.array(rewards['rewards'])
    #     elif env_model == 'GreenDike':
    #         rewards = mat73.loadmat('rewards_grd2_finalscc_07.mat')
    #         self.rewards_sys = np.array(rewards['rewards'])
    #     else:
    #         print(f'not valid environment model')
    #
    #     return self.rewards_sys

    def get_terminal_reward(self, env_model):
        if env_model == 'TwoDikes':
            terminal_rewards = mat73.loadmat('terminal_rewards_two_dikes_scc_tc_245.mat')
            self.terminal_rewards = np.array(terminal_rewards['terminal_rewards'])
        return self.terminal_rewards

    def get_combined(self, state):
        slr = state[0]
        surge = state[1]
        state_combined = self.n_states_surge * slr + surge

        return state_combined

    def immediate_cost(self, next_state, action, old_system, next_system, year):

        # flood damage
        c_f = -0.07 * 15.3 * 1e6        # f_damage = 0.07. Val_z = 15.3 million = 15.3*10^6. c_f = cost per unit length (USD/m)
        gamma = 0.97
        s = 0.0085      # Slope of the city 8.5/1000
        vol_z_region_1 = (0.5 * 8.50 * (1 / s) * 8.50 )   # Total volume (m^3) of the region 1. 1/2*height*base*unit_length = 1/2*8.5* (8.5*slope)*1
        vol_z_region_2 = (0.5 * 8.50 * (1 / s) * 8.50 )
        vol_z = vol_z_region_1 + vol_z_region_2

        slr_value = slr(next_state[0])/100          # Convert state to value in meters
        surge_value = surge(next_state[1])/100      # Convert state to value in meters
        total_height = slr_value + surge_value
        # total_height = slr_value*10
        discounted_sum_scc_ = mat73.loadmat('discounted_sum_scc_7_3.mat')
        discounted_sum_scc = np.array(discounted_sum_scc_['discounted_sum_scc'])    # Discounted sum from a given year to end of planning horizon

        # Base height, mid height and top heights of dikes D1
        # Keep b1_ atleast 0.75 to ensure the corresponding dikes in region 2 can be placed alongside
        # b1 = 1.5          # base of dike D1 (meters)
        m1 = self.b1 + 0.75       # Mid height of dike D1 if constructed to 0.75 meters
        t1 = m1 + 0.75        # top of dike D1 (meters)

        # Base height, mid height and top heights of dikes D2
        # b2 = 3.75          # base of dike D2 (meters)
        m2 = self.b2 + 0.75       # Mid height of dike D2 if constructed to 0.75 meters
        t2 = m2 +0.75        # top of dike D2 (meters)

        # Elevation of region 1 and 2 at shore line
        # r_1_h0 = 0
        r_2_h0 = self.r_1_h0 + 0.75

        print(f"The total height is: {total_height}")
        print(f"Next system: {next_system}")
        # flood cost
        if next_system == (0, 0, 0, 0): # System 1

            area_r_1 = 0.5 * (1 / s) * (total_height - self.r_1_h0)**2 if total_height > self.r_1_h0 else 0
            area_r_2 = 0.5 * (1 / s) * (total_height - r_2_h0) ** 2 if total_height > r_2_h0 else 0

            vol_f = area_r_1+ area_r_2

            c_flood = c_f * vol_f / vol_z
            scc_sum = discounted_sum_scc[year]
            c_flood_carbon = self.FloodDamageGHG(c_flood, scc_sum)

        elif next_system == (0, 0, 0, 0.75):    # System 2

            area_r_1 = 0.5 * (1 / s) * (total_height - self.r_1_h0)**2 if total_height > self.r_1_h0 else 0
            area_r_2 = 0.5 * (1 / s) * (total_height - r_2_h0) ** 2 if total_height > r_2_h0 else 0     # Even though D4 present, when water level between base and top of D4 (Zone 4b), lateral flow will occur from Region 1 to Region 2

            vol_f = area_r_1 + area_r_2

            c_flood = c_f * vol_f/ vol_z
            scc_sum = discounted_sum_scc[year]
            c_flood_carbon = self.FloodDamageGHG(c_flood, scc_sum)

        elif next_system == (0, 0, 0.75, 0): # System 3

            area_r_1 = 0.5 * (1 / s) * (total_height - self.r_1_h0)**2 if total_height > self.r_1_h0 else 0
            area_r_2 = 0.5 * (1 / s) * (total_height - r_2_h0) ** 2 if total_height > r_2_h0 else 0 # Even though D3 present, when water level between base and top of D3 (Zone 2b), lateral flow will occur from Region 1 to Region 2

            vol_f = area_r_1 + area_r_2

            c_flood = c_f * vol_f/ vol_z
            scc_sum = discounted_sum_scc[year]
            c_flood_carbon = self.FloodDamageGHG(c_flood, scc_sum)


        elif next_system == (0, 0, 0.75, 0.75): # System 4

            # Following same arguments as above, Dike D3 and D4 will be ineffective i.e.flooded volume same as do-nothing
            area_r_1 = 0.5 * (1 / s) * (total_height - self.r_1_h0)**2 if total_height > self.r_1_h0 else 0
            area_r_2 = 0.5 * (1 / s) * (total_height - r_2_h0) ** 2 if total_height > r_2_h0 else 0
            vol_f = area_r_1 + area_r_2

            c_flood = c_f * vol_f / vol_z
            scc_sum = discounted_sum_scc[year]
            c_flood_carbon = self.FloodDamageGHG(c_flood, scc_sum)

        elif next_system == (0, 0.75, 0, 0):    # System 5

            if total_height <= self.b2:
                area_r_1 = 0.5 * (1 / s) * (total_height - self.r_1_h0)**2 if total_height > self.r_1_h0 else 0
                area_r_2 = 0.5 * (1 / s) * (total_height - r_2_h0) ** 2 if total_height > r_2_h0 else 0
            elif (self.b2 < total_height <= m2):  # Zone 4a
                area_r_1 = 0.5 * (self.b2 - self.r_1_h0) * (self.b2 - self.r_1_h0) * (1 / s) + 0.5 * (total_height - self.b2) * (self.b2 - self.r_1_h0) * (1 / s)  # D2 will resist
                area_r_2 = 0.5 * (1 / s) * (total_height - r_2_h0) ** 2
            else: # Dike D2 becomes ineffective
                area_r_1 = 0.5 * (1 / s) * (total_height - self.r_1_h0)**2
                area_r_2 = 0.5 * (1 / s) * (total_height - r_2_h0) ** 2

            vol_f = area_r_1 + area_r_2

            c_flood = c_f * vol_f / vol_z
            scc_sum = discounted_sum_scc[year]
            c_flood_carbon = self.FloodDamageGHG(c_flood, scc_sum)

        elif next_system == (0, 0.75, 0, 0.75): # System 6

            # D4 without D2 upto 1.5 meter is ineffective. The system will behave same as the above system.
            # That is D2 will be effective until 0.75 m but everything will be flooded once D2 is overtopped

            if total_height <= self.b2:
                area_r_1 = 0.5 * (1 / s) * (total_height - self.r_1_h0)**2 if total_height > self.r_1_h0 else 0
                area_r_2 = 0.5 * (1 / s) * (total_height - r_2_h0) ** 2 if total_height > r_2_h0 else 0
            elif (self.b2 < total_height <= m2):  # Zone 4a
                area_r_1 = 0.5 * (self.b2 - self.r_1_h0) * (self.b2 - self.r_1_h0) * (1 / s) + 0.5 * (total_height - self.b2) * (self.b2 - self.r_1_h0) * (1 / s)  # D2 will resist
                area_r_2 = 0.5 * (1 / s) * (total_height - r_2_h0) ** 2
            else: # Dike D2 becomes ineffective
                area_r_1 = 0.5 * (1 / s) * (total_height - self.r_1_h0)**2
                area_r_2 = 0.5 * (1 / s) * (total_height - r_2_h0) ** 2

            vol_f = area_r_1 + area_r_2

            c_flood = c_f * vol_f / vol_z
            scc_sum = discounted_sum_scc[year]
            c_flood_carbon = self.FloodDamageGHG(c_flood, scc_sum)


        elif next_system == (0, 0.75, 0.75, 0): # System 7

            # D3 without D1 upto 1.5 meter is ineffective. The system will behave same as the system (0, 0.75, 0, 0).

            if total_height <= self.b2:
                area_r_1 = 0.5 * (1 / s) * (total_height - self.r_1_h0)**2 if total_height > self.r_1_h0 else 0
                area_r_2 = 0.5 * (1 / s) * (total_height - r_2_h0) ** 2 if total_height > r_2_h0 else 0
            elif (self.b2 < total_height <= m2):  # Zone 4a
                area_r_1 = 0.5 * (self.b2 - self.r_1_h0) * (self.b2 - self.r_1_h0) * (1 / s) + 0.5 * (total_height - self.b2) * (self.b2 - self.r_1_h0) * (1 / s)  # D2 will resist
                area_r_2 = 0.5 * (1 / s) * (total_height - r_2_h0) ** 2
            else: # Dike D2 becomes ineffective
                area_r_1 = 0.5 * (1 / s) * (total_height - self.r_1_h0)**2
                area_r_2 = 0.5 * (1 / s) * (total_height - r_2_h0) ** 2

            vol_f = area_r_1 + area_r_2

            c_flood = c_f * vol_f / vol_z
            scc_sum = discounted_sum_scc[year]
            c_flood_carbon = self.FloodDamageGHG(c_flood, scc_sum)


        elif next_system == (0, 0.75, 0.75, 0.75): # System 8

        # Following same arguments as above, Dike D3 and D4 will be ineffective. Same as system (0, 0.75, 0, 0).

            if total_height <= self.b2:
                area_r_1 = 0.5 * (1 / s) * (total_height - self.r_1_h0)**2 if total_height > self.r_1_h0 else 0
                area_r_2 = 0.5 * (1 / s) * (total_height - r_2_h0) ** 2 if total_height > r_2_h0 else 0
            elif (self.b2 < total_height <= m2):  # Zone 4a
                area_r_1 = 0.5 * (self.b2 - self.r_1_h0) * (self.b2 - self.r_1_h0) * (1 / s) + 0.5 * (total_height - self.b2) * (self.b2 - self.r_1_h0) * (1 / s)  # D2 will resist
                area_r_2 = 0.5 * (1 / s) * (total_height - r_2_h0) ** 2
            else: # Dike D2 becomes ineffective
                area_r_1 = 0.5 * (1 / s) * (total_height - self.r_1_h0)**2
                area_r_2 = 0.5 * (1 / s) * (total_height - r_2_h0) ** 2

            vol_f = area_r_1 + area_r_2

            c_flood = c_f * vol_f / vol_z
            scc_sum = discounted_sum_scc[year]
            c_flood_carbon = self.FloodDamageGHG(c_flood, scc_sum)

        elif next_system == (0, 1.5, 0, 0): # System 9

            # D2 will effectively protect zone 4a. As soon as water in zone 4b, everything flooded.
            # Again, same as system (0, 0.75, 0, 0)

            if total_height <= self.b2:
                area_r_1 = 0.5 * (1 / s) * (total_height - self.r_1_h0)**2 if total_height > self.r_1_h0 else 0
                area_r_2 = 0.5 * (1 / s) * (total_height - r_2_h0) ** 2 if total_height > r_2_h0 else 0
            elif (self.b2 < total_height <= m2):  # Zone 4a
                area_r_1 = 0.5 * (self.b2 - self.r_1_h0) * (self.b2 - self.r_1_h0) * (1 / s) + 0.5 * (total_height - self.b2) * (self.b2 - self.r_1_h0) * (1 / s)  # D2 will resist
                area_r_2 = 0.5 * (1 / s) * (total_height - r_2_h0) ** 2
            else: # Dike D2 becomes ineffective
                area_r_1 = 0.5 * (1 / s) * (total_height - self.r_1_h0)**2
                area_r_2 = 0.5 * (1 / s) * (total_height - r_2_h0) ** 2

            vol_f = area_r_1 + area_r_2

            c_flood = c_f * vol_f / vol_z
            scc_sum = discounted_sum_scc[year]
            c_flood_carbon = self.FloodDamageGHG(c_flood, scc_sum)


        elif next_system == (0, 1.5, 0, 0.75): # System 10

            # D2 will protect zone 4a as well as 4b since top of D2 and D4 match, there will be no lateral flow.
            # D4 will be effective as well and protect zone 4
            if total_height <= self.b2:
                area_r_1 = 0.5 * (1 / s) * (total_height - self.r_1_h0)**2 if total_height > self.r_1_h0 else 0
                area_r_2 = 0.5 * (1 / s) * (total_height - r_2_h0) ** 2 if total_height > r_2_h0 else 0
            elif (self.b2 < total_height <= m2): # zone 4a
                area_r_1 = 0.5 * (self.b2 - self.r_1_h0) * (self.b2 - self.r_1_h0) * (1 / s) + 0.5 * (total_height - self.b2) * (self.b2 - self.r_1_h0) * (1 / s)  # D2 will resist
                area_r_2 = 0.5 * (1 / s) * (total_height - r_2_h0) ** 2   # Region 2 completely submerged upto zone 3
            elif (m2 < total_height <= t2):  # Zone 4b. Coincides with zone 4 of region 2
                area_r_1 = 0.5 * (self.b2 - self.r_1_h0) * (self.b2 - self.r_1_h0) * (1 / s) + 0.5 * (total_height - self.b2) * (self.b2 - self.r_1_h0) * (1 / s)
                # Base of D4 = m2
                area_r_2 = 0.5 * (m2 - r_2_h0) * (m2 - r_2_h0) * (1 / s) + 0.5 * (total_height - m2) * (m2 - r_2_h0) * (1 / s)  # D4 will resist
            else: # everything submerged
                area_r_1 = 0.5 * (1 / s) * (total_height - self.r_1_h0)**2
                area_r_2 = 0.5 * (1 / s) * (total_height - r_2_h0) ** 2

            vol_f = area_r_1 + area_r_2

            c_flood = c_f * vol_f / vol_z
            scc_sum = discounted_sum_scc[year]
            c_flood_carbon = self.FloodDamageGHG(c_flood, scc_sum)


        elif next_system == (0, 1.5, 0.75, 0):     # System 11

            # D3 ineffective since D1=0
            # D2 effective upto 0.75 meter since D4=0
            # Hence, same as system (0, 0.75, 0, 0)

            if total_height <= self.b2:
                area_r_1 = 0.5 * (1 / s) * (total_height - self.r_1_h0)**2 if total_height > self.r_1_h0 else 0
                area_r_2 = 0.5 * (1 / s) * (total_height - r_2_h0) ** 2 if total_height > r_2_h0 else 0
            elif (self.b2 < total_height <= m2):  # Zone 4a
                area_r_1 = 0.5 * (self.b2 - self.r_1_h0) * (self.b2 - self.r_1_h0) * (1 / s) + 0.5 * (total_height - self.b2) * (self.b2 - self.r_1_h0) * (1 / s)  # D2 will resist
                area_r_2 = 0.5 * (1 / s) * (total_height - r_2_h0) ** 2
            else: # Dike D2 becomes ineffective
                area_r_1 = 0.5 * (1 / s) * (total_height - self.r_1_h0)**2
                area_r_2 = 0.5 * (1 / s) * (total_height - r_2_h0) ** 2


            vol_f = area_r_1 + area_r_2

            c_flood = c_f * vol_f / vol_z
            scc_sum = discounted_sum_scc[year]
            c_flood_carbon = self.FloodDamageGHG(c_flood, scc_sum)


        elif next_system == (0, 1.5, 0.75, 0.75): # System 12

            # D3 ineffective since D1=0
            # Same as System 10 (0, 1.5, 0, 0.75)
            if total_height <= self.b2:
                area_r_1 = 0.5 * (1 / s) * (total_height - self.r_1_h0)**2 if total_height > self.r_1_h0 else 0
                area_r_2 = 0.5 * (1 / s) * (total_height - r_2_h0) ** 2 if total_height > r_2_h0 else 0
            elif (self.b2 < total_height <= m2): # zone 4a
                area_r_1 = 0.5 * (self.b2 - self.r_1_h0) * (self.b2 - self.r_1_h0) * (1 / s) + 0.5 * (total_height - self.b2) * (self.b2 - self.r_1_h0) * (1 / s)  # D2 will resist
                area_r_2 = 0.5 * (1 / s) * (total_height - r_2_h0) ** 2   # Region 2 completely submerged upto zone 3
            elif (m2 < total_height <= t2):  # Zone 4b. Coincides with zone 4 of region 2
                area_r_1 = 0.5 * (self.b2 - self.r_1_h0) * (self.b2 - self.r_1_h0) * (1 / s) + 0.5 * (total_height - self.b2) * (self.b2 - self.r_1_h0) * (1 / s)
                # Base of D4 = m2
                area_r_2 = 0.5 * (m2 - r_2_h0) * (m2 - r_2_h0) * (1 / s) + 0.5 * (total_height - m2) * (m2 - r_2_h0) * (1 / s)  # D4 will resist
            else: # everything submerged
                area_r_1 = 0.5 * (1 / s) * (total_height - self.r_1_h0)**2
                area_r_2 = 0.5 * (1 / s) * (total_height - r_2_h0) ** 2

            vol_f = area_r_1 + area_r_2

            c_flood = c_f * vol_f / vol_z
            scc_sum = discounted_sum_scc[year]
            c_flood_carbon = self.FloodDamageGHG(c_flood, scc_sum)


        elif next_system == (0.75, 0, 0, 0): # System 13

            # D1 will resist zone 2a, else ineffective

            if total_height <= self.b1:
                area_r_1 = 0.5 * (1 / s) * (total_height - self.r_1_h0)**2 if total_height > self.r_1_h0 else 0
                area_r_2 = 0.5 * (1 / s) * (total_height - r_2_h0) ** 2 if total_height > r_2_h0 else 0
            elif (self.b1 < total_height <= m1):  # Zone 2a
                area_r_1 = 0.5 * (self.b1 - self.r_1_h0) * (self.b1 - self.r_1_h0) * (1 / s) + 0.5 * (total_height - self.b1) * (self.b1 - self.r_1_h0) * (1 / s)
                area_r_2 = 0.5 * (1 / s) * (total_height - r_2_h0) ** 2
            else:  # Dike D1 becomes ineffective, everything submerged
                area_r_1 = 0.5 * (1 / s) * (total_height - self.r_1_h0)**2
                area_r_2 = 0.5 * (1 / s) * (total_height - r_2_h0) ** 2

            vol_f = area_r_1 + area_r_2
            c_flood = c_f * vol_f / vol_z
            scc_sum = discounted_sum_scc[year]
            c_flood_carbon = self.FloodDamageGHG(c_flood, scc_sum)

        elif next_system == (0.75, 0, 0, 0.75):  # System 14

            # D4 ineffective
            # D1 will resist zone 2a, else ineffective
            # Same as System 13 (0.75, 0, 0, 0)

            if total_height <= self.b1:
                area_r_1 = 0.5 * (1 / s) * (total_height - self.r_1_h0)**2 if total_height > self.r_1_h0 else 0
                area_r_2 = 0.5 * (1 / s) * (total_height - r_2_h0) ** 2 if total_height > r_2_h0 else 0
            elif (self.b1 < total_height <= m1):  # Zone 2a
                area_r_1 = 0.5 * (self.b1 - self.r_1_h0) * (self.b1 - self.r_1_h0) * (1 / s) + 0.5 * (total_height - self.b1) * (self.b1 - self.r_1_h0) * (1 / s)
                area_r_2 = 0.5 * (1 / s) * (total_height - r_2_h0) ** 2
            else:  # Dike D1 becomes ineffective, everything submerged
                area_r_1 = 0.5 * (1 / s) * (total_height - self.r_1_h0)**2
                area_r_2 = 0.5 * (1 / s) * (total_height - r_2_h0) ** 2

            vol_f = area_r_1 + area_r_2
            c_flood = c_f * vol_f / vol_z
            scc_sum = discounted_sum_scc[year]
            c_flood_carbon = self.FloodDamageGHG(c_flood, scc_sum)

        elif next_system == (0.75, 0, 0.75, 0):  # System 15

            # D3 ineffective
            # D1 will resist zone 2a, else ineffective
            # Same as system 13 (0.75, 0, 0.75, 0)
            if total_height <= self.b1:
                area_r_1 = 0.5 * (1 / s) * (total_height - self.r_1_h0)**2 if total_height > self.r_1_h0 else 0
                area_r_2 = 0.5 * (1 / s) * (total_height - r_2_h0) ** 2 if total_height > r_2_h0 else 0
            elif (self.b1 < total_height <= m1):  # Zone 2a
                area_r_1 = 0.5 * (self.b1 - self.r_1_h0) * (self.b1 - self.r_1_h0) * (1 / s) + 0.5 * (total_height - self.b1) * (self.b1 - self.r_1_h0) * (1 / s)
                area_r_2 = 0.5 * (1 / s) * (total_height - r_2_h0) ** 2
            else:  # Dike D1 becomes ineffective, everything submerged
                area_r_1 = 0.5 * (1 / s) * (total_height - self.r_1_h0)**2
                area_r_2 = 0.5 * (1 / s) * (total_height - r_2_h0) ** 2

            vol_f = area_r_1 + area_r_2
            c_flood = c_f * vol_f / vol_z
            scc_sum = discounted_sum_scc[year]
            c_flood_carbon = self.FloodDamageGHG(c_flood, scc_sum)

        elif next_system == (0.75, 0, 0.75, 0.75):  # System 16

            # D3 ineffective since D1=0.75
            # D4 ineffective since D2=0
            # D1 will resist zone 2a, else ineffective
            # Same as system 13: (0.75, 0, 0, 0)

            if total_height <= self.b1:
                area_r_1 = 0.5 * (1 / s) * (total_height - self.r_1_h0)**2 if total_height > self.r_1_h0 else 0
                area_r_2 = 0.5 * (1 / s) * (total_height - r_2_h0) ** 2 if total_height > r_2_h0 else 0
            elif (self.b1 < total_height <= m1):  # Zone 2a
                area_r_1 = 0.5 * (self.b1 - self.r_1_h0) * (self.b1 - self.r_1_h0) * (1 / s) + 0.5 * (total_height - self.b1) * (self.b1 - self.r_1_h0) * (1 / s)
                area_r_2 = 0.5 * (1 / s) * (total_height - r_2_h0) ** 2
            else:  # Dike D1 becomes ineffective, everything submerged
                area_r_1 = 0.5 * (1 / s) * (total_height - self.r_1_h0)**2
                area_r_2 = 0.5 * (1 / s) * (total_height - r_2_h0) ** 2

            vol_f = area_r_1 + area_r_2
            c_flood = c_f * vol_f / vol_z
            scc_sum = discounted_sum_scc[year]
            c_flood_carbon = self.FloodDamageGHG(c_flood, scc_sum)

        elif next_system == (0.75, 0.75, 0, 0):  # System 17

            # D1 will resist zone 2a
            # D2 will resits zone 4a

            if total_height <= self.b1:
                area_r_1 = 0.5 * (1 / s) * (total_height - self.r_1_h0)**2 if total_height > self.r_1_h0 else 0
                area_r_2 = 0.5 * (1 / s) * (total_height - r_2_h0) ** 2 if total_height > r_2_h0 else 0
            elif (self.b1 < total_height <= m1):  # Zone 2a
                area_r_1 = 0.5 * (self.b1 - self.r_1_h0) * (self.b1 - self.r_1_h0) * (1 / s) + 0.5 * (total_height - self.b1) * (self.b1 - self.r_1_h0) * (1 / s)   # D1 will resist
                area_r_2 = 0.5 * (1 / s) * (total_height - r_2_h0) ** 2  # Region 2 completely submerged
            elif (self.b2 < total_height <= m2):  # zone 4a
                area_r_1 = 0.5 * (self.b2 - self.r_1_h0) * (self.b2 - self.r_1_h0) * (1 / s) + 0.5 * (total_height - self.b2) * (self.b2 - self.r_1_h0) * (1 / s)
                area_r_2 = 0.5 * (1 / s) * (total_height - r_2_h0) ** 2  # Region 2 completely submerged upto zone 3
            else:  # everything submerged
                area_r_1 = 0.5 * (1 / s) * (total_height - self.r_1_h0)**2
                area_r_2 = 0.5 * (1 / s) * (total_height - r_2_h0) ** 2

            vol_f = area_r_1 + area_r_2
            c_flood = c_f * vol_f / vol_z
            scc_sum = discounted_sum_scc[year]
            c_flood_carbon = self.FloodDamageGHG(c_flood, scc_sum)

        elif next_system == (0.75, 0.75, 0, 0.75):  # System 18

            # D1 will resist zone 2a
            # D2 will resits zone 2b
            # D4 ineffective since D2=0.75
            # Same as system 17 (0.75, 0.75, 0, 0)

            if total_height <= self.b1:
                area_r_1 = 0.5 * (1 / s) * (total_height - self.r_1_h0)**2 if total_height > self.r_1_h0 else 0
                area_r_2 = 0.5 * (1 / s) * (total_height - r_2_h0) ** 2 if total_height > r_2_h0 else 0
            elif (self.b1 < total_height <= m1):  # Zone 2a
                area_r_1 = 0.5 * (self.b1 - self.r_1_h0) * (self.b1 - self.r_1_h0) * (1 / s) + 0.5 * (total_height - self.b1) * (self.b1 - self.r_1_h0) * (1 / s)   # D1 will resist
                area_r_2 = 0.5 * (1 / s) * (total_height - r_2_h0) ** 2  # Region 2 completely submerged
            elif (self.b2 < total_height <= m2):  # zone 4a
                area_r_1 = 0.5 * (self.b2 - self.r_1_h0) * (self.b2 - self.r_1_h0) * (1 / s) + 0.5 * (total_height - self.b2) * (self.b2 - self.r_1_h0) * (1 / s)
                area_r_2 = 0.5 * (1 / s) * (total_height - r_2_h0) ** 2  # Region 2 completely submerged upto zone 3
            else:  # everything submerged
                area_r_1 = 0.5 * (1 / s) * (total_height - self.r_1_h0)**2
                area_r_2 = 0.5 * (1 / s) * (total_height - r_2_h0) ** 2

            vol_f = area_r_1 + area_r_2
            c_flood = c_f * vol_f / vol_z
            scc_sum = discounted_sum_scc[year]
            c_flood_carbon = self.FloodDamageGHG(c_flood, scc_sum)

        elif next_system == (0.75, 0.75, 0.75, 0):  # System 19

            # D1 will resist zone 2a
            # D2 will resits zone 2b
            # D3 ineffective since D1=0.75
            # Same as system 17 (0.75, 0.75, 0, 0)

            if total_height <= self.b1:
                area_r_1 = 0.5 * (1 / s) * (total_height - self.r_1_h0)**2 if total_height > self.r_1_h0 else 0
                area_r_2 = 0.5 * (1 / s) * (total_height - r_2_h0) ** 2 if total_height > r_2_h0 else 0
            elif (self.b1 < total_height <= m1):  # Zone 2a
                area_r_1 = 0.5 * (self.b1 - self.r_1_h0) * (self.b1 - self.r_1_h0) * (1 / s) + 0.5 * (total_height - self.b1) * (self.b1 - self.r_1_h0) * (1 / s)   # D1 will resist
                area_r_2 = 0.5 * (1 / s) * (total_height - r_2_h0) ** 2  # Region 2 completely submerged
            elif (self.b2 < total_height <= m2):  # zone 4a
                area_r_1 = 0.5 * (self.b2 - self.r_1_h0) * (self.b2 - self.r_1_h0) * (1 / s) + 0.5 * (total_height - self.b2) * (self.b2 - self.r_1_h0) * (1 / s)
                area_r_2 = 0.5 * (1 / s) * (total_height - r_2_h0) ** 2  # Region 2 completely submerged upto zone 3
            else:  # everything submerged
                area_r_1 = 0.5 * (1 / s) * (total_height - self.r_1_h0)**2
                area_r_2 = 0.5 * (1 / s) * (total_height - r_2_h0) ** 2

            vol_f = area_r_1 + area_r_2

            c_flood = c_f * vol_f / vol_z
            scc_sum = discounted_sum_scc[year]
            c_flood_carbon = self.FloodDamageGHG(c_flood, scc_sum)

        elif next_system == (0.75, 0.75, 0.75, 0.75):  # System 20

            # D1 will resist zone 2a
            # D2 will resits zone 2b
            # D3 & D4 ineffective
            # Same as system 17 (0.75, 0.75, 0, 0)

            if total_height <= self.b1:
                area_r_1 = 0.5 * (1 / s) * (total_height - self.r_1_h0)**2 if total_height > self.r_1_h0 else 0
                area_r_2 = 0.5 * (1 / s) * (total_height - r_2_h0) ** 2 if total_height > r_2_h0 else 0
            elif (self.b1 < total_height <= m1):  # Zone 2a
                area_r_1 = 0.5 * (self.b1 - self.r_1_h0) * (self.b1 - self.r_1_h0) * (1 / s) + 0.5 * (total_height - self.b1) * (self.b1 - self.r_1_h0) * (1 / s)   # D1 will resist
                area_r_2 = 0.5 * (1 / s) * (total_height - r_2_h0) ** 2  # Region 2 completely submerged
            elif (self.b2 < total_height <= m2):  # zone 4a
                area_r_1 = 0.5 * (self.b2 - self.r_1_h0) * (self.b2 - self.r_1_h0) * (1 / s) + 0.5 * (total_height - self.b2) * (self.b2 - self.r_1_h0) * (1 / s)
                area_r_2 = 0.5 * (1 / s) * (total_height - r_2_h0) ** 2  # Region 2 completely submerged upto zone 3
            else:  # everything submerged
                area_r_1 = 0.5 * (1 / s) * (total_height - self.r_1_h0)**2
                area_r_2 = 0.5 * (1 / s) * (total_height - r_2_h0) ** 2

            vol_f = area_r_1 + area_r_2

            c_flood = c_f * vol_f / vol_z
            scc_sum = discounted_sum_scc[year]
            c_flood_carbon = self.FloodDamageGHG(c_flood, scc_sum)

        elif next_system == (0.75, 1.5, 0, 0):  # System 21

            # D1 will resist zone 2a
            # D2 will resist zone 4a
            # Same as system 17 (0.75, 0.75, 0, 0)

            if total_height <= self.b1:
                area_r_1 = 0.5 * (1 / s) * (total_height - self.r_1_h0)**2 if total_height > self.r_1_h0 else 0
                area_r_2 = 0.5 * (1 / s) * (total_height - r_2_h0) ** 2 if total_height > r_2_h0 else 0
            elif (self.b1 < total_height <= m1):  # Zone 2a
                area_r_1 = 0.5 * (self.b1 - self.r_1_h0) * (self.b1 - self.r_1_h0) * (1 / s) + 0.5 * (total_height - self.b1) * (self.b1 - self.r_1_h0) * (1 / s)   # D1 will resist
                area_r_2 = 0.5 * (1 / s) * (total_height - r_2_h0) ** 2  # Region 2 completely submerged
            elif (self.b2 < total_height <= m2):  # zone 4a
                area_r_1 = 0.5 * (self.b2 - self.r_1_h0) * (self.b2 - self.r_1_h0) * (1 / s) + 0.5 * (total_height - self.b2) * (self.b2 - self.r_1_h0) * (1 / s)
                area_r_2 = 0.5 * (1 / s) * (total_height - r_2_h0) ** 2  # Region 2 completely submerged upto zone 3
            else:  # everything submerged
                area_r_1 = 0.5 * (1 / s) * (total_height - self.r_1_h0)**2
                area_r_2 = 0.5 * (1 / s) * (total_height - r_2_h0) ** 2

            vol_f = area_r_1 + area_r_2

            c_flood = c_f * vol_f / vol_z
            scc_sum = discounted_sum_scc[year]
            c_flood_carbon = self.FloodDamageGHG(c_flood, scc_sum)

        elif next_system == (0.75, 1.5, 0, 0.75):  # System 22

            # D1 will resist zone 2a
            # D2 will resist zone 4a as well as zone 4b
            # D4 will resist zone 4

            if total_height <= self.b1:
                area_r_1 = 0.5 * (1 / s) * (total_height - self.r_1_h0)**2 if total_height > self.r_1_h0 else 0
                area_r_2 = 0.5 * (1 / s) * (total_height - r_2_h0) ** 2 if total_height > r_2_h0 else 0
            elif (self.b1 < total_height <= m1):  # Zone 2a
                area_r_1 = 0.5 * (self.b1 - self.r_1_h0) * (self.b1 - self.r_1_h0) * (1 / s) + 0.5 * (total_height - self.b1) * (self.b1 - self.r_1_h0) * (1 / s)   # D1 will resist
                area_r_2 = 0.5 * (1 / s) * (total_height - r_2_h0) ** 2
            elif (self.b2 < total_height <= m2): # zone 4a
                area_r_1 = 0.5 * (self.b2 - self.r_1_h0) * (self.b2 - self.r_1_h0) * (1 / s) + 0.5 * (total_height - self.b2) * (self.b2 - self.r_1_h0) * (1 / s) # D2 will resist
                area_r_2 = 0.5 * (1 / s) * (total_height - r_2_h0) ** 2   # Region 2 completely submerged upto zone 3
            elif (m2 < total_height <= t2):  # Zone 4b. Coincides with zone 4 of region 2
                area_r_1 = 0.5 * (self.b2 - self.r_1_h0) * (self.b2 - self.r_1_h0) * (1 / s) + 0.5 * (total_height - self.b2) * (self.b2 - self.r_1_h0) * (1 / s)
                # Base of D4 = m2
                area_r_2 = 0.5 * (m2 - r_2_h0) * (m2 - r_2_h0) * (1 / s) + 0.5 * (total_height - m2) * (m2 - r_2_h0) * (1 / s)  # D4 will resist
            else:  # Dike D1 becomes ineffective, everything submerged
                area_r_1 = 0.5 * (1 / s) * (total_height - self.r_1_h0)**2
                area_r_2 = 0.5 * (1 / s) * (total_height - r_2_h0) ** 2

            vol_f = area_r_1 + area_r_2
            c_flood = c_f * vol_f / vol_z
            scc_sum = discounted_sum_scc[year]
            c_flood_carbon = self.FloodDamageGHG(c_flood, scc_sum)

        elif next_system == (0.75, 1.5, 0.75, 0):  # System 23

            # D1 will resist zone 2a
            # D2 will resist zone 4a
            # D3 ineffective
            # Same as system 17 (0.75, 0.75, 0, 0)

            if total_height <= self.b1:
                area_r_1 = 0.5 * (1 / s) * (total_height - self.r_1_h0)**2 if total_height > self.r_1_h0 else 0
                area_r_2 = 0.5 * (1 / s) * (total_height - r_2_h0) ** 2 if total_height > r_2_h0 else 0
            elif (self.b1 < total_height <= m1):  # Zone 2a
                area_r_1 = 0.5 * (self.b1 - self.r_1_h0) * (self.b1 - self.r_1_h0) * (1 / s) + 0.5 * (total_height - self.b1) * (self.b1 - self.r_1_h0) * (1 / s)   # D1 will resist
                area_r_2 = 0.5 * (1 / s) * (total_height - r_2_h0) ** 2  # Region 2 completely submerged
            elif (self.b2 < total_height <= m2):  # zone 4a
                area_r_1 = 0.5 * (self.b2 - self.r_1_h0) * (self.b2 - self.r_1_h0) * (1 / s) + 0.5 * (total_height - self.b2) * (self.b2 - self.r_1_h0) * (1 / s)
                area_r_2 = 0.5 * (1 / s) * (total_height - r_2_h0) ** 2  # Region 2 completely submerged upto zone 3
            else:  # everything submerged
                area_r_1 = 0.5 * (1 / s) * (total_height - self.r_1_h0)**2
                area_r_2 = 0.5 * (1 / s) * (total_height - r_2_h0) ** 2

            vol_f = area_r_1 + area_r_2
            c_flood = c_f * vol_f / vol_z
            scc_sum = discounted_sum_scc[year]
            c_flood_carbon = self.FloodDamageGHG(c_flood, scc_sum)

        elif next_system == (0.75, 1.5, 0.75, 0.75):  # System 24

            # D1 will resist zone 2a
            # D2 will resist zone 4a and 4b
            # D3 ineffective
            # D4 will resist zone 4
            # Same as System 22 (0.75, 1.5, 0, 0.75)

            if total_height <= self.b1:
                area_r_1 = 0.5 * (1 / s) * (total_height - self.r_1_h0)**2 if total_height > self.r_1_h0 else 0
                area_r_2 = 0.5 * (1 / s) * (total_height - r_2_h0) ** 2 if total_height > r_2_h0 else 0
            elif (self.b1 < total_height <= m1):  # Zone 2a
                area_r_1 = 0.5 * (self.b1 - self.r_1_h0) * (self.b1 - self.r_1_h0) * (1 / s) + 0.5 * (total_height - self.b1) * (self.b1 - self.r_1_h0) * (1 / s)   # D1 will resist
                area_r_2 = 0.5 * (1 / s) * (total_height - r_2_h0) ** 2
            elif (self.b2 < total_height <= m2): # zone 4a
                area_r_1 = 0.5 * (self.b2 - self.r_1_h0) * (self.b2 - self.r_1_h0) * (1 / s) + 0.5 * (total_height - self.b2) * (self.b2 - self.r_1_h0) * (1 / s) # D2 will resist
                area_r_2 = 0.5 * (1 / s) * (total_height - r_2_h0) ** 2   # Region 2 completely submerged upto zone 3
            elif (m2 < total_height <= t2):  # Zone 4b. Coincides with zone 4 of region 2
                area_r_1 = 0.5 * (self.b2 - self.r_1_h0) * (self.b2 - self.r_1_h0) * (1 / s) + 0.5 * (total_height - self.b2) * (self.b2 - self.r_1_h0) * (1 / s)
                # Base of D4 = m2
                area_r_2 = 0.5 * (m2 - r_2_h0) * (m2 - r_2_h0) * (1 / s) + 0.5 * (total_height - m2) * (m2 - r_2_h0) * (1 / s)  # D4 will resist
            else:  # Dike D1 becomes ineffective, everything submerged
                area_r_1 = 0.5 * (1 / s) * (total_height - self.r_1_h0)**2
                area_r_2 = 0.5 * (1 / s) * (total_height - r_2_h0) ** 2

            vol_f = area_r_1 + area_r_2
            c_flood = c_f * vol_f / vol_z
            scc_sum = discounted_sum_scc[year]
            c_flood_carbon = self.FloodDamageGHG(c_flood, scc_sum)


        elif next_system == (1.5, 0, 0, 0):  # System 25

            # D1 will resist zone 2a
            # Same as System 13 (0.75, 0, 0, 0)

            if total_height <= self.b1:
                area_r_1 = 0.5 * (1 / s) * (total_height - self.r_1_h0)**2 if total_height > self.r_1_h0 else 0
                area_r_2 = 0.5 * (1 / s) * (total_height - r_2_h0) ** 2 if total_height > r_2_h0 else 0
            elif (self.b1 < total_height <= m1):  # Zone 2a
                area_r_1 = 0.5 * (self.b1 - self.r_1_h0) * (self.b1 - self.r_1_h0) * (1 / s) + 0.5 * (total_height - self.b1) * (self.b1 - self.r_1_h0) * (1 / s)
                area_r_2 = 0.5 * (1 / s) * (total_height - r_2_h0) ** 2
            else:  # Dike D1 becomes ineffective, everything submerged
                area_r_1 = 0.5 * (1 / s) * (total_height - self.r_1_h0)**2
                area_r_2 = 0.5 * (1 / s) * (total_height - r_2_h0) ** 2

            vol_f = area_r_1 + area_r_2
            c_flood = c_f * vol_f / vol_z
            scc_sum = discounted_sum_scc[year]
            c_flood_carbon = self.FloodDamageGHG(c_flood, scc_sum)

        elif next_system == (1.5, 0, 0, 0.75):  # System 26

            # D1 will resist zone 2a
            # D4 ineffective
            # Same as System 13 (0.75, 0, 0, 0)

            if total_height <= self.b1:
                area_r_1 = 0.5 * (1 / s) * (total_height - self.r_1_h0)**2 if total_height > self.r_1_h0 else 0
                area_r_2 = 0.5 * (1 / s) * (total_height - r_2_h0) ** 2 if total_height > r_2_h0 else 0
            elif (self.b1 < total_height <= m1):  # Zone 2a
                area_r_1 = 0.5 * (self.b1 - self.r_1_h0) * (self.b1 - self.r_1_h0) * (1 / s) + 0.5 * (total_height - self.b1) * (self.b1 - self.r_1_h0) * (1 / s)
                area_r_2 = 0.5 * (1 / s) * (total_height - r_2_h0) ** 2
            else:  # Dike D1 becomes ineffective, everything submerged
                area_r_1 = 0.5 * (1 / s) * (total_height - self.r_1_h0)**2
                area_r_2 = 0.5 * (1 / s) * (total_height - r_2_h0) ** 2

            vol_f = area_r_1 + area_r_2
            c_flood = c_f * vol_f / vol_z
            scc_sum = discounted_sum_scc[year]
            c_flood_carbon = self.FloodDamageGHG(c_flood, scc_sum)

        elif next_system == (1.5, 0, 0.75, 0):  # System 27

            # D1 will resist zone 2a as well as zone 2b
            # D3 will resist zone 2

            if total_height <= self.b1:
                area_r_1 = 0.5 * (1 / s) * (total_height - self.r_1_h0)**2 if total_height > self.r_1_h0 else 0
                area_r_2 = 0.5 * (1 / s) * (total_height - r_2_h0) ** 2 if total_height > r_2_h0 else 0
            elif (self.b1 < total_height <= m1):  # Zone 2a
                area_r_1 = 0.5 * (self.b1 - self.r_1_h0) * (self.b1 - self.r_1_h0) * (1 / s) + 0.5 * (total_height - self.b1) * (self.b1 - self.r_1_h0) * (1 / s)
                area_r_2 = 0.5 * (1 / s) * (total_height - r_2_h0) ** 2
            elif (m1 < total_height <= t1): # Zone 2b. Coincides with zone 2 of region 2
                area_r_1 = 0.5 * (self.b1 - self.r_1_h0) * (self.b1 - self.r_1_h0) * (1 / s) + 0.5 * (total_height - self.b1) * (self.b1 - self.r_1_h0) * (1 / s)  # D1 will resist
                # Base of D3 = m1
                area_r_2 = 0.5 * (m1 - r_2_h0) * (m1 - r_2_h0) * (1 / s) + 0.5 * (total_height - m1) * (m1 - r_2_h0) * (1 / s) # D3 will resist
            else: # everything submerged
                area_r_1 = 0.5 * (1 / s) * (total_height - self.r_1_h0)**2
                area_r_2 = 0.5 * (1 / s) * (total_height - r_2_h0) ** 2

            vol_f = area_r_1 + area_r_2
            c_flood = c_f * vol_f / vol_z
            scc_sum = discounted_sum_scc[year]
            c_flood_carbon = self.FloodDamageGHG(c_flood, scc_sum)

        elif next_system == (1.5, 0, 0.75, 0.75):  # System 28

            # D1 will resist zone 2a and 2b
            # D3 will resist zone 2
            # D4 will be ineffective
            # Same as System 27 (1.5, 0, 0.75, 0)

            if total_height <= self.b1:
                area_r_1 = 0.5 * (1 / s) * (total_height - self.r_1_h0)**2 if total_height > self.r_1_h0 else 0
                area_r_2 = 0.5 * (1 / s) * (total_height - r_2_h0) ** 2 if total_height > r_2_h0 else 0
            elif (self.b1 < total_height <= m1):  # Zone 2a
                area_r_1 = 0.5 * (self.b1 - self.r_1_h0) * (self.b1 - self.r_1_h0) * (1 / s) + 0.5 * (total_height - self.b1) * (self.b1 - self.r_1_h0) * (1 / s)
                area_r_2 = 0.5 * (1 / s) * (total_height - r_2_h0) ** 2
            elif (m1 < total_height <= t1): # Zone 2b. Coincides with zone 2 of region 2
                area_r_1 = 0.5 * (self.b1 - self.r_1_h0) * (self.b1 - self.r_1_h0) * (1 / s) + 0.5 * (total_height - self.b1) * (self.b1 - self.r_1_h0) * (1 / s)  # D1 will resist
                # Base of D3 = m1
                area_r_2 = 0.5 * (m1 - r_2_h0) * (m1 - r_2_h0) * (1 / s) + 0.5 * (total_height - m1) * (m1 - r_2_h0) * (1 / s) # D3 will resist
            else: # everything submerged
                area_r_1 = 0.5 * (1 / s) * (total_height - self.r_1_h0)**2
                area_r_2 = 0.5 * (1 / s) * (total_height - r_2_h0) ** 2

            vol_f = area_r_1 + area_r_2
            c_flood = c_f * vol_f / vol_z
            scc_sum = discounted_sum_scc[year]
            c_flood_carbon = self.FloodDamageGHG(c_flood, scc_sum)

        elif next_system == (1.5, 0.75, 0, 0):  # System 29

            # D1 will resist zone 2a
            # D2 will resist zone 4a
            # Same as System 17 (0.75, 0.75, 0, 0)

            if total_height <= self.b1:
                area_r_1 = 0.5 * (1 / s) * (total_height - self.r_1_h0)**2 if total_height > self.r_1_h0 else 0
                area_r_2 = 0.5 * (1 / s) * (total_height - r_2_h0) ** 2 if total_height > r_2_h0 else 0
            elif (self.b1 < total_height <= m1):  # Zone 2a
                area_r_1 = 0.5 * (self.b1 - self.r_1_h0) * (self.b1 - self.r_1_h0) * (1 / s) + 0.5 * (total_height - self.b1) * (self.b1 - self.r_1_h0) * (1 / s)   # D1 will resist
                area_r_2 = 0.5 * (1 / s) * (total_height - r_2_h0) ** 2  # Region 2 completely submerged
            elif (self.b2 < total_height <= m2):  # zone 4a
                area_r_1 = 0.5 * (self.b2 - self.r_1_h0) * (self.b2 - self.r_1_h0) * (1 / s) + 0.5 * (total_height - self.b2) * (self.b2 - self.r_1_h0) * (1 / s)
                area_r_2 = 0.5 * (1 / s) * (total_height - r_2_h0) ** 2  # Region 2 completely submerged upto zone 3
            else:  # everything submerged
                area_r_1 = 0.5 * (1 / s) * (total_height - self.r_1_h0)**2
                area_r_2 = 0.5 * (1 / s) * (total_height - r_2_h0) ** 2

            vol_f = area_r_1 + area_r_2
            c_flood = c_f * vol_f / vol_z
            scc_sum = discounted_sum_scc[year]
            c_flood_carbon = self.FloodDamageGHG(c_flood, scc_sum)

        elif next_system == (1.5, 0.75, 0, 0.75):  # System 30

            # D1 will resist zone 2a
            # D2 will resist zone 4a
            # D4 ineffective
            # Same as System 17 (0.75, 0.75, 0, 0)

            if total_height <= self.b1:
                area_r_1 = 0.5 * (1 / s) * (total_height - self.r_1_h0)**2 if total_height > self.r_1_h0 else 0
                area_r_2 = 0.5 * (1 / s) * (total_height - r_2_h0) ** 2 if total_height > r_2_h0 else 0
            elif (self.b1 < total_height <= m1):  # Zone 2a
                area_r_1 = 0.5 * (self.b1 - self.r_1_h0) * (self.b1 - self.r_1_h0) * (1 / s) + 0.5 * (total_height - self.b1) * (self.b1 - self.r_1_h0) * (1 / s)   # D1 will resist
                area_r_2 = 0.5 * (1 / s) * (total_height - r_2_h0) ** 2  # Region 2 completely submerged
            elif (self.b2 < total_height <= m2):  # zone 4a
                area_r_1 = 0.5 * (self.b2 - self.r_1_h0) * (self.b2 - self.r_1_h0) * (1 / s) + 0.5 * (total_height - self.b2) * (self.b2 - self.r_1_h0) * (1 / s)
                area_r_2 = 0.5 * (1 / s) * (total_height - r_2_h0) ** 2  # Region 2 completely submerged upto zone 3
            else:  # everything submerged
                area_r_1 = 0.5 * (1 / s) * (total_height - self.r_1_h0)**2
                area_r_2 = 0.5 * (1 / s) * (total_height - r_2_h0) ** 2

            vol_f = area_r_1 + area_r_2
            c_flood = c_f * vol_f / vol_z
            scc_sum = discounted_sum_scc[year]
            c_flood_carbon = self.FloodDamageGHG(c_flood, scc_sum)

        elif next_system == (1.5, 0.75, 0.75, 0):  # System 31

            # D1 will resist zone 2a and 2b. D3 resist zone 2
            # D2 will resist zone 4a

            if total_height <= self.b1:
                area_r_1 = 0.5 * (1 / s) * (total_height - self.r_1_h0)**2 if total_height > self.r_1_h0 else 0
                area_r_2 = 0.5 * (1 / s) * (total_height - r_2_h0) ** 2 if total_height > r_2_h0 else 0
            elif (self.b1 < total_height <= m1):  # Zone 2a
                area_r_1 = 0.5 * (self.b1 - self.r_1_h0) * (self.b1 - self.r_1_h0) * (1 / s) + 0.5 * (total_height - self.b1) * (self.b1 - self.r_1_h0) * (1 / s)   # D1 will resist
                area_r_2 = 0.5 * (1 / s) * (total_height - r_2_h0) ** 2
            elif (m1 < total_height <= t1):  # Zone 2b. Coincides with zone 2 of region 2
                area_r_1 = 0.5 * (self.b1 - self.r_1_h0) * (self.b1 - self.r_1_h0) * (1 / s) + 0.5 * (total_height - self.b1) * (self.b1 - self.r_1_h0) * (1 / s)  # D1 will resist
                # Base of D3 = m1
                area_r_2 = 0.5 * (m1 - r_2_h0) * (m1 - r_2_h0) * (1 / s) + 0.5 * (total_height - m1) * (m1 - r_2_h0) * (1 / s)  # D3 will resist
            elif (self.b2 < total_height <= m2):  # zone 4a
                area_r_1 = 0.5 * (self.b2 - self.r_1_h0) * (self.b2 - self.r_1_h0) * (1 / s) + 0.5 * (total_height - self.b2) * (self.b2 - self.r_1_h0) * (1 / s)  # D2 will resist
                area_r_2 = 0.5 * (1 / s) * (total_height - r_2_h0) ** 2  # Region 2 completely submerged upto zone 3
            else:  # Dike D1 becomes ineffective, everything submerged
                area_r_1 = 0.5 * (1 / s) * (total_height - self.r_1_h0)**2
                area_r_2 = 0.5 * (1 / s) * (total_height - r_2_h0) ** 2

            vol_f = area_r_1 + area_r_2
            c_flood = c_f * vol_f / vol_z
            scc_sum = discounted_sum_scc[year]
            c_flood_carbon = self.FloodDamageGHG(c_flood, scc_sum)

        elif next_system == (1.5, 0.75, 0.75, 0.75):  # System 32

            # D1 will resist zone 2a and 2b. D3 resist zone 2
            # D2 will resist zone 4a. D4 ineffective
            # Same as system 31 (1.5, 0.75, 0.75, 0)

            if total_height <= self.b1:
                area_r_1 = 0.5 * (1 / s) * (total_height - self.r_1_h0)**2 if total_height > self.r_1_h0 else 0
                area_r_2 = 0.5 * (1 / s) * (total_height - r_2_h0) ** 2 if total_height > r_2_h0 else 0
            elif (self.b1 < total_height <= m1):  # Zone 2a
                area_r_1 = 0.5 * (self.b1 - self.r_1_h0) * (self.b1 - self.r_1_h0) * (1 / s) + 0.5 * (total_height - self.b1) * (self.b1 - self.r_1_h0) * (1 / s)   # D1 will resist
                area_r_2 = 0.5 * (1 / s) * (total_height - r_2_h0) ** 2
            elif (m1 < total_height <= t1):  # Zone 2b. Coincides with zone 2 of region 2
                area_r_1 = 0.5 * (self.b1 - self.r_1_h0) * (self.b1 - self.r_1_h0) * (1 / s) + 0.5 * (total_height - self.b1) * (self.b1 - self.r_1_h0) * (1 / s)  # D1 will resist
                # Base of D3 = m1
                area_r_2 = 0.5 * (m1 - r_2_h0) * (m1 - r_2_h0) * (1 / s) + 0.5 * (total_height - m1) * (m1 - r_2_h0) * (1 / s)  # D3 will resist
            elif (self.b2 < total_height <= m2):  # zone 4a
                area_r_1 = 0.5 * (self.b2 - self.r_1_h0) * (self.b2 - self.r_1_h0) * (1 / s) + 0.5 * (total_height - self.b2) * (self.b2 - self.r_1_h0) * (1 / s)  # D2 will resist
                area_r_2 = 0.5 * (1 / s) * (total_height - r_2_h0) ** 2  # Region 2 completely submerged upto zone 3
            else:  # Dike D1 becomes ineffective, everything submerged
                area_r_1 = 0.5 * (1 / s) * (total_height - self.r_1_h0)**2
                area_r_2 = 0.5 * (1 / s) * (total_height - r_2_h0) ** 2

            vol_f = area_r_1 + area_r_2
            c_flood = c_f * vol_f / vol_z
            scc_sum = discounted_sum_scc[year]
            c_flood_carbon = self.FloodDamageGHG(c_flood, scc_sum)

        elif next_system == (1.5, 1.5, 0, 0):  # System 33

            # D1 will resist zone 2a
            # D2 will resist zone 4a
            # same as system 17 (0.75, 0.75, 0, 0)

            if total_height <= self.b1:
                area_r_1 = 0.5 * (1 / s) * (total_height - self.r_1_h0)**2 if total_height > self.r_1_h0 else 0
                area_r_2 = 0.5 * (1 / s) * (total_height - r_2_h0) ** 2 if total_height > r_2_h0 else 0
            elif (self.b1 < total_height <= m1):  # Zone 2a
                area_r_1 = 0.5 * (self.b1 - self.r_1_h0) * (self.b1 - self.r_1_h0) * (1 / s) + 0.5 * (total_height - self.b1) * (self.b1 - self.r_1_h0) * (1 / s)   # D1 will resist
                area_r_2 = 0.5 * (1 / s) * (total_height - r_2_h0) ** 2  # Region 2 completely submerged
            elif (self.b2 < total_height <= m2):  # zone 4a
                area_r_1 = 0.5 * (self.b2 - self.r_1_h0) * (self.b2 - self.r_1_h0) * (1 / s) + 0.5 * (total_height - self.b2) * (self.b2 - self.r_1_h0) * (1 / s)
                area_r_2 = 0.5 * (1 / s) * (total_height - r_2_h0) ** 2  # Region 2 completely submerged upto zone 3
            else:  # everything submerged
                area_r_1 = 0.5 * (1 / s) * (total_height - self.r_1_h0)**2
                area_r_2 = 0.5 * (1 / s) * (total_height - r_2_h0) ** 2

            vol_f = area_r_1 + area_r_2
            c_flood = c_f * vol_f / vol_z
            scc_sum = discounted_sum_scc[year]
            c_flood_carbon = self.FloodDamageGHG(c_flood, scc_sum)

        elif next_system == (1.5, 1.5, 0, 0.75):  # System 34

            # D1 will resist zone 2a
            # D2 will resist zone 4a and 4b. D4 will resist zone 4

            if total_height <= self.b1:
                area_r_1 = 0.5 * (1 / s) * (total_height - self.r_1_h0)**2 if total_height > self.r_1_h0 else 0
                area_r_2 = 0.5 * (1 / s) * (total_height - r_2_h0) ** 2 if total_height > r_2_h0 else 0
            elif (self.b1 < total_height <= m1):  # Zone 2a
                area_r_1 = 0.5 * (self.b1 - self.r_1_h0) * (self.b1 - self.r_1_h0) * (1 / s) + 0.5 * (total_height - self.b1) * (self.b1 - self.r_1_h0) * (1 / s)  # D1 will resist
                area_r_2 = 0.5 * (1 / s) * (total_height - r_2_h0) ** 2  # Region 2 completely submerged
            elif (self.b2 < total_height <= m2):  # zone 4a
                area_r_1 = 0.5 * (self.b2 - self.r_1_h0) * (self.b2 - self.r_1_h0) * (1 / s) + 0.5 * (total_height - self.b2) * (self.b2 - self.r_1_h0) * (1 / s)  # D2 will resist
                area_r_2 = 0.5 * (1 / s) * (total_height - r_2_h0) ** 2  # Region 2 completely submerged upto zone 3
            elif (m2 < total_height <= t2):  # Zone 4b. Coincides with zone 4 of region 2
                area_r_1 = 0.5 * (self.b2 - self.r_1_h0) * (self.b2 - self.r_1_h0) * (1 / s) + 0.5 * (total_height - self.b2) * (self.b2 - self.r_1_h0) * (1 / s)
                # Base of D4 = m2
                area_r_2 = 0.5 * (m2 - r_2_h0) * (m2 - r_2_h0) * (1 / s) + 0.5 * (total_height - m2) * (m2 - r_2_h0) * (1 / s)  # D4 will resist
            else:  # Dike D1 becomes ineffective, everything submerged
                area_r_1 = 0.5 * (1 / s) * (total_height - self.r_1_h0)**2
                area_r_2 = 0.5 * (1 / s) * (total_height - r_2_h0) ** 2

            vol_f = area_r_1 + area_r_2
            c_flood = c_f * vol_f / vol_z
            scc_sum = discounted_sum_scc[year]
            c_flood_carbon = self.FloodDamageGHG(c_flood, scc_sum)

        elif next_system == (1.5, 1.5, 0.75, 0):  # System 35

            # D1 will resist zone 2a and 2b. D3 resist zone 2
            # D2 will resist zone 4a
            # Same as system 34

            if total_height <= self.b1:
                area_r_1 = 0.5 * (1 / s) * (total_height - self.r_1_h0)**2 if total_height > self.r_1_h0 else 0
                area_r_2 = 0.5 * (1 / s) * (total_height - r_2_h0) ** 2 if total_height > r_2_h0 else 0
            elif (self.b1 < total_height <= m1):  # Zone 2a
                area_r_1 = 0.5 * (self.b1 - self.r_1_h0) * (self.b1 - self.r_1_h0) * (1 / s) + 0.5 * (total_height - self.b1) * (self.b1 - self.r_1_h0) * (1 / s)  # D1 will resist
                area_r_2 = 0.5 * (1 / s) * (total_height - r_2_h0) ** 2  # Region 2 completely submerged
            elif (self.b2 < total_height <= m2):  # zone 4a
                area_r_1 = 0.5 * (self.b2 - self.r_1_h0) * (self.b2 - self.r_1_h0) * (1 / s) + 0.5 * (total_height - self.b2) * (self.b2 - self.r_1_h0) * (1 / s)  # D2 will resist
                area_r_2 = 0.5 * (1 / s) * (total_height - r_2_h0) ** 2  # Region 2 completely submerged upto zone 3
            elif (m2 < total_height <= t2):  # Zone 4b. Coincides with zone 4 of region 2
                area_r_1 = 0.5 * (self.b2 - self.r_1_h0) * (self.b2 - self.r_1_h0) * (1 / s) + 0.5 * (total_height - self.b2) * (self.b2 - self.r_1_h0) * (1 / s)
                # Base of D4 = m2
                area_r_2 = 0.5 * (m2 - r_2_h0) * (m2 - r_2_h0) * (1 / s) + 0.5 * (total_height - m2) * (m2 - r_2_h0) * (1 / s)  # D4 will resist
            else:  # Dike D1 becomes ineffective, everything submerged
                area_r_1 = 0.5 * (1 / s) * (total_height - self.r_1_h0)**2
                area_r_2 = 0.5 * (1 / s) * (total_height - r_2_h0) ** 2

            vol_f = area_r_1 + area_r_2
            c_flood = c_f * vol_f / vol_z
            scc_sum = discounted_sum_scc[year]
            c_flood_carbon = self.FloodDamageGHG(c_flood, scc_sum)

        elif next_system == (1.5, 1.5, 0.75, 0.75):  # System 36

            # D1 will resist zone 2a and 2b. D3 will resist zone 2
            # D2 will resist zone 4a and 4b. D4 will resist zone 4

            if total_height <= self.b1:  # Zone 1
                area_r_1 = 0.5 * (1 / s) * (total_height - self.r_1_h0)**2 if total_height > self.r_1_h0 else 0
                area_r_2 = 0.5 * (1 / s) * (total_height - r_2_h0) ** 2 if total_height > r_2_h0 else 0
            elif (self.b1 < total_height <= m1):  # Zone 2a
                area_r_1 = 0.5 * (self.b1 - self.r_1_h0) * (self.b1 - self.r_1_h0) * (1 / s) + 0.5 * (total_height - self.b1) * (self.b1 - self.r_1_h0) * (1 / s)  # D1 will resist
                area_r_2 = 0.5 * (1 / s) * (total_height - r_2_h0) ** 2  # Region 2 completely submerged
            elif (m1 < total_height <= t1):  # Zone 2b. Coincides with zone 2 of region 2
                area_r_1 = 0.5 * (self.b1 - self.r_1_h0) * (self.b1 - self.r_1_h0) * (1 / s) + 0.5 * (total_height - self.b1) * (self.b1 - self.r_1_h0) * (1 / s)  # D1 will resist
                # Base of D3 = m1
                area_r_2 = 0.5 * (m1 - r_2_h0) * (m1 - r_2_h0) * (1 / s) + 0.5 * (total_height - m1) * (m1 - r_2_h0) * (1 / s)  # D3 will resist
            elif (self.b2 < total_height <= m2):  # zone 4a
                area_r_1 = 0.5 * (self.b2 - self.r_1_h0) * (self.b2 - self.r_1_h0) * (1 / s) + 0.5 * (total_height - self.b2) * (self.b2 - self.r_1_h0) * (1 / s)  # D2 will resist
                area_r_2 = 0.5 * (1 / s) * (total_height - r_2_h0) ** 2  # Region 2 completely submerged upto zone 3
            elif (m2 < total_height <= t2):  # Zone 4b. Coincides with zone 4 of region 2
                area_r_1 = 0.5 * (self.b2 - self.r_1_h0) * (self.b2 - self.r_1_h0) * (1 / s) + 0.5 * (total_height - self.b2) * (self.b2 - self.r_1_h0) * (1 / s)
                # Base of D4 = m2
                area_r_2 = 0.5 * (m2 - r_2_h0) * (m2 - r_2_h0) * (1 / s) + 0.5 * (total_height - m2) * (m2 - r_2_h0) * (1 / s)  # D4 will resist
            else:  # Dike D1 becomes ineffective, everything submerged
                area_r_1 = 0.5 * (1 / s) * (total_height - self.r_1_h0)**2
                area_r_2 = 0.5 * (1 / s) * (total_height - r_2_h0) ** 2


            vol_f = area_r_1 + area_r_2
            c_flood = c_f * vol_f / vol_z
            scc_sum = discounted_sum_scc[year]
            c_flood_carbon = self.FloodDamageGHG(c_flood, scc_sum)

        else:
            print(f'invalid system')

        # print(f"Cost of flooding: {c_flood}")
        # print(f"Corresponding carbon: {c_flood_carbon}")

        flood_damage = gamma*(c_flood + c_flood_carbon)

        # calculate construction and carbon costs based on the action
        action_cost, action_carbon_cost = self.calculate_construction_costs(action, year, discounted_sum_scc)
        construction = action_cost + action_carbon_cost

        # construction
        # if action == 0:#do nothing
        #     action_cost = 0
        #     scc_sum = discounted_sum_scc[year]
        #     action_carbon_cost = self.CarbonConstruction(0, scc_sum)
        # elif action == 1:#construct dike 1
        #     action_cost = -1.38e+04
        #     scc_sum = discounted_sum_scc[year]
        #     action_carbon_cost = self.CarbonConstruction(1.5, scc_sum)
        # elif action == 2: #construct dike 2
        #     action_cost = -1.38e+04
        #     scc_sum = discounted_sum_scc[year]
        #     action_carbon_cost = self.CarbonConstruction(1.5, scc_sum)
        # elif action == 3: #construct both dikes
        #     action_cost = -2*-1.38e+04
        #     scc_sum = discounted_sum_scc[year]
        #     action_carbon_cost = 2*self.CarbonConstruction(1.5, scc_sum)
        # else:
        #     print(f'invalid action')

        # if old_system == 0:
        #     main_cost = 0
        #     scc_sum = discounted_sum_scc[year]
        #     main_carbon_cost = 0
        # elif old_system == 1:
        #     main_cost = -100
        #     scc_sum = discounted_sum_scc[year]
        #     main_carbon_cost = self.AnnualMaintain(scc_sum)
        # elif old_system == 2:
        #     main_cost = -100
        #     scc_sum = discounted_sum_scc[year]
        #     main_carbon_cost = self.AnnualMaintain(scc_sum)
        # elif old_system == 3:
        #     main_cost = -2*100
        #     scc_sum = discounted_sum_scc[year]
        #     main_carbon_cost = self.AnnualMaintain(scc_sum)
        # else:
        #     print(f'invalid system')

        main_cost, main_carbon_cost = self.calculate_maintenance_costs(old_system, year, discounted_sum_scc)
        maintenance = main_cost + main_carbon_cost

        total_cost = flood_damage + construction + maintenance
        return total_cost

    def FloodDamageGHG(self, flood_damage, scc_sum):

        Flood_Cost_2007 = (flood_damage / 1000000) * 0.77
        FD_GHGI = 445.1
        FD_GHG = 445.1 * Flood_Cost_2007
        FD_GHG_C = FD_GHG * scc_sum

        return FD_GHG_C

    def CarbonConstruction(self, dike_height, scc_sum):

        Dike_Cost_2007 = (8000 * dike_height) * 0.89
        Dike_Cost_Million = Dike_Cost_2007 / 1000000
        DCM_GHG = Dike_Cost_Million * 243       # Amount of GHG released for that cost
        DCM_GHG_C = DCM_GHG * scc_sum           # Carbon cost for that amount of GHG

        return DCM_GHG_C

    def AnnualMaintain(self, scc_sum):

        Dike_Maintenance_Cost_2007 = 100 * 0.89
        Dike_Maintenance_Cost_2007_Million = Dike_Maintenance_Cost_2007 / 1000000
        AM_GHG = Dike_Maintenance_Cost_2007_Million * 385
        AM_GHG_C = AM_GHG * scc_sum

        return AM_GHG_C


    def calculate_construction_costs(self, action, year, discounted_sum_scc):
        """
        Calculate the construction costs and carbon costs based on the action taken.
        - action: tuple of four elements (actions at D1, D2, D3, and D4)
        - year: the current year index for accessing the discounted sum of social cost of carbon
        - discounted_sum_scc: array of discounted social cost of carbon values
        """
        # Costs constants
        cost_per_meter = -1.38e+04  # Cost for 1.5 meters dike
        base_cost = cost_per_meter / 1.5  # Cost per meter

        # Initialize total costs
        total_construction_cost = 0
        total_carbon_cost = 0

        action = self.action_space[action]
        # Iterate through each dike in the action tuple
        for dike_index, dike_action in enumerate(action):
            if dike_action == 0:
                continue  # Do nothing action has no cost

            if dike_action == 'elevate':
                # Elevate from 0.75 to 1.5 meters, cost calculation for 0.75 additional meters
                height_to_construct = 0.75
            elif dike_action == 1.5:
                # Construct directly to 1.5 meters
                height_to_construct = 1.5
            elif dike_action == 0.75:
                # Construct to 0.75 meters
                height_to_construct = 0.75
            else:
                print(f'invalid action')

            # Calculate construction cost for the current dike action
            current_construction_cost = base_cost * height_to_construct
            total_construction_cost += current_construction_cost

            # Calculate carbon cost associated with the construction
            scc_sum = discounted_sum_scc[year]
            current_carbon_cost = self.CarbonConstruction(height_to_construct, scc_sum)
            total_carbon_cost += current_carbon_cost

        return total_construction_cost, total_carbon_cost

    def calculate_maintenance_costs(self, old_system, year, discounted_sum_scc):
        """
        Calculate the maintenance cost and carbon cost based on the old system configuration.
        """
        # Base maintenance cost for dike of height (1.5m)
        base_maintenance_cost = -100
        total_maintenance_cost = 0
        total_carbon_cost = 0

        # Calculate maintenance cost for each dike based on its height
        for dike_height in old_system:
            if dike_height == 1.5:
                # Full maintenance cost
                maintenance_cost = base_maintenance_cost
                carbon_cost = self.AnnualMaintain(discounted_sum_scc[year])
            elif dike_height == 0.75:
                # Half maintenance cost since dike is only at mid-height
                maintenance_cost = base_maintenance_cost / 2
                carbon_cost = self.AnnualMaintain(discounted_sum_scc[year]) / 2
            else:
                # No cost if no dike or dike is at 0m
                maintenance_cost = 0
                carbon_cost = 0

            total_maintenance_cost += maintenance_cost
            total_carbon_cost += carbon_cost

        return total_maintenance_cost, total_carbon_cost

