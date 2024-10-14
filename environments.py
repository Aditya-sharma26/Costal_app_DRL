from gym import Env
from gym.spaces import Box, Discrete
import random
import numpy as np
import mat73
from values_slr import slr
from values_surge import surge

class Environment(Env):
    def __init__(self, env_name, climate_model):
        # define your environment
        # action space, observation space
        self.year = 0
        self.n_actions = 4
        # self.action_space = np.array([0, 1, 2, 3])
        self.action_space = Discrete(self.n_actions)
        self.horizon = 39
        self.discount_reward = 0.97
        self.n_states_slr = 77
        self.n_states_surge = 72
        # self.n_states_total = self.n_states_surge + self.n_states_slr
        self.n_states_total = self.n_states_slr
        # self.state_space = []
        # for i in range(0, self.n_states):
        #     self.state_space.append(i)
        # self.state_space = np.array(self.state_space)
        # self.observation_space = np.array(self.state_space)

        self.observation_space = Discrete(self.n_states_total)
        self.initial_state = np.array([4, 1])
        self.state = self.initial_state
        self.initial_system = np.array([1, 0, 0, 0])
        self.system = self.initial_system
        self.previous_system = self.initial_system
        self.ssp = climate_model
        self.model = env_name
        self.system_trans = {0: np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]),
                             1: np.array([[0, 1, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 0, 1]]),
                             2: np.array([[0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]]),
                             3: np.array([[0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1]])}

        self.trans = self.get_transition(ssp=self.ssp)
        self.trans_slr = self.trans[0]
        self.trans_surge = self.trans[1]
        self.rewards_sys = self.get_rewards(env_model=self.model)
        self.terminal_rewards = self.get_terminal_reward(env_model=self.model)

    def step(self, action):
        # take some action


        print(self.year)
        # next state according to year and action taken from current state
        # independent of system and action: just depends on the year and the current state
        trans_slr = self.trans_slr[self.year]
        trans_slr_state = trans_slr[int(self.state[0])]
        next_slr = np.random.choice(np.array(range(0, self.n_states_slr)), 1, p=trans_slr_state)
        print(next_slr)
        trans_surge_state = self.trans_surge[0][int(self.state[1])]
        next_surge = np.random.choice(np.array(range(0, self.n_states_surge)), 1, p=trans_surge_state)
        print(next_surge)
        next_state = np.array([next_slr, next_surge])
        # reward according to the action taken, next system, and current state
        prev_system = np.argmax(self.system)
        self.previous_system = self.system
        system_trans_act = self.system_trans[int(action)]
        self.system = self.system.dot(system_trans_act)
        next_system = np.argmax(self.system)
        # get rewards model according to the environment set up- dikes or green resistance, etc.
        # rewards_sys = self.get_rewards(env_model=self.model)
        reward_ = self.immediate_cost(next_state, action, prev_system, next_system, self.year)
        # rewards_act = self.rewards_sys[int(action)]
        # rewards_act_sys = rewards_act[int(next_system)]
        # rewards_act_sys_t = rewards_act_sys[self.year]
        # next_state_combined = self.get_combined(next_state)
        # reward = rewards_act_sys_t[next_state_combined]
        print(reward_)
        # discounting and scaling the rewards
        # reward = (self.discount_reward ** self.year) * reward_
        reward = reward_
        reward = reward/1e6
        print(reward)

        next_state_combined = self.get_combined(next_state)
        if self.year > self.horizon:
            done = True
            terminal_reward_act = self.terminal_rewards[int(action)]
            terminal_reward_act_sys = terminal_reward_act[int(next_system)]
            terminal_reward = terminal_reward_act_sys[next_state_combined]
            print(f'terminal reward')
            reward = terminal_reward[0]/1e6
            print(reward)
        else:
            done = False

        info = {}
        self.year += 1
        print(reward)
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
        # state_combined = np.concatenate(((np.array([time / horizon])), slr, surge), axis=None)
        state_combined = np.concatenate(((np.array([time / horizon])), slr), axis=None)
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
            transitions_slr = mat73.loadmat('t_slr_245.mat')
            trans_slr = np.array(transitions_slr['t_slr_avg'])
            transitions_surge = mat73.loadmat('t_surge.mat')
            trans_surge = np.array(transitions_surge['t_surge'])
        elif ssp == '585':
            transitions_slr = mat73.loadmat('t_slr_245.mat')
            trans_slr = np.array(transitions_slr['t_slr_avg'])
            transitions_surge = mat73.loadmat('t_surge.mat')
            trans_surge = np.array(transitions_surge['t_surge'])
        else:
            print(f'not valid ssp')

        return trans_slr, trans_surge

    def get_rewards(self, env_model):
        if env_model == 'TwoDikes':
            rewards = mat73.loadmat('rewards_two_dikes_scc_tc_245.mat')
            self.rewards_sys = np.array(rewards['rewards'])
        elif env_model == 'GreenDike':
            rewards = mat73.loadmat('rewards_grd2_finalscc_07.mat')
            self.rewards_sys = np.array(rewards['rewards'])
        else:
            print(f'not valid environment model')

        return self.rewards_sys

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
        c_f = -0.07 * 15.3 * 1e6
        gamma = 0.97
        s = 0.0085
        vol_z = 0.5 * 8.50 * (1 / s) * 8.50
        slr_value = slr(next_state[0])
        surge_value = surge(next_state[1])
        # total_height = slr_value + surge_value
        total_height = slr_value*10
        discounted_sum_scc_ = mat73.loadmat('discounted_sum_scc_7_3.mat')
        discounted_sum_scc = np.array(discounted_sum_scc_['discounted_sum_scc'])
        b1 = 0.2
        t1 = 1.7
        b2 = 1.9
        t2 = 3.4
        # flood cost
        if next_system == 0:
            vol_f = 0.5 * (1 / s) * total_height**2
            c_flood = c_f * vol_f / vol_z
            scc_sum = discounted_sum_scc[year]
            c_flood_carbon = self.FloodDamageGHG(c_flood, scc_sum)
        elif next_system == 1:
            if (total_height > b1) and (total_height <= t1):
                area1 = 0.5 * b1 * b1 * (1 / s) + (total_height - b1) * b1 * (1 / s)
            else:
                area1 = 0.5 * (1 / s) * total_height**2
            vol_f = (area1 - 0)
            c_flood = c_f * vol_f/ vol_z
            scc_sum = discounted_sum_scc[year]
            c_flood_carbon = self.FloodDamageGHG(c_flood, scc_sum)
        elif next_system == 2:
            if (total_height > b2) and (total_height <= t2):
                area1 = 0.5 * b2 * b2* (1 / s) + (total_height - b2) * b2 * (1 / s)
            else:
                area1 = 0.5 * (1 / s) * total_height**2
            vol_f = (area1 - 0)
            c_flood = c_f * vol_f/ vol_z
            scc_sum = discounted_sum_scc[year]
            c_flood_carbon = self.FloodDamageGHG(c_flood, scc_sum)
        elif next_system == 3:
            if total_height <= b1:
                area1 = 0.5 * (1 / s) * total_height**2
            elif total_height > b1 and total_height <= t1:
                area1 = 0.5 * (1/s) * b1**2 + (total_height - b1) * b1 * (1 / s)
            elif total_height > t1 and total_height <= b2:
                area1 = 0.5 * (1/s) * total_height**2
            elif total_height > b2 and total_height <= t2:
                area1 = 0.5 * (1/s) * b2**2 + (total_height - b2) * b2 * (1 / s)
            else:
                area1 = 0.5 * (1 / s) * total_height**2
            vol_f = (area1 - 0)
            c_flood = c_f * vol_f / vol_z
            scc_sum = discounted_sum_scc[year]
            c_flood_carbon = self.FloodDamageGHG(c_flood, scc_sum)
        else:
            print(f'invalid system')

        flood_damage = gamma*(c_flood + c_flood_carbon)

        # construction
        if action == 0:#do nothing
            action_cost = 0
            scc_sum = discounted_sum_scc[year]
            action_carbon_cost = self.CarbonConstruction(1.5, scc_sum)
        elif action == 1:#construct dike 1
            action_cost = -1.38e+04
            scc_sum = discounted_sum_scc[year]
            action_carbon_cost = self.CarbonConstruction(1.5, scc_sum)
        elif action == 2: #construct dike 2
            action_cost = -1.38e+04
            scc_sum = discounted_sum_scc[year]
            action_carbon_cost = self.CarbonConstruction(1.5, scc_sum)
        elif action == 3: #construct both dikes
            action_cost = -2*-1.38e+04
            scc_sum = discounted_sum_scc[year]
            action_carbon_cost = self.CarbonConstruction(1.5, scc_sum)
        else:
            print(f'invalid action')

        construction = action_cost + action_carbon_cost

        if old_system == 0:
            main_cost = 0
            scc_sum = discounted_sum_scc[year]
            main_carbon_cost = self.AnnualMaintain(scc_sum)
        elif old_system == 1:
            main_cost = -100
            scc_sum = discounted_sum_scc[year]
            main_carbon_cost = self.AnnualMaintain(scc_sum)
        elif old_system == 2:
            main_cost = -100
            scc_sum = discounted_sum_scc[year]
            main_carbon_cost = self.AnnualMaintain(scc_sum)
        elif old_system == 3:
            main_cost = -2*100
            scc_sum = discounted_sum_scc[year]
            main_carbon_cost = self.AnnualMaintain(scc_sum)
        else:
            print(f'invalid system')

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
        DCM_GHG = Dike_Cost_Million * 243
        DCM_GHG_C = DCM_GHG * scc_sum

        return DCM_GHG_C

    def AnnualMaintain(self, scc_sum):

        Dike_Maintenance_Cost_2007 = 100 * 0.89
        Dike_Maintenance_Cost_2007_Million = Dike_Maintenance_Cost_2007 / 1000000
        AM_GHG = Dike_Maintenance_Cost_2007_Million * 385
        AM_GHG_C = AM_GHG * scc_sum

        return AM_GHG_C

