"""
the environment in which the RL agent takes actions

the environment has two actions- two dikes
the environment should have the following features -
1. reset method which brings the environment to the initial state after each episode ends
2. action space to sample from
3. step function which gives the next state, reward, done based on current state and action taken

"""
import numpy as np
import mat73

# model ssp119
transitions = mat73.loadmat('t_total_norm_100_ssp245.mat')
trans = np.array(transitions['t_total_norm'])

# model 2 dikes
rewards = mat73.loadmat('rewards_two_dikes_finalscc_07.mat')
rewards_sys = np.array(rewards['rewards'])

system_trans = {}
system_trans[0] = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
system_trans[1] = np.array([[0, 1, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 0, 1]])
system_trans[2] = np.array([[0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]])
system_trans[3] = np.array([[0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1]])

class Environment:

    def __init__(self):
        self.year = 0
        self.n_actions = 4
        self.action_space = np.array([0, 1, 2, 3])
        self.horizon = 39
        self.n_states = 72 * 77
        self.state_space = []
        for i in range(0, self.n_states):
            self.state_space.append(i)
        self.state_space = np.array(self.state_space)
        self.observation_space = np.array(self.state_space)
        self.initial_state = 290 - 1
        self.state = self.initial_state
        self.initial_system = np.array([1, 0, 0, 0])
        self.system = self.initial_system

    def step(self, state, action):
        self.year += 1
        self.state = state
        # next state according to year and action taken from current state
        trans_act = trans[self.year]
        trans_state_act = trans_act[int(self.state)]
        next_state = np.random.choice(self.observation_space, 1, p=trans_state_act)

        # reward according to the action taken, next system, and current state
        current_system = np.argmax(self.system)
        system_trans_act = system_trans[int(action)]
        self.system = self.system.dot(system_trans_act)
        next_system = np.argmax(self.system)

        rewards_act = rewards_sys[int(action)]
        rewards_act_sys = rewards_act[int(next_system)]
        rewards_act_sys_t = rewards_act_sys[self.year]
        reward = rewards_act_sys_t[next_state]

        # penalties: do not work!!!
        # if current_system == 1:
        #     if action == 1 or action == 3:
        #         reward = np.array([-10e20])
        # elif current_system == 2:
        #     if action == 2 or action == 3:
        #         reward = np.array([-10e20])
        # elif current_system == 3:
        #     if action == 1 or action == 2 or action == 3:
        #         reward = np.array([-10e20])

        # check if episode ends

        if self.year >= self.horizon:
            done = True
        else:
            done = False

        info = {}

        return next_state, reward, done, info

    def reset(self):
        self.year = 0
        self.state = self.initial_state
        self.system = self.initial_system
        return self.state
