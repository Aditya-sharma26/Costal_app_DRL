################################### Training ###################################

import copy
import random
import os, argparse
import glob
import time
from datetime import datetime

import numpy as np
import gym
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
# import pybullet_envs
from collections import deque
import matplotlib.pyplot as plt

from PPO_main_actor_critic import PPO
# from utils import plot_learning_curve, make_env
from utils2 import plot_learning_curve, make_env

from values_slr import slr
from values_surge import surge

####### initialize environment hyperparameters ######

parser = argparse.ArgumentParser(
        description='Deep Q Learning: From Paper to Code')
parser.add_argument('-gpu', type=str, default='0', help='GPU: 0 or 1')
parser.add_argument('-climate_ssp', type=str, default='245',
                        help='119/245/585')
parser.add_argument('-env', type=str, default='TwoDikes',
                        help='TwoDikes/GreenDike/others')
args = parser.parse_args()

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

# env_name = "CartPole-v0"
has_continuous_action_space = False
episodes = 20000
max_ep_len = 40  # max timesteps in one episode
max_training_timesteps = int(1e6)  # break training loop if timeteps > max_training_timesteps

print_freq = max_ep_len * 4  # print avg reward in the interval (in num timesteps)
log_freq = max_ep_len * 2  # log avg reward in the interval (in num timesteps)
save_model_freq = int(2e4)  # save model frequency (in num timesteps)

action_std = None

#####################################################

## Note : print/log frequencies should be > than max_ep_len

################ PPO hyperparameters ################

update_timestep = max_ep_len * 100  # update policy every n timesteps
K_epochs = 40  # update policy for K epochs
eps_clip = 0.2  # clip parameter for PPO
gamma = 0.90  # discount factor

lr_actor = 0.0003  # learning rate for actor network
lr_critic = 0.001  # learning rate for critic network

random_seed = 0  # set random seed if required (0 = no random seed)

#####################################################

# print("training environment name : " + env_name)
# env = gym.make(env_name)
env = make_env(env_name=args.env, climate_model=args.climate_ssp)

# state space dimension
state_dim = env.observation_space.n

# action space dimension
if has_continuous_action_space:
    action_dim = env.action_space.shape[0]
else:
    action_dim = env.action_space.n

############# print all hyperparameters #############

print("--------------------------------------------------------------------------------------------")

print("max training timesteps : ", max_training_timesteps)
print("max timesteps per episode : ", max_ep_len)

print("model saving frequency : " + str(save_model_freq) + " timesteps")
print("log frequency : " + str(log_freq) + " timesteps")
print("printing average reward over episodes in last : " + str(print_freq) + " timesteps")

print("--------------------------------------------------------------------------------------------")

print("state space dimension : ", state_dim + 5)
print("action space dimension : ", action_dim)

print("--------------------------------------------------------------------------------------------")

if has_continuous_action_space:
    print("Initializing a continuous action space policy")
    print("--------------------------------------------------------------------------------------------")
    print("starting std of action distribution : ", action_std)
    print("decay rate of std of action distribution : ", action_std_decay_rate)
    print("minimum std of action distribution : ", min_action_std)
    print("decay frequency of std of action distribution : " + str(action_std_decay_freq) + " timesteps")

else:
    print("Initializing a discrete action space policy")

print("--------------------------------------------------------------------------------------------")

print("PPO update frequency : " + str(update_timestep) + " timesteps")
print("PPO K epochs : ", K_epochs)
print("PPO epsilon clip : ", eps_clip)
print("discount factor (gamma) : ", gamma)

print("--------------------------------------------------------------------------------------------")

print("optimizer learning rate actor : ", lr_actor)
print("optimizer learning rate critic : ", lr_critic)

if random_seed:
    print("--------------------------------------------------------------------------------------------")
    print("setting random seed to ", random_seed)
    torch.manual_seed(random_seed)
    env.seed(random_seed)
    np.random.seed(random_seed)

#####################################################

print("============================================================================================")

################# training procedure ################

# initialize a PPO agent
ppo_agent = PPO(state_dim + 5, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space,
                action_std)

# track total training time
start_time = datetime.now().replace(microsecond=0)
print("Started training at (GMT) : ", start_time)

# printing and logging variables
print_running_reward = 0
print_running_episodes = 0

log_running_reward = 0
log_running_episodes = 0

time_step = 0
i_episode = 0
reward_window = deque(maxlen=100)
avg_rew = []

# training loop
num_episodes = 0
while time_step <= max_training_timesteps and num_episodes < episodes:
    year = 0
    observation = env.reset()
    # state = state[0]
    done = False
    current_ep_reward = 0
    num_episodes += 1
    reward_sum = 0
    # for t in range(1, max_ep_len + 1):
    while not done:
        year += 1
        slr_value = slr(observation[0])
        surge_value = surge(observation[1])
        print(f'year: {year}; combined state: {observation}, slr: {slr_value} cm, surge: {surge_value} cm')
        state = env.get_state_vector(observation, time=year)
        # select action with policy
        action = ppo_agent.select_action(env, state)
        print(f'action selected: {action}')
        state, reward, done, _ = env.step(action)
        print(reward)
        observation = state
        reward_sum += reward
        # saving reward and is_terminals
        ppo_agent.buffer.rewards.append(reward)
        ppo_agent.buffer.is_terminals.append(done)

        time_step += 1
        current_ep_reward += reward

        # update PPO agent
        if time_step % update_timestep == 0:
            ppo_agent.update()

        # if continuous action space; then decay action std of ouput action distribution
        if has_continuous_action_space and time_step % action_std_decay_freq == 0:
            ppo_agent.decay_action_std(action_std_decay_rate, min_action_std)

        if done:
            if num_episodes > 0 and num_episodes % 50 == 0:
                print('Training Episode %s reward %s' % (num_episodes, reward_sum))
            reward_window.append(reward_sum)
            avg_rew.append(np.average(list(reward_window)))
            break

    print_running_reward += current_ep_reward
    print_running_episodes += 1

    log_running_reward += current_ep_reward
    log_running_episodes += 1

    i_episode += 1

# env.close()

# print total training time
print("============================================================================================")
end_time = datetime.now().replace(microsecond=0)
print("Started training at (GMT) : ", start_time)
print("Finished training at (GMT) : ", end_time)
print("Total training time  : ", end_time - start_time)
print("============================================================================================")


#evaluation/ testing
num_episodes = 0
test_episodes = 100
test_reward_window = deque(maxlen=100)
test_avg_rew = []

while num_episodes < test_episodes:
    year = 0
    observation = env.reset()
    # state = state[0]

    current_ep_reward = 0
    num_episodes += 1
    reward_sum = 0
    for t in range(1, max_ep_len + 1):
        year += 1
        slr_value = slr(observation[0])
        surge_value = surge(observation[1])
        print(f'year: {year}; combined state: {observation}, slr: {slr_value} cm, surge: {surge_value} cm')
        state = env.get_state_vector(observation, time=year)
        # select action with policy
        action = ppo_agent.select_action(env, state)
        print(f'action selected: {action}')
        state, reward, done, _ = env.step(action)
        observation = state
        reward_sum += reward

        time_step += 1
        current_ep_reward += reward

        if done:
            if num_episodes > 0 and num_episodes % 50 == 0:
                print('Testing Episode %s reward %s' % (num_episodes, reward_sum))
            test_reward_window.append(reward_sum)
            test_avg_rew.append(np.average(list(test_reward_window)))
            break

    print_running_reward += current_ep_reward
    print_running_episodes += 1

    log_running_reward += current_ep_reward
    log_running_episodes += 1

    i_episode += 1

# env.close()

lb_frtdp = (-(861912/1e6)*np.ones(len(avg_rew))).tolist()
plt.plot(avg_rew, label='PPO')
plt.plot(lb_frtdp, label='FRTDP lower bound')
plt.xlabel("Episodes")
plt.ylabel("Mean Reward")
plt.legend()
plt.show()


lb_frtdp = (-(861912/1e6)*np.ones(len(avg_rew))).tolist()
plt.plot(avg_rew, label='PPO')
plt.plot(test_avg_rew, label='PPO evaluation')
plt.plot(lb_frtdp, label='FRTDP lower bound')
plt.xlabel("Episodes")
plt.ylabel("Mean Reward")
plt.legend()
plt.show()