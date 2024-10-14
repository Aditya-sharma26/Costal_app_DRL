import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from network import ActorNetwork, CriticNetwork
from utils import plot_learning_curve, make_env
from values_slr import slr
from values_surge import surge

# from environment import Environment
from torch.optim import Adam
from components import components
from values_slr import slr
from values_surge import surge

class PPO:
    def __init__(self, env):
        # define hyperparameters
        self._init_hyperparameters()
        # extract env info
        self.env = env
        self.in_dims = int(np.prod(self.env.observation_space.n) + 5)
        self.actor_dims = int(np.prod(self.env.action_space.n))
        # initialize actor and critic
        self.actor = ActorNetwork(self.in_dims, self.actor_dims)
        self.critic = CriticNetwork(self.in_dims, 1)
        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)
        #self.cov_var = torch.full(size=(self.actor_dims,), fill_value=self.cov)
        #self.cov_mat = torch.diag(self.cov_var)


    def _init_hyperparameters(self):
        # Default values for hyperparameters, will need to change later.
        self.episodes_per_batch = 5  # timesteps per batch
        self.train_epi = 5
        self.gamma = 0.95
        #self.cov = 0.5
        self.n_updates_per_iter = 5
        self.clip = 0.2
        self.lr = 0.005
        #self.max_timesteps_per_episode = 1600  # timesteps per episode

    def state_vector(self, env, state, time):
        print(f'state: {state}')
        s = np.zeros(int(np.prod(env.observation_space.shape)))
        print(s.shape)
        s[int(state)] = 1
        print(s[int(state)])
        state_combined = np.concatenate((s, np.array([time])), axis=None)
        print(state_combined.shape)
        state_combined = np.concatenate((state_combined, np.array(env.system).T), axis=None)
        print(state_combined.shape)
        return state_combined

    def get_action(self, obs):
        logits = self.actor(obs)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.detach().numpy(), log_prob.detach()

    def compute_rtgs(self, batch_rews):
        batch_rtgs = []
        for ep_rews in reversed(batch_rews):
            discounted_rew = 0
            for rew in reversed(ep_rews):
                discounted_rew = rew + self.gamma*discounted_rew
                batch_rtgs.insert(0, discounted_rew)
        batch_rtgs = torch.as_tensor(batch_rtgs, dtype=torch.float)
        return batch_rtgs

    def rollout(self):
        #batch data
        batch_obs =[]
        batch_acts = []
        batch_log_probs = []
        batch_rews = []
        batch_rtgs = []
        #batch_lens = []

        for epi in range(self.episodes_per_batch):
            ep_rews = []
            year = 1
            obs = self.env.reset()
            done = False

            while not done:
                year += 1
                state = self.state_vector(env, obs, year)
                print(state)
                batch_obs.append(state)
                action, log_prob = self.get_action(state)
                print(f'action taken: {action}')
                new_obs, reward, done, _ = self.env.step(obs, action)
                batch_acts.append(np.array([action]))
                batch_log_probs.append(np.array([log_prob]))
                #np.append(ep_rews, reward)
                ep_rews.append(reward)
                obs = new_obs
                if done:
                    break

            #np.append(batch_rews, ep_rews)
            batch_rews.append(ep_rews)
            # batch_rews.append([ep_rews])
        batch_obs = torch.as_tensor(batch_obs, dtype=torch.float)
        batch_acts = torch.as_tensor(batch_acts, dtype=torch.float)
        batch_log_probs = torch.as_tensor(batch_log_probs, dtype=torch.float)
        batch_rtgs = self.compute_rtgs(batch_rews)
        return batch_obs, batch_acts, batch_log_probs, batch_rtgs

    def evaluate(self, batch_obs, batch_acts):
        v = self.critic(batch_obs)
        v = v.squeeze()

        logits = self.actor(batch_obs)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(batch_acts)
        return v, log_probs

    def learn(self, train_episodes):
        for epi in range(train_episodes):
            batch_obs, batch_acts, batch_log_probs, batch_rtgs = self.rollout()
            # calculate advantage functions
            batch_value, _ = self.evaluate(batch_obs, batch_acts)
            a_batch = batch_rtgs - batch_value.detach()
            #normalize advantages for better learning
            a_batch = (a_batch - a_batch.mean())/(a_batch.std() + 1e-10) #1e-10 added just to avoid dividing by 0
            for _ in range(self.n_updates_per_iter):
                _, curr_log_probs = self.evaluate(batch_obs, batch_acts)
                ratios = torch.exp(curr_log_probs - batch_log_probs)
                surr1 = ratios*a_batch
                surr2 = torch.clamp(ratios, 1-self.clip, 1+self.clip)*a_batch
                actor_loss = (-torch.min(surr1, surr2)).mean()
                self.actor_optim.zero_grad()
                actor_loss.backward(retain_graph=True)
                self.actor_optim.step()
                v, curr_log_probs = self.evaluate(batch_obs, batch_acts)
                critic_loss = nn.MSELoss()(v, batch_rtgs)
                self.critic.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()



env = Environment()
model = PPO(env)
model.learn(10000)

# testing
print(f'testing start')
observation = env.reset()
test_episodes = 100

for episode in range(test_episodes):
    observation = env.reset()
    done = False
    rewards = 0
    steps = 0
    while not done:
        steps += 1
        [slr_state, surge_state] = components(observation)
        slr_value = slr(slr_state)
        surge_value = surge(surge_state)
        print(f'combined state: {observation}, slr: {slr_value} cm, surge: {surge_value} cm')
        action = model.get_action(model.state_vector(env, observation, steps))
        print(f'action taken by agent: {action}')
        print(f'system: {env.system}')
        print(f'year: {env.year}')
        new_observation, reward, done, info = env.step(observation, action)
        rewards += reward
        observation = new_observation

        if done:
            print('epi {} ends with total rewards {}'.format(episode, rewards))
            break















