import numpy as np
import torch as T
from deep_q_network import DeepQNetwork, DuelingDeepQNetwork
# from replay_memory import ReplayBuffer
from per import ReplayBuffer


class Agent():
    def __init__(self, gamma, epsilon, lr, n_actions, input_dims,
                 mem_size, batch_size, eps_min=0.01, eps_dec=1e-5,
                 replace=80, algo=None, env_name=None, chkpt_dir='tmp/dqn'):
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.eps_max = 1.0
        self.eps_dec_counter = 0
        self.action_space = [i for i in range(n_actions)]
        self.learn_step_counter = 0
        self.batch_size = batch_size
        self.replace_target_cnt = replace
        # print(self.replace_target_cnt)
        self.algo = algo
        self.env_name = env_name
        self.chkpt_dir = chkpt_dir

        # self.memory = ReplayBuffer(mem_size, input_dims, n_actions)
        self.memory = ReplayBuffer(2 ** 20, 0.6, input_dims)

    def store_transition(self, state, action, reward, state_, done):
        self.memory.add(state, action, reward, state_, done)

    def choose_action(self, env, observation):
        raise NotImplementedError

    def replace_target_network(self):
        if self.learn_step_counter % self.replace_target_cnt == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())

    def decrement_epsilon(self):
        # linear decay
        # self.epsilon = self.epsilon - self.eps_dec \
        #                    if self.epsilon > self.eps_min else self.eps_min

        # exponential decay
        self.eps_dec_counter += 1
        self.epsilon = self.eps_min + (self.eps_max - self.eps_min) * np.exp(-self.eps_dec * self.eps_dec_counter)

    def sample_memory(self):
        samples = \
            self.memory.sample(self.batch_size, 0.6)

        print(samples['action'])

        states = T.tensor(samples['obs'], device=self.q_eval.device)
        rewards = T.tensor(samples['reward'], device=self.q_eval.device)
        dones = T.tensor(samples['done'], device=self.q_eval.device, dtype=T.bool)
        actions = T.tensor(samples['action'], device=self.q_eval.device, dtype=T.int64)
        states_ = T.tensor(samples['next_obs'], device=self.q_eval.device)

        return states, actions, rewards, states_, dones, samples['indexes'], samples['weights']


    def learn(self, env):
        raise NotImplementedError

    def save_models(self):
        self.q_eval.save_checkpoint()
        self.q_next.save_checkpoint()

    def load_models(self):
        self.q_eval.load_checkpoint()
        self.q_next.load_checkpoint()


class DQNAgent(Agent):
    def __init__(self, *args, **kwargs):
        super(DQNAgent, self).__init__(*args, **kwargs)

        self.q_eval = DeepQNetwork(self.lr, self.n_actions,
                                   input_dims=self.input_dims,
                                   name=self.env_name + '_' + self.algo + '_q_eval_',
                                   chkpt_dir=self.chkpt_dir)
        self.q_next = DeepQNetwork(self.lr, self.n_actions,
                                   input_dims=self.input_dims,
                                   name=self.env_name + '_' + self.algo + '_q_next_',
                                   chkpt_dir=self.chkpt_dir)

    def choose_action(self, env, observation):

        if np.random.random() > self.epsilon:
            state = T.tensor([observation], dtype=T.float).to(self.q_eval.device)
            actions = self.q_eval.forward(state)
            # action masking
            print(f'q values: {actions}')
            system = np.argmax(env.system)
            # action masking
            valid_actions = env.get_valid_actions(env.system)
            action_mask = np.zeros(len(env.action_space), dtype=int)
            action_mask[valid_actions] = 1
            action_mask_tensor = T.tensor(action_mask, device=actions.device).unsqueeze(0)

            masked_q_values = actions * action_mask_tensor
            masked_q_values[action_mask_tensor == 0] = float('-inf')

            max_q_index = T.argmax(masked_q_values, dim=1)[0]
            action = np.int64(max_q_index.detach().item())
            print(f'year:{env.year}: action taken by agent: {action}')

        else:
            # random exploration on valid action space
            valid_actions = env.get_valid_actions(env.system)
            action = np.random.choice(valid_actions)
            action = np.int64(action)

            print(f'year: {env.year}: random action taken: {action}')

        return action

    def learn(self, env):
        if self.memory.size < self.batch_size or self.memory.size < 1000:
            return

        self.q_eval.optimizer.zero_grad()

        self.replace_target_network()

        states, actions, rewards, states_, dones, indexes, weights = self.sample_memory()
        indices = np.arange(self.batch_size)

        q_pred = self.q_eval.forward(states)[indices, actions]
        # target masking
        q_next = self.q_next.forward(states_)
        for i in range(actions.size(dim=0)):
            system = states[i][-4:]
            valid_actions = env.get_valid_actions(system)
            action_mask = np.zeros(len(env.action_space), dtype=int)
            action_mask[valid_actions] = 1
            action_mask_tensor = T.tensor(action_mask, device=actions.device).unsqueeze(0)

            masked_q_values = q_next[i] * action_mask_tensor
            masked_q_values[action_mask_tensor == 0] = float('-inf')
            q_next[i] = masked_q_values

        q_next = q_next.max(dim=1, keepdim=True)[0]
        q_next[dones] = 0.0

        q_target = rewards + self.gamma * q_next.squeeze(-1)
        td_errors = q_pred - q_target
        loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
        loss.backward()
        self.q_eval.optimizer.step()
        self.learn_step_counter += 1

        new_priorities = np.abs(td_errors.detach().cpu().numpy()) + 1e-6
        self.memory.update_priorities(indexes, new_priorities)

        self.decrement_epsilon()


class DDQNAgent(Agent):
    def __init__(self, *args, **kwargs):
        super(DDQNAgent, self).__init__(*args, **kwargs)

        self.q_eval = DeepQNetwork(self.lr, self.n_actions,
                                   input_dims=self.input_dims,
                                   name=self.env_name + '_' + self.algo + '_q_eval',
                                   chkpt_dir=self.chkpt_dir)
        self.q_next = DeepQNetwork(self.lr, self.n_actions,
                                   input_dims=self.input_dims,
                                   name=self.env_name + '_' + self.algo + '_q_next',
                                   chkpt_dir=self.chkpt_dir)

    # def choose_action_eval(self, env, observation):
    #     state = T.tensor([observation], dtype=T.float).to(self.q_eval.device)
    #     actions = self.q_eval.forward(state)
    #     # action masking
    #     # print(f'q values: {actions}')
    #
    #     # action masking
    #     valid_actions = env.get_valid_actions(env.system)
    #     action_mask = np.zeros(len(env.action_space), dtype=int)
    #     action_mask[valid_actions] = 1
    #     action_mask_tensor = T.tensor(action_mask, device=actions.device).unsqueeze(0)
    #
    #     masked_q_values = actions*action_mask_tensor
    #     masked_q_values[action_mask_tensor == 0] = float('-inf')
    #
    #     max_q_index = T.argmax(masked_q_values, dim=1)[0]
    #     action = np.int64(max_q_index.detach().item())
    #     print(f'evaluation: year:{env.year}: action taken by agent: {action}')
    #
    #     return action

    def choose_action_eval(self, env, observation):
        state = T.tensor([observation], dtype=T.float).to(self.q_eval.device)
        actions = self.q_eval.forward(state)

        # Get system configuration and map it to index
        system = tuple(env.system)
        system_idx = env.system_to_index[system]

        # Retrieve the precomputed action mask and adjust its shape
        action_mask = env.action_masks[system_idx].to(self.q_eval.device).unsqueeze(0)

        # Apply action mask using masked_fill
        masked_q_values = actions.masked_fill(action_mask == 0, float('-inf'))

        # Alternatively, if you prefer using indexing:
        # masked_q_values[action_mask == 0] = float('-inf')  # This will now work

        max_q_index = T.argmax(masked_q_values, dim=1)[0]
        action = np.int64(max_q_index.detach().item())
        print(f'Evaluation: year:{env.year}, action taken by agent: {action}')

        return action

    def choose_action(self, env, observation):
        if np.random.random() > self.epsilon:
            state = T.tensor([observation], dtype=T.float).to(self.q_eval.device)
            actions = self.q_eval.forward(state)

            # Get system configuration and map it to index
            system = tuple(env.system)
            system_idx = env.system_to_index[system]

            # Retrieve the precomputed action mask and adjust its shape
            action_mask = env.action_masks[system_idx].to(self.q_eval.device).unsqueeze(0)

            # Apply action mask using masked_fill
            masked_q_values = actions.masked_fill(action_mask == 0, float('-inf'))

            max_q_index = T.argmax(masked_q_values, dim=1)[0]
            action = np.int64(max_q_index.detach().item())
            print(f'Year:{env.year}, action taken by agent: {action}')
        else:
            # Random exploration on valid action space
            valid_actions = env.dynamic_action_space[env.system]
            action = np.random.choice(valid_actions)
            action = np.int64(action)
            print(f'Year: {env.year}, random action taken: {action}')

        return action
    # def choose_action(self, env, observation):
    #
    #     if np.random.random() > self.epsilon:
    #         state = T.tensor([observation], dtype=T.float).to(self.q_eval.device)
    #         actions = self.q_eval.forward(state)
    #         # action masking
    #         # print(f'q values: {actions}')
    #         system = np.argmax(env.system)
    #         # action masking
    #         valid_actions = env.get_valid_actions(env.system)
    #         action_mask = np.zeros(len(env.action_space), dtype=int)
    #         action_mask[valid_actions] = 1
    #         action_mask_tensor = T.tensor(action_mask, device=actions.device).unsqueeze(0)
    #
    #         masked_q_values = actions * action_mask_tensor
    #         masked_q_values[action_mask_tensor == 0] = float('-inf')
    #
    #         max_q_index = T.argmax(masked_q_values, dim=1)[0]
    #         action = np.int64(max_q_index.detach().item())
    #         print(f'year:{env.year}: action taken by agent: {action}')
    #
    #     else:
    #         # random exploration on valid action space
    #         valid_actions = env.get_valid_actions(env.system)
    #         action = np.random.choice(valid_actions)
    #         action = np.int64(action)
    #
    #         print(f'year: {env.year}: random action taken: {action}')
    #
    #     return action

    # def learn(self, env):
    #     if self.memory.size < self.batch_size or self.memory.size < 4000:
    #         return
    #
    #     self.q_eval.optimizer.zero_grad()
    #
    #     self.replace_target_network()
    #
    #     states, actions, rewards, states_, dones, indexes, weights = self.sample_memory()
    #     indices = np.arange(self.batch_size)
    #
    #     q_pred = self.q_eval.forward(states)[indices, actions]
    #     q_next = self.q_next.forward(states_)
    #     q_eval = self.q_eval.forward(states_)
    #
    #     # Original implementation
    #     # for i in range(actions.size(dim=0)):
    #     #     system = states[i][-4:]
    #     #     valid_actions = env.get_valid_actions(system)
    #     #     action_mask = np.zeros(len(env.action_space), dtype=int)
    #     #     action_mask[valid_actions] = 1
    #     #     action_mask_tensor = T.tensor(action_mask, device=actions.device).unsqueeze(0)
    #     #
    #     #     masked_q_values = q_eval[i] * action_mask_tensor
    #     #     masked_q_values[action_mask_tensor == 0] = float('-inf')
    #     #     q_eval[i] = masked_q_values
    #
    #     # Vectorized action masking
    #     systems = states_[:, -4:]  # Assuming the last 4 elements represent the system state
    #     action_mask_tensor = T.zeros(q_eval.size(), device=actions.device)
    #
    #     for i in range(self.batch_size):
    #         system = systems[i]
    #         valid_actions = env.get_valid_actions(system)
    #         action_mask_tensor[i, valid_actions] = 1
    #
    #     masked_q_values = q_eval * action_mask_tensor
    #     masked_q_values[action_mask_tensor == 0] = float('-inf')
    #
    #     max_actions = T.argmax(masked_q_values, dim=1)
    #     q_next[dones] = 0.0
    #
    #     q_target = rewards + self.gamma * q_next[indices, max_actions]
    #     td_errors = q_pred - q_target
    #     loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
    #     loss.backward()
    #
    #     self.q_eval.optimizer.step()
    #     self.learn_step_counter += 1
    #
    #     new_priorities = np.abs(td_errors.detach().cpu().numpy()) + 1e-6
    #     self.memory.update_priorities(indexes, new_priorities)
    #     self.decrement_epsilon()
    #
    #     loss_value = loss.detach().cpu().item()
    #     print('Loss: {:.4f}'.format(loss_value))

    def learn(self, env):
        if self.memory.size < self.batch_size or self.memory.size < 4000:
            return

        self.q_eval.optimizer.zero_grad()

        self.replace_target_network()

        states, actions, rewards, states_, dones, indexes, weights = self.sample_memory()
        indices = np.arange(self.batch_size)

        q_pred = self.q_eval.forward(states)[indices, actions]
        q_next = self.q_next.forward(states_)
        q_eval = self.q_eval.forward(states_)

        # Extract system configurations from the states
        systems = states[:, -4:]  # Assuming the last 4 elements are the system state

        # Map systems to indices
        systems_np = systems.cpu().numpy()
        system_indices = np.array([env.system_to_index[tuple(system)] for system in systems_np])

        # Retrieve action masks for the batch
        action_masks = env.action_masks[system_indices].to(self.q_eval.device)

        # Apply action masks using masked_fill
        q_eval = q_eval.masked_fill(action_masks == 0, float('-inf'))
        q_next = q_next.masked_fill(action_masks == 0, float('-inf'))

        max_actions = T.argmax(q_eval, dim=1)
        q_next[dones] = 0.0

        q_target = rewards + self.gamma * q_next[indices, max_actions]
        td_errors = q_pred - q_target
        loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
        loss.backward()

        self.q_eval.optimizer.step()
        self.learn_step_counter += 1

        new_priorities = np.abs(td_errors.detach().cpu().numpy()) + 1e-6
        self.memory.update_priorities(indexes, new_priorities)
        self.decrement_epsilon()

        loss_value = loss.detach().cpu().item()
        print('Loss: {:.4f}'.format(loss_value))


class DuelingDQNAgent(Agent):
    def __init__(self, *args, **kwargs):
        super(DuelingDQNAgent, self).__init__(*args, **kwargs)

        self.q_eval = DuelingDeepQNetwork(self.lr, self.n_actions,
                                          input_dims=self.input_dims,
                                          name=self.env_name + '_' + self.algo + '_q_eval',
                                          chkpt_dir=self.chkpt_dir)
        self.q_next = DuelingDeepQNetwork(self.lr, self.n_actions,
                                          input_dims=self.input_dims,
                                          name=self.env_name + '_' + self.algo + '_q_next',
                                          chkpt_dir=self.chkpt_dir)

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = T.tensor([observation], dtype=T.float).to(self.q_eval.device)
            _, advantage = self.q_eval.forward(state)
            action = T.argmax(advantage).item()
        else:
            action = np.random.choice(self.action_space)

        return action

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        self.q_eval.optimizer.zero_grad()

        self.replace_target_network()

        states, actions, rewards, states_, dones = self.sample_memory()
        indices = np.arange(self.batch_size)

        V_s, A_s = self.q_eval.forward(states)
        V_s_, A_s_ = self.q_next.forward(states_)

        q_pred = T.add(V_s,
                       (A_s - A_s.mean(dim=1, keepdim=True)))[indices, actions]
        q_next = T.add(V_s_,
                       (A_s_ - A_s_.mean(dim=1, keepdim=True))).max(dim=1)[0]

        q_next[dones] = 0.0
        q_target = rewards + self.gamma * q_next

        loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
        loss.backward()
        self.q_eval.optimizer.step()

        self.learn_step_counter += 1
        self.decrement_epsilon()


class DuelingDDQNAgent(Agent):
    def __init__(self, *args, **kwargs):
        super(DuelingDDQNAgent, self).__init__(*args, **kwargs)

        self.q_eval = DuelingDeepQNetwork(self.lr, self.n_actions,
                                          input_dims=self.input_dims,
                                          name=self.env_name + '_' + self.algo + '_q_eval',
                                          chkpt_dir=self.chkpt_dir)
        self.q_next = DuelingDeepQNetwork(self.lr, self.n_actions,
                                          input_dims=self.input_dims,
                                          name=self.env_name + '_' + self.algo + '_q_next',
                                          chkpt_dir=self.chkpt_dir)

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = T.tensor([observation], dtype=T.float).to(self.q_eval.device)
            _, advantage = self.q_eval.forward(state)
            action = T.argmax(advantage).item()
        else:
            action = np.random.choice(self.action_space)

        return action

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        self.q_eval.optimizer.zero_grad()

        self.replace_target_network()

        states, actions, rewards, states_, dones = self.sample_memory()

        indices = np.arange(self.batch_size)

        V_s, A_s = self.q_eval.forward(states)
        V_s_, A_s_ = self.q_next.forward(states_)

        V_s_eval, A_s_eval = self.q_eval.forward(states_)

        q_pred = T.add(V_s,
                       (A_s - A_s.mean(dim=1, keepdim=True)))[indices, actions]
        q_next = T.add(V_s_, (A_s_ - A_s_.mean(dim=1, keepdim=True)))

        q_eval = T.add(V_s_eval,
                       (A_s_eval - A_s_eval.mean(dim=1, keepdim=True)))

        max_actions = T.argmax(q_eval, dim=1)
        q_next[dones] = 0.0

        q_target = rewards + self.gamma * q_next[indices, max_actions]
        loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
        loss.backward()
        self.q_eval.optimizer.step()
        self.learn_step_counter += 1

        self.decrement_epsilon()
