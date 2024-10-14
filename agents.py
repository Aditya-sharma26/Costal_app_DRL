import numpy as np
import torch as T
from deep_q_network import DeepQNetwork, DuelingDeepQNetwork
from replay_memory import ReplayBuffer
# from per import ReplayBuffer


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

        self.memory = ReplayBuffer(mem_size, input_dims, n_actions)
        # self.memory = ReplayBuffer(2**14, 0.6)

    def store_transition(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

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
        state, action, reward, new_state, done = \
                                self.memory.sample_buffer(batch_size= self.batch_size)

        states = T.tensor(state).to(self.q_eval.device)
        rewards = T.tensor(reward).to(self.q_eval.device)
        dones = T.tensor(done).to(self.q_eval.device)
        actions = T.tensor(action).to(self.q_eval.device)
        states_ = T.tensor(new_state).to(self.q_eval.device)

        return states, actions, rewards, states_, dones

    def learn(self):
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
                                    name=self.env_name+'_'+self.algo+'_q_eval',
                                    chkpt_dir=self.chkpt_dir)
        self.q_next = DeepQNetwork(self.lr, self.n_actions,
                                    input_dims=self.input_dims,
                                    name=self.env_name+'_'+self.algo+'_q_next',
                                    chkpt_dir=self.chkpt_dir)

    def choose_action(self, env, observation):

        if np.random.random() > self.epsilon:
            state = T.tensor([observation],dtype=T.float).to(self.q_eval.device)
            actions = self.q_eval.forward(state)
            #action masking
            print(f'q values: {actions}')
            system = np.argmax(env.system)
            # action masking
            if system == 1:
                actions[0][1] = T.min(actions, dim=1)[0] - 1
                actions[0][3] = T.min(actions, dim=1)[0] - 1
            elif system == 2:
                actions[0][2] = T.min(actions, dim=1)[0] - 1
                actions[0][3] = T.min(actions, dim=1)[0] - 1
            elif system == 3:
                actions[0][1] = T.min(actions, dim=1)[0] - 1
                actions[0][2] = T.min(actions, dim=1)[0] - 1
                actions[0][3] = T.min(actions, dim=1)[0] - 1

            max_q_index = T.argmax(actions, dim=1)[0]
            action = np.int64(max_q_index.detach().item())
            print(f'action taken by agent: {action}')

        else:

            # # random exploration with action masking and without do-nothing prioritization
            # if np.argmax(env.system) == 0:
            #     action_space = np.array([0, 1, 2, 3])
            #     # p_action = np.array([0.70, 0.1, 0.1, 0.1])
            #     action = np.random.choice(action_space)
            #     action = np.int64(action)
            # elif np.argmax(env.system) == 1:
            #     action_space = np.array([0, 2])
            #     # p_action = np.array([0.80, 0.20])
            #     action = np.random.choice(action_space)
            #     action = np.int64(action)
            # elif np.argmax(env.system) == 2:
            #     action_space = np.array([0, 1])
            #     # p_action = np.array([0.80, 0.20])
            #     action = np.random.choice(action_space)
            #     action = np.int64(action)
            # elif np.argmax(env.system) == 3:
            #     action_space = np.array([0])
            #     # p_action = np.array([1.00])
            #     action = np.random.choice(action_space)
            #     action = np.int64(action)

            # # random exploration with action masking and do-nothing prioritization
            # if np.argmax(env.system) == 0:
            #     action_space = np.array([0, 1, 2, 3])
            #     p_action = np.array([0.60, 0.20, 0.1, 0.1])
            #     action = np.random.choice(action_space, p=p_action)
            #     action = np.int64(action)
            # elif np.argmax(env.system) == 1:
            #     action_space = np.array([0, 2])
            #     p_action = np.array([0.90, 0.10])
            #     action = np.random.choice(action_space, p=p_action)
            #     action = np.int64(action)
            # elif np.argmax(env.system) == 2:
            #     action_space = np.array([0, 1])
            #     p_action = np.array([0.80, 0.20])
            #     action = np.random.choice(action_space, p=p_action)
            #     action = np.int64(action)
            # elif np.argmax(env.system) == 3:
            #     action_space = np.array([0])
            #     p_action = np.array([1.00])
            #     action = np.random.choice(action_space, p=p_action)
            #     action = np.int64(action)
            #
            # print(f'random action taken: {action}')
            # action = np.random.choice(self.action_space)

            # random exploration with action masking and initial DN prioritization
            if np.argmax(env.system) == 0:
                action_space = np.array([0, 1, 2, 3])
                p_dn = 0.25 + (0.6 - 0.25) * np.exp(-self.eps_dec*self.learn_step_counter)
                p_a1 = 0.25 - (0.25 - 0.2) * np.exp(-self.eps_dec*self.learn_step_counter)
                p_a2 = 0.25 - (0.25 - 0.1) * np.exp(-self.eps_dec*self.learn_step_counter)
                p_a3 = 0.25 - (0.25 - 0.1) * np.exp(-self.eps_dec*self.learn_step_counter)
                
                p_action = np.array([p_dn, p_a1, p_a2, p_a3])
                action = np.random.choice(action_space, p=p_action)
                action = np.int64(action)
            elif np.argmax(env.system) == 1:
                action_space = np.array([0, 2])
                p_dn = 0.50 + (0.8 - 0.50) * np.exp(-self.eps_dec * self.learn_step_counter)
                p_a2 = 0.50 - (0.50 - 0.2) * np.exp(-self.eps_dec * self.learn_step_counter)
                p_action = np.array([p_dn, p_a2])
                action = np.random.choice(action_space, p=p_action)
                action = np.int64(action)
            elif np.argmax(env.system) == 2:
                action_space = np.array([0, 1])
                p_dn = 0.50 + (0.8 - 0.50) * np.exp(-self.eps_dec * self.learn_step_counter)
                p_a1 = 0.50 - (0.50 - 0.2) * np.exp(-self.eps_dec * self.learn_step_counter)
                p_action = np.array([p_dn, p_a1])
                action = np.random.choice(action_space, p=p_action)
                action = np.int64(action)
            elif np.argmax(env.system) == 3:
                action_space = np.array([0])
                p_action = np.array([1.00])
                action = np.random.choice(action_space, p=p_action)
                action = np.int64(action)

            print(f'random action taken: {action}')

        return action

    def learn(self):
        if self.memory.mem_cntr < self.batch_size or self.memory.mem_cntr < 1000:
            return

        self.q_eval.optimizer.zero_grad()

        self.replace_target_network()

        states, actions, rewards, states_, dones = self.sample_memory()
        indices = np.arange(self.batch_size)

        q_pred = self.q_eval.forward(states)[indices, actions]
        # q_pred = self.q_eval.forward(states)
        # q_pred = T.gather(input=q_pred, dim=1, index=actions)
        # target masking
        q_next = self.q_next.forward(states_)
        for i in range(actions.size(dim=0)):
            state_t = states_[i]
            system_t = T.argmax(state_t[-4:])
            if system_t == 1:
                q_next[i][1] = T.min(q_next[i], dim=0)[0] - 1
                q_next[i][3] = T.min(q_next[i], dim=0)[0] - 1
            elif system_t == 2:
                q_next[i][2] = T.min(q_next[i], dim=0)[0] - 1
                q_next[i][3] = T.min(q_next[i], dim=0)[0] - 1
            elif system_t == 3:
                q_next[i][1] = T.min(q_next[i], dim=0)[0] - 1
                q_next[i][2] = T.min(q_next[i], dim=0)[0] - 1
                q_next[i][3] = T.min(q_next[i], dim=0)[0] - 1

        q_next = q_next.max(dim=1, keepdim=True)[0]
        q_next[dones] = 0.0

        q_target = rewards + self.gamma*q_next.squeeze(-1)

        loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
        loss.backward()
        self.q_eval.optimizer.step()
        self.learn_step_counter += 1

        self.decrement_epsilon()

        # new_priorities = np.abs(td_errors.cpu().numpy()) + 1e-6
        # self.memory.update_priorities()

class DDQNAgent(Agent):
    def __init__(self, *args, **kwargs):
        super(DDQNAgent, self).__init__(*args, **kwargs)

        self.q_eval = DeepQNetwork(self.lr, self.n_actions,
                                    input_dims=self.input_dims,
                                    name=self.env_name+'_'+self.algo+'_q_eval',
                                    chkpt_dir=self.chkpt_dir)
        self.q_next = DeepQNetwork(self.lr, self.n_actions,
                                    input_dims=self.input_dims,
                                    name=self.env_name+'_'+self.algo+'_q_next',
                                    chkpt_dir=self.chkpt_dir)

    def choose_action(self, env, observation):

        if np.random.random() > self.epsilon:
            state = T.tensor([observation], dtype=T.float).to(self.q_eval.device)
            actions = self.q_eval.forward(state)
            # action masking
            print(f'q values: {actions}')
            system = np.argmax(env.system)
            # action masking
            if system == 1:
                actions[0][1] = T.min(actions, dim=1)[0] - 1
                actions[0][3] = T.min(actions, dim=1)[0] - 1
            elif system == 2:
                actions[0][2] = T.min(actions, dim=1)[0] - 1
                actions[0][3] = T.min(actions, dim=1)[0] - 1
            elif system == 3:
                actions[0][1] = T.min(actions, dim=1)[0] - 1
                actions[0][2] = T.min(actions, dim=1)[0] - 1
                actions[0][3] = T.min(actions, dim=1)[0] - 1

            max_q_index = T.argmax(actions, dim=1)[0]
            action = np.int64(max_q_index.detach().item())
            print(f'action taken by agent: {action}')

        else:
            # # random exploration with action masking and without do-nothing prioritization
            # if np.argmax(env.system) == 0:
            #     action_space = np.array([0, 1, 2, 3])
            #     # p_action = np.array([0.70, 0.1, 0.1, 0.1])
            #     action = np.random.choice(action_space)
            #     action = np.int64(action)
            # elif np.argmax(env.system) == 1:
            #     action_space = np.array([0, 2])
            #     # p_action = np.array([0.80, 0.20])
            #     action = np.random.choice(action_space)
            #     action = np.int64(action)
            # elif np.argmax(env.system) == 2:
            #     action_space = np.array([0, 1])
            #     # p_action = np.array([0.80, 0.20])
            #     action = np.random.choice(action_space)
            #     action = np.int64(action)
            # elif np.argmax(env.system) == 3:
            #     action_space = np.array([0])
            #     # p_action = np.array([1.00])
            #     action = np.random.choice(action_space)
            #     action = np.int64(action)
            #
            # print(f'random action taken: {action}')


            # # random exploration with action masking and do-nothing prioritization
            # if np.argmax(env.system) == 0:
            #     action_space = np.array([0, 1, 2, 3])
            #     p_action = np.array([0.50, 0.30, 0.1, 0.1])
            #     action = np.random.choice(action_space, p=p_action)
            #     action = np.int64(action)
            # elif np.argmax(env.system) == 1:
            #     action_space = np.array([0, 2])
            #     p_action = np.array([0.90, 0.10])
            #     action = np.random.choice(action_space, p=p_action)
            #     action = np.int64(action)
            # elif np.argmax(env.system) == 2:
            #     action_space = np.array([0, 1])
            #     p_action = np.array([0.80, 0.20])
            #     action = np.random.choice(action_space, p=p_action)
            #     action = np.int64(action)
            # elif np.argmax(env.system) == 3:
            #     action_space = np.array([0])
            #     p_action = np.array([1.00])
            #     action = np.random.choice(action_space, p=p_action)
            #     action = np.int64(action)
            #
            # print(f'random action taken: {action}')
            # action = np.random.choice(self.action_space)


            # random exploration with action masking and initial DN prioritization
            if self.memory.mem_cntr < 400:
                action = 0
            else:
                if np.argmax(env.system) == 0:
                    action_space = np.array([0, 1, 2, 3])
                    p_dn = 0.25 + (0.03 - 0.25) * np.exp(-self.eps_dec * self.learn_step_counter)
                    p_a1 = 0.25 - (0.25 - 0.91) * np.exp(-self.eps_dec * self.learn_step_counter)
                    p_a2 = 0.25 - (0.25 - 0.03) * np.exp(-self.eps_dec * self.learn_step_counter)
                    p_a3 = 0.25 - (0.25 - 0.03) * np.exp(-self.eps_dec * self.learn_step_counter)

                    p_action = np.array([p_dn, p_a1, p_a2, p_a3])
                    action = np.random.choice(action_space, p=p_action)
                    action = np.int64(action)
                elif np.argmax(env.system) == 1:
                    action_space = np.array([0, 2])
                    p_dn = 0.50 + (0.92 - 0.50) * np.exp(-self.eps_dec * self.learn_step_counter)
                    p_a2 = 0.50 - (0.50 - 0.08) * np.exp(-self.eps_dec * self.learn_step_counter)
                    p_action = np.array([p_dn, p_a2])
                    action = np.random.choice(action_space, p=p_action)
                    action = np.int64(action)
                elif np.argmax(env.system) == 2:
                    action_space = np.array([0, 1])
                    p_dn = 0.50 + (0.92 - 0.50) * np.exp(-self.eps_dec * self.learn_step_counter)
                    p_a1 = 0.50 - (0.50 - 0.08) * np.exp(-self.eps_dec * self.learn_step_counter)
                    p_action = np.array([p_dn, p_a1])
                    action = np.random.choice(action_space, p=p_action)
                    action = np.int64(action)
                elif np.argmax(env.system) == 3:
                    action_space = np.array([0])
                    p_action = np.array([1.00])
                    action = np.random.choice(action_space, p=p_action)
                    action = np.int64(action)

            print(f'random action taken: {action}')

        return action

    def learn(self):
        if self.memory.mem_cntr < self.batch_size or self.memory.mem_cntr < 4000:
            return

        self.q_eval.optimizer.zero_grad()

        self.replace_target_network()

        states, actions, rewards, states_, dones = self.sample_memory()
        indices = np.arange(self.batch_size)

        q_pred = self.q_eval.forward(states)[indices, actions]
        q_next = self.q_next.forward(states_)
        q_eval = self.q_eval.forward(states_)
        for i in range(actions.size(dim=0)):
            state_t = states_[i]
            system_t = T.argmax(state_t[-4:])
            if system_t == 1:
                q_eval[i][1] = T.min(q_eval[i], dim=0)[0] - 1
                q_eval[i][3] = T.min(q_eval[i], dim=0)[0] - 1
            elif system_t == 2:
                q_eval[i][2] = T.min(q_eval[i], dim=0)[0] - 1
                q_eval[i][3] = T.min(q_eval[i], dim=0)[0] - 1
            elif system_t == 3:
                q_eval[i][1] = T.min(q_eval[i], dim=0)[0] - 1
                q_eval[i][2] = T.min(q_eval[i], dim=0)[0] - 1
                q_eval[i][3] = T.min(q_eval[i], dim=0)[0] - 1
        max_actions = T.argmax(q_eval, dim=1)
        q_next[dones] = 0.0

        q_target = rewards + self.gamma*q_next[indices, max_actions]
        loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
        loss.backward()

        self.q_eval.optimizer.step()
        self.learn_step_counter += 1

        self.decrement_epsilon()

        loss_value = loss.detach().cpu().item()
        print('Loss: {:.4f}'.format(loss_value))

class DuelingDQNAgent(Agent):
    def __init__(self, *args, **kwargs):
        super(DuelingDQNAgent, self).__init__(*args, **kwargs)

        self.q_eval = DuelingDeepQNetwork(self.lr, self.n_actions,
                        input_dims=self.input_dims,
                        name=self.env_name+'_'+self.algo+'_q_eval',
                        chkpt_dir=self.chkpt_dir)
        self.q_next = DuelingDeepQNetwork(self.lr, self.n_actions,
                        input_dims=self.input_dims,
                        name=self.env_name+'_'+self.algo+'_q_next',
                        chkpt_dir=self.chkpt_dir)

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = T.tensor([observation],dtype=T.float).to(self.q_eval.device)
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
        q_target = rewards + self.gamma*q_next

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
                        name=self.env_name+'_'+self.algo+'_q_eval',
                        chkpt_dir=self.chkpt_dir)
        self.q_next = DuelingDeepQNetwork(self.lr, self.n_actions,
                        input_dims=self.input_dims,
                        name=self.env_name+'_'+self.algo+'_q_next',
                        chkpt_dir=self.chkpt_dir)

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = T.tensor([observation],dtype=T.float).to(self.q_eval.device)
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

        q_target = rewards + self.gamma*q_next[indices, max_actions]
        loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
        loss.backward()
        self.q_eval.optimizer.step()
        self.learn_step_counter += 1

        self.decrement_epsilon()
