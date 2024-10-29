import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class DeepQNetwork(nn.Module):
    def __init__(self, lr, n_actions, name, input_dims, chkpt_dir):
        super(DeepQNetwork, self).__init__()
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)

        # self.conv1 = nn.Conv2d(input_dims[0], 32, 8, stride=4)
        # self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        # self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
        #
        # fc_input_dims = self.calculate_conv_output_dims(input_dims)

        self.fc1 = nn.Linear(input_dims, 64)
        # self.bn = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 32)
        self.fc4 = nn.Linear(32, n_actions)

        self.optimizer = optim.RMSprop(self.parameters(), lr=lr)
        # self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        # self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.device = T.device('cpu')
        self.to(self.device)

    # def calculate_conv_output_dims(self, input_dims):
    #     state = T.zeros(1, *input_dims)
    #     dims = self.conv1(state)
    #     dims = self.conv2(dims)
    #     dims = self.conv3(dims)
    #     return int(np.prod(dims.size()))

    def forward(self, state):
        # conv1 = F.relu(self.conv1(state))
        # conv2 = F.relu(self.conv2(conv1))
        # conv3 = F.relu(self.conv3(conv2))
        # conv_state = conv3.view(conv3.size()[0], -1)
        flat1 = F.relu(self.fc1(state))

        # flat1 = self.fc1(state)
        # flat2 = F.relu(self.bn(flat1))
        flat2 = F.relu(self.fc2(flat1))
        flat3 = F.relu(self.fc3(flat2))
        actions = self.fc4(flat3)

        return actions

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))

class DuelingDeepQNetwork(nn.Module):
    def __init__(self, lr, n_actions, name, input_dims, chkpt_dir):
        super(DuelingDeepQNetwork, self).__init__()

        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)

        # self.conv1 = nn.Conv2d(input_dims[0], 32, 8, stride=4)
        # self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        # self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
        #
        # fc_input_dims = self.calculate_conv_output_dims(input_dims)

        self.fc1 = nn.Linear(input_dims, 512)
        self.fc2 = nn.Linear(512, 256)

        self.V = nn.Linear(256, 1)
        self.A = nn.Linear(256, n_actions)

        self.optimizer = optim.RMSprop(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    # def calculate_conv_output_dims(self, input_dims):
    #     state = T.zeros(1, *input_dims)
    #     dims = self.conv1(state)
    #     dims = self.conv2(dims)
    #     dims = self.conv3(dims)
    #     return int(np.prod(dims.size()))

    def forward(self, state):
        # conv1 = F.relu(self.conv1(state))
        # conv2 = F.relu(self.conv2(conv1))
        # conv3 = F.relu(self.conv3(conv2))
        # conv_state = conv3.view(conv3.size()[0], -1)
        flat1 = F.relu(self.fc1(state))
        flat2 = F.relu(self.fc2(flat1))

        V = self.V(flat2)
        A = self.A(flat2)

        return V, A

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))
