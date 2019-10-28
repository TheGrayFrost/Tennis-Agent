import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

units1 = 400
units2 = 300

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)


class ActorNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc_1_units=units1, fc_2_units=units2):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc_1_units (int): Number of nodes in first hidden layer
            fc_2_units (int): Number of nodes in second hidden layer
        """
        super(ActorNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc_1 = nn.Linear(state_size, fc_1_units)
        self.fc_2 = nn.Linear(fc_1_units, fc_2_units)
        self.fc_3 = nn.Linear(fc_2_units, action_size)
        self.initialize()

    def initialize(self): # initialization 
        self.fc_1.weight.data.uniform_(*hidden_init(self.fc_1))
        self.fc_2.weight.data.uniform_(*hidden_init(self.fc_2))
        self.fc_3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state): # forward pass
        """Build an actor (policy) network that maps states -> actions."""
        x = F.relu(self.fc_1(state))
        x = F.relu(self.fc_2(x))
        return torch.tanh(self.fc_3(x))


class CriticNetwork(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, seed, fc_s1_units=units1, fc_2_units=units2):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc_s1_units (int): Number of nodes in the first hidden layer
            fc_2_units (int): Number of nodes in the second hidden layer
        """
        super(CriticNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc_s1 = nn.Linear(state_size, fc_s1_units)
        self.fc_2 = nn.Linear(fc_s1_units+action_size, fc_2_units)
        self.fc_3 = nn.Linear(fc_2_units, 1)
        self.initialize()

    def initialize(self):
        self.fc_s1.weight.data.uniform_(*hidden_init(self.fc_s1))
        self.fc_2.weight.data.uniform_(*hidden_init(self.fc_2))
        self.fc_3.weight.data.uniform_(-3e-3, 3e-3)


    def forward(self, state, action): # concatenate the action-value
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        xs = F.relu(self.fc_s1(state)) # can try F.leaky_relu
        x = torch.cat((xs, action), dim=1)
        x = F.relu(self.fc_2(x))
        return self.fc_3(x)
