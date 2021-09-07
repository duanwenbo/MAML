from os import write
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
import math
import csv

class CategoricalPolicy(nn.Module):
    """
    input_size: init
    hidden_size: init
    output_size: init
    return: list ? probabilities of each possible action
    """
    def __init__(self,input_size, hidden_size, output_size):
        super(CategoricalPolicy, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
    
    def forward(self, state):
        """
        state: array, from the env
        """
        x = F.relu(self.fc1(state))
        x = F.relu((self.fc2(x)))
        x = F.softmax(self.fc3(x))
        return x


class StateValueNet(nn.Module):
    """
    input_size: init
    hidden_size: init
    output_size: init
    return: list ? probabilities of each possible action
    """
    def __init__(self,input_size, hidden_size, output_size):
        super(StateValueNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
    
    def forward(self, state):
        """
        state: array, from the env
        """
        x = F.relu(self.fc1(state))
        x = F.relu((self.fc2(x)))
        x = self.fc3(x)
        return x