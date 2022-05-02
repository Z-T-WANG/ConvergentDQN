import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np
import random
import math
from functools import partial

class DQNBase(nn.Module):
    """
    Basic DQN
    
    parameters
    ---------
    env         environment (openai gym)
    deep        whether to use a deeper convolutional network (bool)
    """
    def __init__(self, env, width=128):
        super(DQNBase, self).__init__()
        
        self.input_shape = env.observation_space.shape
        self.num_actions = env.action_space.n
        self.Linear = Linear # We have overridden the "reset_parameters" method for a more well-principled initialization

        self.fc = nn.Sequential(
            self.Linear(self.input_shape[0], width),
            nn.ReLU(),
            self.Linear(width, width),
            nn.ReLU(),
            self.Linear(width, width),
            nn.ReLU(),
            self.Linear(width, width),
            nn.ReLU(),
            self.Linear(width, self.num_actions)
        )

    def forward(self, x, **kwargs):
        x = self.fc(x)
        return x
    
    def act(self, state, **kwargs):
        """
        Parameters
        ----------
        state       torch.Tensor with appropritate device type
        epsilon     epsilon for epsilon-greedy
        """
        if state.dim() == 1:
            with torch.no_grad():
                state   = state.unsqueeze(0)
                q_values = self.forward(state, **kwargs).cpu().numpy().squeeze()
                bestAction = np.argmax(q_values, axis = 0)
                action = bestAction
        elif state.dim() == 2:
            with torch.no_grad():
                q_values = self.forward(state, **kwargs).cpu().numpy()
                bestAction = np.argmax(q_values, axis = 1)
                action = np.copy(bestAction)
        else: assert False, "The input state has an invalid shape {}".format(state.size())
        return action, q_values

class Linear(nn.Linear):
    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, nonlinearity = "relu", a=0.)
        if self.bias is not None:
            fan_in, _  = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1./math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)
            #self.bias.data.zero_()

