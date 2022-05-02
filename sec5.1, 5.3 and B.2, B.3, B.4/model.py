import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import itertools
import numpy as np
import random
import math
from functools import partial
from common.heuristics import get_initialization_stat
from numba import njit

def DQN(env, args):
    if args.dueling:
        model = DuelingDQN(env)
    else:
        model = DQNBase(env)
    return model


class DQNBase(nn.Module):
    """
    Basic DQN
    
    parameters
    ---------
    env         environment (openai gym)
    deep        whether to use a deeper convolutional network (bool)
    """
    def __init__(self, env):
        super(DQNBase, self).__init__()
        
        self.input_shape = env.observation_space.shape
        self.num_actions = env.action_space.n
        self.Linear = Linear # We have overridden the "reset_parameters" method for a more well-principled initialization
        
        self.flatten = Flatten()
        self.features = nn.Sequential(
            nn.Conv2d(self.input_shape[0], 32, kernel_size=8, stride=4, padding=0 if self.input_shape[1]!=105 else 2),
            #nn.Conv2d(4, 32, kernel_size=8, stride=4, padding=0 if self.input_shape[1]!=105 else 2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        for module in self.modules():
            if type(module)==nn.Conv2d: init.kaiming_uniform_(module.weight.data, nonlinearity = "relu", a=0.); module.bias.data.zero_()

        self.fc = nn.Sequential(
            self.Linear(self._feature_size(), 512),
            nn.ReLU(),
            self.Linear(512, self.num_actions)
        )
        self.out_bias = self.fc[-1].bias

    def forward(self, x, **kwargs):
        x = self.features(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x
    
    def _feature_size(self):
        return self.features(torch.zeros(1, *self.input_shape)).view(1, -1).size(1)
    
    def act(self, state, epsilon, **kwargs):
        """
        Parameters
        ----------
        state       torch.Tensor with appropritate device type
        epsilon     epsilon for epsilon-greedy
        """
        if state.dim() == 3:
            with torch.no_grad():
                state   = state.unsqueeze(0)
                q_values = self.forward(state, **kwargs).cpu().numpy().squeeze()
                bestAction = np.argmax(q_values, axis = 0)
            if random.random() >= epsilon:
                action = bestAction
            else:
                action = random.randrange(self.num_actions)
        elif state.dim() == 4:
            with torch.no_grad():
                q_values = self.forward(state, **kwargs).cpu().numpy()
                bestAction = np.argmax(q_values, axis = 1)
                action = np.copy(bestAction)
            for i, e in enumerate(epsilon):
                if random.random() < e:
                    action[i] = random.randrange(self.num_actions)
        else: assert False, "The input state has an invalid shape {}".format(state.size())
        return action, action == bestAction, (q_values, bestAction)


class DuelingDQN(DQNBase):
    """
    Dueling Network Architectures for Deep Reinforcement Learning
    https://arxiv.org/abs/1511.06581
    """
    def __init__(self, env):
        super(DuelingDQN, self).__init__(env)
        self.advantage = self.fc
        self.value = nn.Sequential(
            self.Linear(self._feature_size(), 512),
            nn.ReLU(),
            self.Linear(512, 1)
        )
        self.fc = nn.Sequential(
            self.Linear(self._feature_size(), 512 * 2),
            nn.ReLU(),
            self.Linear(512 * 2, self.num_actions + 1),
            DuelingOutput(self.num_actions)
        )
        # rewrite the parameters of "self.advantage" and "self.value" into "self.fc" so that they are combined into a single computation
        with torch.no_grad():
            for p, p_a, p_v in zip(self.fc[0].parameters(), self.advantage[0].parameters(), self.value[0].parameters()):
                p[:512] = p_a; p[512:512*2] = p_v
            self.fc[2].weight.zero_()
            self.fc[2].weight[:self.num_actions,:512] = self.advantage[2].weight; self.fc[2].weight[-1,512:512*2] = self.value[2].weight
            self.fc[2].bias[:self.num_actions] = self.advantage[2].bias; self.fc[2].bias[-1] = self.value[2].bias
            del self.value, self.advantage
        # mask the backpropagated gradient on "self.fc[2].weight"
        self.register_buffer('grad_mask', torch.zeros(self.num_actions+1, 512 * 2))
        self.grad_mask[:self.num_actions,:512] = 1.; self.grad_mask[-1,512:512*2] = 1.
        self.dueling_grad_hook = self.fc[2].weight.register_hook(lambda grad: self.grad_mask*grad)

        self.out_bias = self.fc[-2].bias

class DuelingOutput(nn.Module):
    def __init__(self, num_actions):
        super(DuelingOutput, self).__init__()

        self.register_buffer('output_matrix', torch.Tensor(num_actions, num_actions+1))
        # set the "-advantage.mean(1, keepdim=True)" term
        self.output_matrix[:,:] = -1./num_actions 
        # set the last input dim, the average value, added to all Qs
        self.output_matrix[:,-1] = 1. 
        # set the diagonal term
        for i in range(num_actions):
            self.output_matrix[i,i] = (num_actions-1)/num_actions 
        # this complete the definition of "output_matrix", which computes "value + (advantage - advantage.mean(1, keepdim=True)) * rescale "
        assert not self.output_matrix.requires_grad

    def forward(self, input):
        return F.linear(input, self.output_matrix, None)

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class Linear(nn.Linear):
    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, nonlinearity = "relu", a=0.)
        if self.bias is not None:
            #fan_in, _  = init._calculate_fan_in_and_fan_out(self.weight)
            #bound = 1./math.sqrt(fan_in)
            #init.uniform_(self.bias, -bound, bound)
            self.bias.data.zero_()


epsilon = 0.01 # the epsilon coefficient used in transformation (for Lipschitz continuity)
_2epsilon, _4epsilon = 2.*epsilon, 4.*epsilon
def inverse_transform_Q(Qs):
    if type(Qs) is torch.Tensor:
        return Qs.sign() * ((torch.sqrt((1.+_4epsilon*(1.+epsilon))/_2epsilon**2 + 1./epsilon*Qs.abs()) - 1./_2epsilon).square() - 1.)
    elif type(Qs) is np.ndarray: return jit_inverse_transform_Q(Qs) 
    else: return np.sign(Qs) * ((math.sqrt((1.+_4epsilon*(1.+epsilon))/_2epsilon**2 + 1./epsilon*abs(Qs)) - 1./_2epsilon)**2 - 1.) # this case is for floating number input

@njit(fastmath=True, parallel=False)
def jit_inverse_transform_Q(Qs):
    return (np.sign(Qs) * ((np.sqrt((1.+_4epsilon*(1.+epsilon))/_2epsilon**2 + 1./epsilon*np.abs(Qs)) - 1./_2epsilon)**2 - 1.)).astype(np.float32) 

def transform_Q(Qs): 
    if type(Qs) is torch.Tensor:
        return torch.sign(Qs) * (torch.sqrt(Qs.abs()+1.) - 1.).add(Qs, alpha=epsilon) 
    elif type(Qs) is np.ndarray: return jit_transform_Q(Qs) 
    else: return np.sign(Qs) * (math.sqrt(abs(Qs)+1.) - 1.) + epsilon*Qs # this case is for floating number input

@njit(fastmath=True, parallel=False)
def jit_transform_Q(Qs):
    return (np.sign(Qs) * (np.sqrt(np.abs(Qs)+1.) - 1.) + epsilon*Qs).astype(np.float32) 

# derivative of transform_Q
@njit(fastmath=True, parallel=False)
def d_transform_Q(Qs): 
    return (0.5/np.sqrt(np.abs(Qs)+1.) + epsilon).astype(np.float32) 

# derivative of inverse_transform_Q
@njit(fastmath=True, parallel=False)
def d_inverse_transform_Q(Qs): 
    return (1./epsilon - (1./epsilon)/np.sqrt(1.+_4epsilon*(1.+epsilon) + _4epsilon*np.abs(Qs))).astype(np.float32) 

@njit(fastmath=True, parallel=False)
def transform_backpropagate(next_q_value_grad, input_transform, coefficient, input_inverse_transform): 
    next_q_value_grad = next_q_value_grad * d_transform_Q(input_transform) 
    next_q_value_grad = next_q_value_grad * coefficient 
    next_q_value_grad = next_q_value_grad * d_inverse_transform_Q(input_inverse_transform) 
    return next_q_value_grad.astype(np.float32) 


def auto_initialize(replay_buffer, model, args):
    storage = replay_buffer._storage 
    num_lives = args.init_lives if args.episode_life else 1
    rescaled_mean, scale, gamma = get_initialization_stat(storage, args, time_scale_multiply = 15, num_lives = num_lives, gamma_bounds = (0.99, 0.9998), provided_gamma=args.gamma) 
    # if there is no complete trajectory to estimate the scale and the mean, we check if we still have the first observation of a nonzero reward, which is used as a scale
    if scale is None:
        for data in storage: 
            if data[2] != 0.: 
                scale = max(data[2]/(15/2.), 1.); print("mean Q is not initialized and scale is set to be {:.1f}".format(scale)); break 
    if scale is not None: 
        for data in storage: 
            data[2] = data[2] / scale 
    else:
        print("no reward is observed to initialize the mean and scale")
    #if rescaled_mean is not None: 
    #    mean_init = transform_Q(rescaled_mean) if args.transform_Q else rescaled_mean 
    #    with torch.no_grad(): 
    #        model.out_bias[:] = mean_init 
    return gamma, scale, rescaled_mean

def compute_value_error(reward_list, predicted_value_list, gamma):
    prev_value = 0. 
    reversed_value_list = [] 
    for r in reversed(reward_list): 
        prev_value *= gamma 
        prev_value += r 
        reversed_value_list.append(prev_value) 
    reversed_value_list.reverse()
    value_list = reversed_value_list
    value_list = np.array(value_list)
    predicted_value_list = np.array(predicted_value_list)
    value_diff = predicted_value_list - value_list
    return np.mean(value_diff), np.mean(value_diff**2)
