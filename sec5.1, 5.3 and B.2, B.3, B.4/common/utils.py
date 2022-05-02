import math
import os
import time
import random

import torch
import numpy as np
from numba import njit

def update_target(current_model, target_model):
    dic = current_model.state_dict()
    target_model.load_state_dict(dic, strict = False)

def epsilon_scheduler(eps_start, eps_mid, eps_final):
    exponential_decay_rate = 1000000/math.log(eps_start/eps_mid)
    def function(step_idx):
        if step_idx <= 1000000:
            return eps_start * math.exp( -step_idx / exponential_decay_rate)
        elif step_idx <= 40000000:
            return eps_mid - (eps_mid - eps_final) * (step_idx-1000000) / 39000000
        else:
            return eps_final
    return function

def beta_scheduler(beta_start, beta_frames):
    def function(frame_idx):
        return min(1.0, beta_start + (1.0 - beta_start) * frame_idx / beta_frames)
    return function

def print_log(frame, prev_frame, prev_time, reward_list, length_list, loss_list, args, data_string='', **kwargs):
    fps = (frame - prev_frame) / (time.time() - prev_time)
    avg_reward = np.mean(reward_list)
    avg_length = np.mean(length_list)
    avg_loss = np.mean(loss_list) if len(loss_list) != 0 else 0.
    additionals, additionalSave = "", ""
    if kwargs:
        for k, v in kwargs.items(): 
            # if v is a list, we calculate its mean and print with {:.4f}; otherwise we should preprocess the value with a desired precision into the str format 
            if type(v) == list: 
                v = np.mean(v) if len(v) != 0 else 0.
                v = "{:.4f}".format(v)
            additionals += " {}: {}".format(k, v)
            additionalSave += "\t{}".format(v)
    print("Step: {:<8} FPS: {:.1f} Avg. Reward: {:.1f} Avg.Lf.Length: {:.1f} Avg.Loss: {:.4f}{}".format(
        frame, fps, avg_reward, avg_length, avg_loss, additionals
    ))
    if not args.silent:
        with open('data_{}_{}{}.txt'.format(args.optim, args.env, data_string), 'a') as f:
            f.write('{:.0f}\t{}\t{}{}\n'.format((frame + prev_frame)/2., avg_reward, avg_loss, additionalSave))

def print_args(args):
    print(' ' * 26 + 'Options')
    for k, v in vars(args).items():
        print(' ' * 26 + k + ': ' + str(v))

def load_model(model, args):
    if args.load_model != "":
        fname = os.path.join("models", args.env, args.load_model)
    else:
        fname = ""
        if args.prioritized_replay:
            fname += "per-"
        if args.dueling:
            fname += "dueling-"
        if args.double:
            fname += "double-"
        fname += "dqn-{}.pth".format(args.save_model) # when "args.load_model" == '', we use "args.save_model"
        fname = os.path.join("models", args.env, fname)
    if not os.path.exists(fname):
        raise ValueError("No model saved at {}".format(fname))
    model.load_state_dict(torch.load(fname, map_location = args.device))

def set_global_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    set_numba_seeds(seed)

@njit
def set_numba_seeds(seed):
    np.random.seed(seed)
    random.seed(seed)
