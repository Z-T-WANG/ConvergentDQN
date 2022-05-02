import torch
import numpy as np
from torch.utils.data import DataLoader
import random
from wetchicken1d import WetChicken1dEnv
from model import DQNBase as DQN
from train import train
#from plot import plot_results
from arguments import get_args
import sys, datetime

def main():
    args = get_args()
    if args.env!="wetchicken1d": assert False, "The specified environment is wrong or not supported."
    
    results = []
    Qs = []
    for i in range(args.num_runs):
        print("RUN {}".format(i+1))
        set_seeds(args.seed+i)
        if args.env=="wetchicken1d":
            env = WetChicken1dEnv()
        elif args.env=="wetchicken2d":
            pass
        dataset = generate_dataset(env, args.dataset_size)
        performance_result, Q = train(env, dataset, args)
        results.append(performance_result)
        Qs.append(Q)
    if not args.no_test: torch.save(results, "results_{}.pth".format(args.comment))
    torch.save(Qs, "Qs_{}.pth".format(args.comment))

def generate_dataset(env, dataset_size):
    num_actions = env.action_space.n
    dataset_state, dataset_next_state, dataset_action, dataset_reward, dataset_done = [], [], [], [], [] 
    state = env.reset()
    for i in range(dataset_size):
        action = random.randrange(num_actions)
        next_state, reward, done, _ = env.step(action)
        dataset_state.append(state); dataset_next_state.append(next_state); dataset_action.append(action); dataset_reward.append(reward); dataset_done.append(done)
        state = next_state
        if done:
            state = env.reset() # However, for this environment, the reset does not change its state
    return (dataset_state, dataset_next_state, dataset_action, dataset_reward, dataset_done)

def set_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

if __name__ == "__main__":
    main()
