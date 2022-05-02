import torch
import torch.optim as optim
import os, glob, random
import numpy as np
from common.utils import epsilon_scheduler
from model import DQN
import copy, math
from matplotlib import pyplot

def test(env, args): 
    current_model = DQN(env, args).to(args.device)
    epsilon_by_frame = epsilon_scheduler(args.eps_start, args.eps_mid, args.eps_final)
    #current_model.eval()

    results = []
    for filename in args.load_model:
        model_dict, step_idx = torch.load(os.path.join(args.env, '{}.pth'.format(filename)), map_location = args.device)
        print("load {} at training step {}".format(filename, step_idx))
        model_dict.pop("scale", None)
        current_model.load_state_dict(model_dict)
        
        mean, std = test_whole(current_model, env, args, filename, 0.01, num_trial=args.num_trial)#epsilon_by_frame(step_idx))
        results.append(mean)
        with open('{}.txt'.format(args.env), 'a') as data_f:
            data_f.write('{}: {} +- {}\n'.format(filename, mean, std))
    if len(args.load_model)>1:
        with open('{}.txt'.format(args.env), 'a') as data_f:
            data_f.write('{} +- {}\n'.format(np.mean(results), np.std(results, ddof=1)/math.sqrt(len(results))))
            print('{} +- {}'.format(np.mean(results), np.std(results, ddof=1)/math.sqrt(len(results))))

def test_whole(current_model, env, args, idx, epsilon, num_parallel=16, num_trial=400): #16
    image_shape = env.observation_space.shape[1:]
    envs = [env] 
    num_parallel = min(num_parallel, num_trial)
    for _i in range(num_parallel-1): envs.append(copy.deepcopy(env)) 
    for env in envs: env.seed(random.randrange(1000000))
    states = [env.reset() for env in envs] 
    episode_rewards = [0. for _i in range(num_parallel)] 
    episode_lengths = [0 for _i in range(num_parallel)] 
    reward_results = []
    length_results = []

    trial = len(states)
    mark_remove = []
    while len(envs)>0:
        # decide the action
        epsilons = ( epsilon for _ in range(len(states)) ) 

        tensored_states = torch.from_numpy(np.array([state._frames for state in states]).reshape((len(states), -1) + image_shape)).to(args.device).float().div_(255.)
        actions, evaluateds, (Qss, bestActions) = current_model.act(tensored_states, epsilons) 
        for _i, (env, action) in enumerate(zip(envs, actions)): 

            # the environment proceeds by one step (4 frames)
            next_state, reward, done, info = env.step(action)

            episode_rewards[_i] += reward
            episode_lengths[_i] += 1

            if done:
                if env.unwrapped.ale.game_over() or "TimeLimit.truncated" in info:
                    reward_results.append(episode_rewards[_i]); length_results.append(episode_lengths[_i])
                    episode_rewards[_i] = 0.; episode_lengths[_i] = 0 
                    if trial < num_trial:
                        trial += 1
                    else:
                        mark_remove.append(_i)
                states[_i] = env.reset()
            else:
                states[_i] = next_state
        for _i in reversed(mark_remove):
            envs.pop(_i); states.pop(_i); episode_lengths.pop(_i); episode_rewards.pop(_i)
        mark_remove.clear()

    mean_reward = np.mean(reward_results)
    std_reward = np.std(reward_results, ddof=1)/math.sqrt(len(reward_results))
    mean_length = np.mean(length_results)
    print("Test Result - Reward {:.2f}+-{:.2f} Length {:.1f} for {}".format(mean_reward, std_reward, mean_length, idx))
    return mean_reward, std_reward
    
