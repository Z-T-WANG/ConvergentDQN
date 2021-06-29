import torch
import torch.optim as optim
import torch.nn.functional as F

import optimizers
import time, os, random
import numpy as np
import math, copy 
from collections import deque

from common.utils import epsilon_scheduler, beta_scheduler, update_target, print_log, load_model, print_args
from model import DQN 
from common.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
#from matplotlib import pyplot

def train(env, args):
    image_shape = env.observation_space.shape[1:]

    current_model = DQN(env, args).to(args.device)
    target_model = DQN(env, args).to(args.device)

    print('    Total params: %.2fM' % (sum(p.numel() for p in current_model.parameters())/1000000.0))
    for para in target_model.parameters(): para.requires_grad = False
    update_target(current_model, target_model)

    epsilon_by_frame = epsilon_scheduler(args.eps_start, args.eps_mid, args.eps_final)
    beta_by_frame = beta_scheduler(args.beta_start, args.beta_frames)

    if args.prioritized_replay:
        replay_buffer = PrioritizedReplayBuffer(args.buffer_size, args.alpha, args.IS_weight_only_smaller, allowed_avg_min_ratio = args.ratio_min_prio)
    else:
        replay_buffer = ReplayBuffer(args.buffer_size)

    #args.action_space = env.unwrapped.get_action_meanings()
    args.init_lives = env.unwrapped.ale.lives()
    print_args(args)

    args.do_update_target = False
    # specify the RL algorithm to use 
    if args.algorithm != "DQN" and args.algorithm != "Residual" and args.algorithm != "CDQN" :
        currentTask = "DQN"
        args.currentTask = currentTask
    else:
        currentTask = args.algorithm
        args.currentTask = args.algorithm

    # prepare the optimizer
    lr = args.lr
    beta1 = args.beta1
    beta2 = args.beta2
    parameters = current_model.parameters
    args.optim = args.optim.lower()
    if args.optim=='sgd':
        optimizer = optim.SGD(parameters(), lr=lr, momentum=beta1)
    elif args.optim=='adam':
        optimizer = optimizers.AdamW(parameters(), lr=lr, eps = args.adam_eps, betas=(beta1,beta2), amsgrad=False)
    elif args.optim.startswith("adamb"):
        args.optim = "adambelief"
        optimizer = optimizers.AdamBelief(parameters(), lr=lr, eps = args.adam_eps, betas=(beta1,beta2), amsgrad=False)
    elif args.optim=='laprop':
        optimizer = optimizers.LaProp(parameters(), lr=lr, eps = args.adam_eps, betas=(beta1,beta2), amsgrad=False)
    else:
        assert False, "The specified optimizer name {} is non-existent".format(args.optim)
    
    print(currentTask)


    reward_list, length_list, loss_list, off_policy_rate_list, gen_loss_list = [], [], [], [], []

    clip_reward = True ###
    state = env.reset() 

    # the number of parallelized computation is maximally "arg.train_freq" to guarantee that the computation order is still consistent with the original method
    num_task = args.train_freq 
    args.num_task = num_task 
    episode_rewards = [0. for _i in range(num_task)] 
    life_lengths = [0 for _i in range(num_task)] 

    envs = [env] 
    for _i in range(num_task-1): envs.append(copy.deepcopy(env)) 
    for env in envs: env.seed(random.randrange(1000000))
    states = [state for _i in range(num_task)] 
    rewards = [0 for _i in range(num_task)] 

    evaluation_interval = args.evaluation_interval 
    data_to_store = []

    prev_time = time.time()
    prev_step = 0
    step_idx = 1 # initialization of step_idx
    image_shape = env.observation_space.shape[1:]
    if args.save_best:
        recent_performances = deque([], maxlen = 40)
        recent_models = deque([], maxlen = 20)
        best_performance = -float("inf")
    while step_idx <= args.max_steps:
        # decide the action
        epsilons = ( epsilon_by_frame(idx) for idx in range(step_idx, step_idx + num_task) ) if step_idx>args.learning_start else (1. for idx in range(num_task))

        tensored_states = torch.from_numpy(np.array([state._frames for state in states]).reshape((num_task, -1) + image_shape)).to(args.device).float().div_(255.)
        actions, evaluateds, (Qss, bestActions) = current_model.act(tensored_states, epsilons) 

        for _i, (env, state, action, Qs, bestAction, reward) in enumerate(zip(envs, states, actions, Qss, bestActions, rewards)): 

            # the environment proceeds by one step (4 frames)
            next_state, reward, done, info = env.step(action)

            if clip_reward:
                raw_reward, reward = reward
            else:
                raw_reward = reward

            rewards[_i] = float(reward)

            episode_rewards[_i] += raw_reward
            life_lengths[_i] += 1

            # store the transition into the memory replay
            if not args.randomly_discard_experience or (args.randomly_discard_experience and random.random()>=0.5): # the data may be randomly discarded
                data_to_store.append((state, action, reward, next_state, float(done)))
            if data_to_store: 
                for data in data_to_store: 
                    replay_buffer.add(*data)
                    if args.randomly_replace_memory and len(replay_buffer) >= args.buffer_size:
                        # probably randomly choose an index to replace 
                        replay_buffer._next_idx = random.randrange(args.buffer_size)
                data_to_store.clear()

            # record the performance of a trajectory
            if done:
                length_list.append(life_lengths[_i])
                life_lengths[_i] = 0
                # only the reward of a real full episode is recorded 
                if env.unwrapped.ale.game_over() or "TimeLimit.truncated" in info:
                    reward_list.append(episode_rewards[_i])
                    if not args.silent:
                        if not os.path.isdir(args.env): os.mkdir(args.env)
                        with open(os.path.join(args.env, '{}_{}.txt'.format(currentTask, args.comment)), 'a') as f:
                            f.write('{:.0f}\t{}\n'.format(step_idx*4, episode_rewards[_i]))
                        if args.save_best and step_idx > args.learning_start:
                            recent_performances.append(episode_rewards[_i])
                            mean_performance = np.mean(recent_performances)
                            if best_performance < mean_performance and len(recent_performances)>=40:
                                assert len(recent_models)==20
                                best_performance = mean_performance
                                torch.save((recent_models[0], step_idx), os.path.join(args.env, '{}_{}.pth'.format(currentTask, args.comment)))
                            recent_models.append(current_model.state_dict().copy())
                    episode_rewards[_i] = 0.
                states[_i] = env.reset()
            else:
                states[_i] = next_state

            # optimize
            if step_idx % args.train_freq == 0 and step_idx > max(args.learning_start, 2*args.batch_size): 
                beta = beta_by_frame(step_idx) 
                loss, off_policy_rate = compute_td_loss(current_model, target_model, replay_buffer, optimizer, args, beta) 
                loss_list.append(loss); off_policy_rate_list.append(off_policy_rate)

            # update the target network
            if step_idx % args.update_target == 0 and currentTask != "Residual": 
                # we defer the update of the target network to the optimization routine to ensure that the target network is not exactly equal to current network
                args.do_update_target = True 
                #update_target(current_model, target_model)

            # print the statistics
            if step_idx % evaluation_interval == 0:
                # it works only if there is at least one episode to report; otherwise "evaluation_interval" is increased
                if len(reward_list) > 0:
                    kwargs = {}
                    kwargs["Off-Policy"] = off_policy_rate_list 
                    print_log(step_idx, prev_step, prev_time, reward_list, length_list, loss_list, args, '{}{:.0e}{}'.format(currentTask, args.lr, args.comment), **kwargs) 
                    reward_list.clear(); length_list.clear(); loss_list.clear()
                    for v in kwargs.values(): 
                        if type(v)==list: v.clear()
                    prev_step = step_idx
                    prev_time = time.time()
                else:
                    evaluation_interval += args.evaluation_interval

            step_idx += 1

i_count=0
accu1, accu2 = 0., 0. 
accu_loss = 0.
def compute_td_loss(current_model, target_model, replay_buffer, optimizer, args, beta=None):
    """
    Calculate loss and optimize
    """
    global i_count, accu1, accu2, accu_loss 

    # sample data
    if args.prioritized_replay:
        state_next_state, action_, reward_, done, weights_, true_weights, indices = replay_buffer.sample(args.batch_size, beta)
        weights = torch.from_numpy(weights_).to(args.device, non_blocking=True)
    else:
        state_next_state, action_, reward_, done, indices = replay_buffer.sample(args.batch_size)
        weights = torch.ones(args.batch_size); weights_ = weights.numpy(); true_weights = weights_
        weights = weights.to(args.device, non_blocking=True)

    # we move data to GPU in chunks
    state_next_state = torch.from_numpy(state_next_state).to(args.device, non_blocking=True).float().div_(255)
    state, next_state = state_next_state
    action = torch.from_numpy(action_).to(args.device, non_blocking=True)
    gamma_mul_one_minus_done_ = (args.gamma * (1. - done)).astype(np.float32)
    if args.currentTask == "DQN":
        # in some cases these data do not really need to be copied to GPU 
        reward, gamma_mul_one_minus_done = torch.from_numpy(np.stack((reward_, gamma_mul_one_minus_done_))).to(args.device, non_blocking=True)
    ##### start training ##### 
    optimizer.zero_grad() 
    # we use "values" to refer to Q values for all state-actions, and use "value" to refer to Q values for states
    if args.currentTask == "DQN": 
        if args.double: 
            with torch.no_grad():
                next_q_values = current_model(next_state)
                next_q_action = next_q_values.max(1)[1].unsqueeze(1) # **unsqueeze
                target_next_q_values = target_model(next_state)
                next_q_value = target_next_q_values.gather(1, next_q_action).squeeze()
                next_q_action = next_q_action.squeeze()
        else:
            with torch.no_grad():
                next_q_value, next_q_action = target_model(next_state).max(1)

        expected_q_value = torch.addcmul(reward, tensor1=next_q_value, tensor2=gamma_mul_one_minus_done)
        q_values = current_model(state)
        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)

        loss = F.mse_loss(q_value, expected_q_value, reduction='none')

        if args.prioritized_replay:
            diff = (q_value.detach() - expected_q_value).cpu().numpy() 
            prios = np.abs(diff) + args.prio_eps #
        loss = (loss * weights).mean()/2.
        loss.backward()

        # we report the mean squared error instead of the Huber loss as the loss
        with torch.no_grad():
            report_loss = (F.mse_loss(q_value, expected_q_value, reduction='none')*weights).mean().item()

    if args.currentTask == "CDQN": 
        # compute the current and next state values in a single pass 
        size = list(state_next_state.size())
        current_and_next_states = state_next_state.view([-1]+size[2:]) 
        # compute the q values and the gradient
        all_q_values = current_model(current_and_next_states)
        with torch.no_grad():
            q_values, next_q_values = all_q_values[:args.batch_size], all_q_values[args.batch_size:2*args.batch_size]
            q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
            next_q_value, next_q_action = next_q_values.max(1)
            q_value, next_q_value = torch.stack((q_value, next_q_value)).cpu().numpy()
            
            next_q_values_target = target_model(next_state) 
            if args.double: 
                next_q_value_target = next_q_values_target.gather(1, next_q_action.unsqueeze(1)).squeeze().cpu().numpy() 
            else:
                next_q_value_target = np.max(next_q_values_target.cpu().numpy(), axis=1) 

            expected_q_value_self = reward_ + gamma_mul_one_minus_done_ * next_q_value 
            expected_q_value_target = reward_ + gamma_mul_one_minus_done_ * next_q_value_target 
            target_mask = (np.abs(q_value - expected_q_value_target) >= np.abs(q_value - expected_q_value_self))
            expected_q_value = np.where(target_mask, expected_q_value_target, expected_q_value_self) 
            target_mask = target_mask.astype(np.float32)
        
        diff = q_value - expected_q_value 
        if args.prioritized_replay: 
            prio_diff = diff 
            prios = np.abs(prio_diff) + args.prio_eps 
        # the Huber loss is used 
        weighted_diff = weights_ * diff
        q_value_grad =  1./args.batch_size *weighted_diff 

        all_grads = torch.zeros_like(all_q_values) 
        # manually backpropagate the gradient through the term "expected_q_value" 
        next_q_value_grad = - (1.-target_mask) * q_value_grad
        next_q_value_grad = next_q_value_grad * gamma_mul_one_minus_done_ 
        grads = torch.from_numpy(np.concatenate([q_value_grad, next_q_value_grad], axis=0)).unsqueeze(1).to(args.device) 
        all_grads.scatter_(1, torch.cat([action, next_q_action], dim=0).unsqueeze(1), grads) 
        all_q_values.backward(all_grads) # this method makes it run faster 

        report_loss = np.dot(diff, weights_ * diff)/args.batch_size 

    if args.currentTask == "Residual": 
        # compute the current and next state values in a single pass 
        size = list(state_next_state.size())
        current_and_next_states = state_next_state.view([-1]+size[2:]) 
        # compute the q values and the gradient
        all_q_values = current_model(current_and_next_states)
        with torch.no_grad():
            q_values, next_q_values = all_q_values[:args.batch_size], all_q_values[args.batch_size:2*args.batch_size]
            q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1) 
            next_q_value, next_q_action = next_q_values.max(1)
            q_value, next_q_value = torch.stack((q_value, next_q_value)).cpu().numpy()
            expected_q_value = reward_ + gamma_mul_one_minus_done_ * next_q_value         
        
        # then compute the q values and the loss
        diff = q_value - expected_q_value 
        if args.prioritized_replay: 
            prio_diff = diff 
            prios = np.abs(prio_diff) + args.prio_eps 
        # the Huber loss is used 
        weighted_diff = weights_ * diff 
        q_value_grad =  1./args.batch_size *weighted_diff   

        all_grads = torch.zeros_like(all_q_values) 
        # manually backpropagate the gradient through the term "expected_q_value" 
        next_q_value_grad = - q_value_grad
        next_q_value_grad = next_q_value_grad * gamma_mul_one_minus_done_ 
        grads = torch.from_numpy(np.concatenate([q_value_grad, next_q_value_grad], axis=0)).unsqueeze(1).to(args.device) 
        all_grads.scatter_(1, torch.cat([action, next_q_action], dim=0).unsqueeze(1), grads) 
        all_q_values.backward(all_grads) # this method makes it run faster 

        report_loss = np.dot(diff, weights_ * diff)/args.batch_size 

    if args.prioritized_replay: 
        replay_buffer.update_priorities(indices, prios)
    # gradient clipping 
    if args.grad_clip > 0.:
        grad_norm = torch.nn.utils.clip_grad_norm_(current_model.parameters(), max_norm = args.grad_clip)
        accu1 += grad_norm
        accu2 += grad_norm**2
    if args.do_update_target: update_target(current_model, target_model); args.do_update_target=False
    optimizer.step() 

    off_policy_rate = np.mean((np.argmax(q_values.detach().cpu().numpy(), axis=1)!=action_).astype(np.float)*true_weights)

    i_count += 1
    accu_loss += report_loss
    report_period = math.ceil(args.evaluation_interval/args.train_freq) 
    if i_count % report_period == 0 and accu1 != 0.: 
        print("gradient norm {:.3f} +- {:.3f}".format(accu1/report_period, math.sqrt(accu2/report_period-(accu1/report_period)**2))) 
        accu1, accu2 = 0., 0.
        if not args.silent:
            with open(os.path.join(args.env, '{}mse_{}.txt'.format(args.currentTask, args.comment)), 'a') as f:
                f.write('{:.0f}\t{}\n'.format((i_count*args.train_freq+args.learning_start)*4, accu_loss/report_period))
        accu_loss = 0.

    return report_loss, off_policy_rate

