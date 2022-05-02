import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from model import DQNBase as DQN
from arguments import get_args
import os, sys, datetime, copy, random, math


def train(env, dataset, args):
    current_model = DQN(env, width=128)
    target_model = copy.deepcopy(current_model)
    current_model, target_model = current_model.to(args.device), target_model.to(args.device)
    target_model.load_state_dict(current_model.state_dict())
    for para in target_model.parameters(): para.requires_grad = False
    performance_results = []
    Qs = []
    optimizer = optim.Adam(current_model.parameters(), lr=args.lr, eps = args.adam_eps, betas=(args.beta1,args.beta2), amsgrad=False)
    test_input = torch.tensor(np.arange(env.metadata['river_waterfall_x'])/env.metadata['river_waterfall_x'], dtype=torch.float32, device=args.device).unsqueeze(1)
    with torch.no_grad():
        Qs.append(current_model(test_input).cpu().numpy())
        
    for epoch in range(args.epochs):
        adjust_lr(optimizer, epoch, args)
        dataset = shuffle_dateset(dataset)
        state, next_state, action, reward, done = dataset
        state, next_state = torch.tensor(state, device=args.device)/env.metadata['river_waterfall_x'], torch.tensor(next_state, device=args.device)/env.metadata['river_waterfall_x']
        action, reward, done = torch.tensor(action, device=args.device, dtype=int), torch.tensor(reward, device=args.device)/env.metadata['river_waterfall_x'], torch.tensor(done, device=args.device)
        tensor_data = state, next_state, action, reward, done
        loss = optimize_loss(current_model, target_model, tensor_data, optimizer, epoch, args)
        if not args.no_test and (epoch+1)%5==0:
            performance, performance_ste = test(current_model, env, epoch+1, args)
            performance_results.append(performance)
            performance_str = "{:.2f}+-{:.2f}".format(performance, performance_ste)
        else:
            performance_str = "n/a"
        with torch.no_grad():
            Qs.append(current_model(test_input).cpu().numpy())
        print("Epoch: {}\tLoss: {:.4g}\tReward: {}".format(epoch+1, loss, performance_str))
    return performance_results, Qs

def optimize_loss(current_model, target_model, tensor_data, optimizer, epoch, args):
    state_data, next_state_data, action_data, reward_data, done_data = tensor_data
    # minimize the L2 loss
    i_start = 0
    batch_size = args.batch_size
    accu_loss = 0.
    completed_batch = 0
    while i_start < len(state_data):
        # "done" is not used
        state, next_state, action, reward, done = state_data[i_start:i_start+batch_size], next_state_data[i_start:i_start+batch_size], action_data[i_start:i_start+batch_size], reward_data[i_start:i_start+batch_size], done_data[i_start:i_start+batch_size]
        i_start += batch_size
        num_data = state.size(0)
        reward_ = reward.cpu().numpy()
        
        optimizer.zero_grad()
        
        # 
        if completed_batch == 0: state_dict_to_load = copy.deepcopy(current_model.state_dict())
        elif completed_batch == 1: target_model.load_state_dict(state_dict_to_load)
        if args.alg == "DQN": 
            with torch.no_grad():
                next_q_value, next_q_action = target_model(next_state).max(1)

            expected_q_value = torch.add(reward, other=next_q_value, alpha=args.gamma)
            q_values = current_model(state)
            q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)

            loss = F.mse_loss(q_value, expected_q_value, reduction='none')

            loss = loss.mean()
            loss.backward()

            with torch.no_grad():
                report_loss = loss.item()

        if args.alg == "CDQN": 
            # compute the current and next state values in a single pass 
            current_and_next_states = torch.cat([state, next_state], dim=0)
            # compute the q values and the gradient
            all_q_values = current_model(current_and_next_states)
            with torch.no_grad():
                q_values, next_q_values = all_q_values[:num_data], all_q_values[num_data:2*num_data]
                q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
                next_q_value, next_q_action = next_q_values.max(1)
                q_value, next_q_value = torch.stack((q_value, next_q_value)).cpu().numpy()
            
            next_q_values_target = target_model(next_state) 
            next_q_value_target = np.max(next_q_values_target.cpu().numpy(), axis=1) 

            expected_q_value_self = reward_ + args.gamma * next_q_value 
            expected_q_value_target = reward_ + args.gamma * next_q_value_target 

            target_mask = (np.abs(q_value - expected_q_value_target) >= np.abs(q_value - expected_q_value_self))
            expected_q_value = np.where(target_mask, expected_q_value_target, expected_q_value_self) 
            
            target_mask = target_mask.astype(np.float32)
            #accu_CDQNratio += np.mean(target_mask) # this is the ratio N_{L_{DQN}}/N_{step}
        
            diff = q_value - expected_q_value 

            q_value_grad =  1./num_data *diff

            all_grads = torch.zeros_like(all_q_values) 
            # manually backpropagate the gradient through the term "expected_q_value" 
            next_q_value_grad = - (1.-target_mask) * q_value_grad

            next_q_value_grad = next_q_value_grad * args.gamma 
            grads = torch.from_numpy(np.concatenate([q_value_grad, next_q_value_grad], axis=0)).unsqueeze(1).to(args.device) 
            all_grads.scatter_(1, torch.cat([action, next_q_action], dim=0).unsqueeze(1), grads) 
            all_q_values.backward(all_grads) # this method makes it run faster 

            report_loss = np.dot(diff, diff)/num_data

        if args.alg == "RG": 
            # compute the current and next state values in a single pass 
            current_and_next_states = torch.cat([state, next_state], dim=0)
            # compute the q values and the gradient
            all_q_values = current_model(current_and_next_states)
            with torch.no_grad():
                q_values, next_q_values = all_q_values[:num_data], all_q_values[num_data:2*num_data]
                q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1) 
                next_q_value, next_q_action = next_q_values.max(1)
                q_value, next_q_value = torch.stack((q_value, next_q_value)).cpu().numpy()
                expected_q_value = reward_ + args.gamma * next_q_value         
        
            # then compute the q values and the loss
            diff = q_value - expected_q_value 
            q_value_grad =  1./num_data *diff   

            all_grads = torch.zeros_like(all_q_values) 
            # manually backpropagate the gradient through the term "expected_q_value" 
            next_q_value_grad = -args.gamma * q_value_grad

            grads = torch.from_numpy(np.concatenate([q_value_grad, next_q_value_grad], axis=0)).unsqueeze(1).to(args.device) 
            all_grads.scatter_(1, torch.cat([action, next_q_action], dim=0).unsqueeze(1), grads) 
            all_q_values.backward(all_grads) # this method makes it run faster 

            report_loss = np.dot(diff, diff)/num_data 

        optimizer.step() 

        completed_batch += 1
        accu_loss += report_loss

    avg_loss = accu_loss/completed_batch
    if not args.silent:
        if not os.path.isdir(args.env): os.mkdir(args.env)
        with open(os.path.join(args.env, '{}mse_{}.txt'.format(args.alg, args.comment)), 'a') as f:
            f.write('{:.0f}\t{}\n'.format(epoch, avg_loss))
    return avg_loss

def test(model, env, epoch, args):
    state = env.reset()
    envs = [copy.deepcopy(env) for i in range(args.num_trial)]
    for env in envs: env.seed() 
    states = [env.reset() for env in envs]
    next_states = []
    n_completes = 0
    mark_remove = []
    total_reward = 0.
    print_an_episode = True
    epi_reward = [0. for i in range(args.num_trial)]
    
    num_of_steps = 300
    for i in range(num_of_steps):
    #while len(envs) > 0:
        # we compute the actions for different copies of the environment simultaneouly (with different random seeds)
        tensor_states = torch.tensor(states, device=args.device)/env.metadata['river_waterfall_x']
        actions, Qs = model.act(tensor_states)
        #if i == 100:
        #    actual_actions=list(zip([state.item() for state in states], show_moves(actions)))
        #    actual_actions.sort(key=lambda x:x[0])
        #    print(actual_actions)
        for _i, (env, action) in enumerate(zip(envs, actions)): 
            # the environment proceeds by one step 
            next_state, reward, done, _ = env.step(action)
            assert reward >= 0. 
            total_reward += reward
            #num_of_steps += 1
            epi_reward[_i] = epi_reward[_i] + reward
            #if print_an_episode and _i ==1 and epoch>=30: 
            #    if action==0: move = "->"
            #    elif action == 1: move = "- "
            #    elif action == 2: move = "<-"
                #print("{:.2f} {}  ".format(states[_i].item(), move), end="")#; epi_reward += reward; epi_steps += 1
            #if num_of_steps>10000 and num_of_steps<10000+1000:
            #    print("{:.1f},{}: {:.1f}".format(reward, num_of_steps, total_reward/num_of_steps), end=" ")
            #if done or env.step_n >args.max_episode_steps: # 
            #    mark_remove.append(_i)
            #    if _i == 1 : print_an_episode=False; print(epi_reward/epi_steps)
            #else:
            states[_i] = next_state
        #for _i in reversed(mark_remove):
        #    envs.pop(_i); states.pop(_i)
        #mark_remove.clear()
    epi_reward = np.array(epi_reward)/num_of_steps
    return np.mean(epi_reward), np.std(epi_reward, ddof=1)/math.sqrt(len(epi_reward))#total_reward/num_of_steps, 

def adjust_lr(optimizer, epoch, args):
    lr_decay = 1
    if epoch == round(args.epochs/2) - 1:
        lr_decay = 0.1
        lr = args.lr * lr_decay
    elif epoch == round(args.epochs/4*3)  - 1:
        lr_decay = 0.01
        lr = args.lr * lr_decay
    
    if lr_decay != 1:
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

def show_moves(actions):
    move = []
    for a in actions:
        if a == 0: move.append("->")
        elif a == 1: move.append("--")
        elif a == 2: move.append("<-")
    return move

def shuffle_dateset(dataset):
    c = list(zip(*dataset))
    random.shuffle(c)
    state, next_state, action, reward, done = zip(*c)
    return state, next_state, action, reward, done
