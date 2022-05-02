import numpy as np
import math
import random
#from matplotlib import pyplot

def get_initialization_stat(replay_storage, args, time_scale_multiply, provided_gamma=None, num_lives = 1., gamma_bounds = (0.99, 0.9999)):
    reward_lists = extract_reward_lists(replay_storage, args)
    assert num_lives >= 0., "invalid arguments: num_lives {}".format(num_lives)
    # the reward structure is analyzed and "time_scale_multiply" is used
    assert time_scale_multiply >= 1., "invalid arguments: time_scale_multiply {}".format(time_scale_multiply)
    #time_scale_multiply *= num_lives 
    # clean the lists 
    nonzero_reward_lists = remove_zero_return_episodes(reward_lists)
    #nonzero_reward_lists = select_behind_zero_head(nonzero_reward_lists)
    _gamma = 1.-find_out_weighted_inverse_time_scale(nonzero_reward_lists, time_scale_multiply, consider_abs_only = True) #find_out_time_scale_of_nonzero(reward_lists, multiply=time_scale_multiply) 
    assert gamma_bounds[0] <= gamma_bounds[1], "gamma min {}, max {} are not valid".format(gamma_bounds[0], gamma_bounds[1])
    gamma = min(max(_gamma, gamma_bounds[0]), gamma_bounds[1])
    print("gamma is estimated to be {:.6f}".format(gamma))
    # if a complete episode with several lives have been stored as several episodes for learning, we recombine them
    if num_lives > 1:
        reward_lists = extract_reward_lists_with_life_counts(replay_storage, args, num_lives)
    if provided_gamma is None: provided_gamma = gamma
    mean, std = calculate_mean_and_init_std(remove_zero_return_episodes(reward_lists), gamma, time_scale_multiply/2., provided_gamma) 
    if std is not None:
        scale = std
        #scale = std/math.sqrt(time_scale_multiply/2.)
        #scale = max(scale, 1.)
    else:
        scale = None
    print("mean estimated to be {:.1f}, scale set to be {:.2f}".format(mean, scale)) if mean is not None else None
    rescaled_mean = mean/scale if scale is not None else None 
    return rescaled_mean, scale, provided_gamma 

def calculate_mean_and_init_std(reward_lists, gamma, gamma_per_reward, provided_gamma): 
    if len(reward_lists) == 0: 
        return None, None 
    # first we compute the mean using the provided external gamma
    reversed_value_lists = [] 
    for reward_list in reward_lists: 
        prev_value = 0. 
        reversed_value_list = [] 
        for r in reversed(reward_list): 
            prev_value *= provided_gamma 
            prev_value += r 
            reversed_value_list.append(prev_value) 
        reversed_value_lists.append(reversed_value_list)
    mean = np.mean(np.concatenate(reversed_value_lists))
    # then we use the estimated gamma
    reversed_value_lists = [] 
    for reward_list in reward_lists: 
        prev_value = 0. 
        reversed_value_list = [] 
        for r in reversed(reward_list): 
            prev_value *= gamma 
            prev_value += r 
            reversed_value_list.append(prev_value) 
        reversed_value_lists.append(reversed_value_list)
    all_values = np.concatenate(reversed_value_lists)
    init_values = [l[-1] for l in reversed_value_lists]
    gamma0 = 1.-(1.-gamma)*gamma_per_reward
    if len(init_values) == 1:
        return mean, init_values[0]/( (1.-gamma**len(reversed_value_lists[0]))/(1.-gamma) )*( (1.-gamma0**len(reversed_value_lists[0]))/(1.-gamma0) )
    #std = np.std(init_values, ddof=1) if len(init_values)>1 else init_values[0]
    #std = np.std(all_values, ddof=1)
    #print("reward std: {:.6f}".format(np.std(np.hstack([np.array(l) for l in reward_lists]), ddof=1)))
    mu_r = np.mean([init_values[i]*(1.-gamma)/(1.-gamma**len(reversed_value_lists[i])) for i in range(len(reversed_value_lists))])
    real_mu_r = np.mean(np.hstack([np.array(l) for l in reward_lists])) 

    #mu_r = real_mu_r #
    real_sigma_r = np.std(np.hstack([np.array(l) for l in reward_lists]), ddof=1)
    print("mu_r {:.6f}, real mu_r {:.6f}, real std_r {:.6f}".format(mu_r, real_mu_r, real_sigma_r))
    #mu_r = real_mu_r
    Qs = [init_values[i]-(1.-gamma**len(reversed_value_lists[i]))/(1.-gamma)*mu_r  for i in range(len(reversed_value_lists))]
    sigma_r = np.std([(Qs[i])/math.sqrt((1.-gamma**(2*len(reversed_value_lists[i])))/(1.-gamma**2)) for i in range(len(reversed_value_lists))], ddof=1)
    sigma_Q0 = np.mean([math.sqrt((1.-gamma0**(2*len(l)))/(1.-gamma0**2)) for l in reversed_value_lists])*sigma_r
    print("r_std estimated at trjectory head: {:.6f}, std around the means: {:.6f}, c-independent Q std: {:.6f}, Q std predicted by r_std: {:.6f}".format(sigma_r, np.std(Qs), sigma_Q0, np.mean([math.sqrt((1.-gamma0**(2*len(l)))/(1.-gamma0**2)) for l in reversed_value_lists])*real_sigma_r)) #np.std([sigma_r*math.sqrt((1.-gamma0**(2*len(l)))/(1.-gamma0**2)) for l in reversed_value_lists], ddof=1)
    #std = max(1., std)
    return mean, sigma_Q0 

def find_out_weighted_inverse_time_scale(reward_lists, multiply, consider_abs_only = False): 
    # this is our main algorithm, which is used to properly give an estimate of the time scale of the reward 
    if len(reward_lists) == 0: 
        return 0. 
    inverse_expected_dists = [] 
    # we take an average of the estimates on all episodes
    for i, _reward_list in enumerate(reward_lists): 
        reward_list = np.array(_reward_list) 
        sublists = [] 
        if consider_abs_only:
            reward_list = np.abs(reward_list) ###
        if not (np.all(reward_list>=0.) or np.all(reward_list<=0.)): 
            pos_mask = (reward_list>=0.).astype(float) 
            pos_sublist = reward_list*pos_mask 
            neg_sublist = reward_list - pos_sublist 
            while not np.all(pos_sublist==0.):
                sublist, pos_sublist = separate_out_smallest_reward_sublist(pos_sublist)
                sublists.append(sublist)
            while not np.all(neg_sublist==0.):
                sublist, neg_sublist = separate_out_smallest_reward_sublist(neg_sublist)
                sublists.append(sublist)
        else:
            while not np.all(reward_list==0.):
                sublist, reward_list = separate_out_smallest_reward_sublist(reward_list)
                sublists.append(sublist)
        subreturns = np.array([abs(sum(sublist)) for sublist in sublists])
        subdists = np.array([return_weighted_expected_time_until_next_reward(sublist) for sublist in sublists])
        dist = np.sum(subdists*subreturns)/np.sum(subreturns)
        inverse_expected_dists.append(1./dist)
        #if subdists[-1]<20.: # debugging
        #    print(_reward_list)
        #    start_i = sum(len(reward_lists[j]) for j in range(i))
        #    for step_idx in range(start_i, start_i+len(_reward_list)):
        #        state = storage[step_idx][3]
        #        pyplot.imsave("images/{}.png".format(step_idx), state._frames[-1].squeeze())
    inverse_expected_dists = np.array(inverse_expected_dists)
    return_weights = [math.sqrt(np.sum(np.abs(np.array(reward_list)))) for reward_list in reward_lists] # we use square root as the weights for the weighted average
    #print(1-np.average(inverse_expected_dists, weights=return_weights)/ multiply, 1-np.sqrt(np.average(inverse_expected_dists**2, weights=return_weights))/ multiply, 1-1./np.average(1./inverse_expected_dists, weights=return_weights)/ multiply) # we actually have three different possible averaging strategies
    return  np.sqrt(np.average(inverse_expected_dists**2, weights=return_weights))/ multiply #

def separate_out_smallest_reward_sublist(reward_list):
    # take out the sublist that contains the reward with the smallest absolute value, and also substract larger rewards by it 
    nonzero_rewards = reward_list[reward_list!=0.]
    r = nonzero_rewards[np.argmin(np.abs(nonzero_rewards))] 
    sublist_mask = (reward_list==r).astype(float)
    sublist = reward_list*sublist_mask
    list_remain = reward_list - sublist
    return sublist, list_remain

def return_weighted_expected_time_until_next_reward(reward_list):
    dist_list = [] 
    future_return = 0. 
    dist = 0 
    total_return_weight = 0. 
    for r in reversed(reward_list): 
        if r != 0.: 
            dist = 1 
            future_return += abs(r) 
        else: 
            dist += 1 
        dist_list.append(future_return*dist) 
        total_return_weight += future_return 
    return sum(dist_list)/total_return_weight 

def find_out_time_scale_of_nonzero(reward_lists, multiply):
    # use the time scale of observing any nonzero reward 
    if len(reward_lists) == 0: 
        return float("inf") 
    expected_dists = []
    for reward_list in reward_lists:
        expected_dists.append(return_weighted_expected_time_until_next_reward(reward_list))
    return np.mean(expected_dists) * multiply

def extract_reward_lists(replay_storage, args):
    reward_lists = []
    for start_idx in range(0, args.num_task):
        # We append the reward data into "l", and whenever "done" is True, we store "l" as a single reward list and start a new "l". 
        # In this way we can automatically ignore the incomplete episode at the endpoint of the storage.
        l = []
        for i in range(start_idx, len(replay_storage), args.num_task):
            data = replay_storage[i]
            l.append(data[2]) # reward
            if data[4]: # done
                reward_lists.append(l)
                l = []
    return reward_lists

def extract_reward_lists_with_life_counts(replay_storage, args, num_lives):
    reward_lists = []
    for start_idx in range(0, args.num_task):
        # We append the reward data into "l", and whenever "done" is True, we store "l" as a single reward list and start a new "l". 
        # In this way we can automatically ignore the incomplete episode at the endpoint of the storage.
        l = []
        lives = num_lives
        for i in range(start_idx, len(replay_storage), args.num_task):
            data = replay_storage[i]
            l.append(data[2]) # reward
            if data[4]: # done
                lives -= 1
                if lives <= 0:
                    reward_lists.append(l)
                    l = []
    return reward_lists

def remove_zero_return_episodes(reward_lists):
    nonzero_lists = []
    for l in reward_lists:
        if sum(np.abs(l)) != 0.:
            nonzero_lists.append(l)
    return nonzero_lists

def select_behind_zero_head(reward_lists):
    result_lists = []
    for l in reward_lists:
        for i, r in enumerate(l):
            if r != 0.: break
        result_lists.append(l[i+1:])
    result_lists = remove_zero_return_episodes(result_lists)
    return result_lists

def all_zero(reward_list):
    return np.all(np.array(reward_list)==0.)
