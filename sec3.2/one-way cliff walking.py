import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import datetime
import random
import math, os
import time
plt.rcParams["mathtext.fontset"]="cm"
plt.rcParams["axes.formatter.use_mathtext"]=True


class GridWorld:
    # this environment does not hold a state; instead, it takes the state as an argument to the "step" function and always returns a state
    height = 2 
    num_action = 2
    action_meaning = {0: "up", 1: "right"}
    cliff_color, goal_color = colors.to_rgb(colors.CSS4_COLORS["dimgrey"]), colors.to_rgb(colors.CSS4_COLORS["darkorange"])
    def __init__(self, width = 8):
        self.width = width
        self.upper_bound, self.right_bound = self.height-1, self.width-1
        self.render_grid = np.ones((self.height,self.width,3))
        self.render_grid[0,0:] = self.cliff_color
        self.render_grid[-1,-1] = self.goal_color
    def step(self, state, action):
        if action == 0: 
            reward = -1.; done = 1. 
        elif action == 1:
            state += 1; reward = 2.
            done = 0. if state < self.right_bound else 1.
        else: assert False, "the action is invalid!"
        if done: # automatically reset
            state = self.reset()
        return state, reward, done
    def reset(self):
        return 0
    def flatten_state_action(self, state, action):
        return state*self.num_action + action
    def get_random(self):
        state_action = random.randrange((self.width-1)*self.num_action)
        state, action = state_action // self.num_action, state_action % self.num_action
        return state, action
    def render(self, q_table, figsize, name = ""):
        plt.figure(figsize=figsize)
        ax = plt.gca()
        ax.imshow(self.render_grid, extent = (0, self.width, self.height,0), interpolation = "none" )
        ax.grid(color='black', linewidth = 2.5)
        ax.set_xticks(list(range(0, self.width+1)))
        ax.set_yticks(list(range(0, self.height+1)))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_frame_on(True)
        actions = self.to_action_table(q_table)
        for j in range(self.width-1):
            i = 1
            displacement = 0.4
            qs = q_table[i]
            max_q = max(qs)
            actions = [idx for idx, q in enumerate(qs) if abs(q-max_q) < 1e-10]
            for action in actions:
                center = (j+0.5, i+0.5)
                if action == 0: x_y_dx_dy = (center[0], center[1]+0.15, 0, -1.*displacement) 
                elif action == 1: x_y_dx_dy = (center[0]-0.15, center[1], 1.*displacement, 0) 
                ax.arrow(*x_y_dx_dy, width = 0.05, head_width = 0.15, head_length = 0.15, length_includes_head = True, color = "tab:red")
        ax.text(self.width-0.5, self.height-0.5, "Goal", horizontalalignment = "center", verticalalignment = "center", fontsize = 13)
        plt.tight_layout()
        plt.savefig(datetime.datetime.now().strftime("%m-%d %H:%M:%S") + name + ".pdf") # the file name
        plt.close()
    def to_action_table(self, q_table):
        q_table = np.array(q_table).reshape(self.width-1, self.num_action)
        return np.argmax(q_table, axis = 1)
    def loss_bellman(self, q_table, gamma): 
        loss = 0. 
        for state, qs in enumerate(q_table): 
            for action, q in enumerate(qs): 
                next_state, reward, done = self.step(state, action) 
                target_q_value = reward + (1.-done) * gamma  * max(q_table[next_state]) 
                loss += (q - target_q_value)**2
        if not loss < 1e20: assert False, "The optimization diverged"
        return loss
    def greedy_total_reward(self, q_table):
        state = self.reset()
        total_reward, done = 0., 0.
        while not done:
            # to keep a memory of states that have been previously seen
            optimal_q = max(q_table[state])
            optimal_actions = [idx for idx, q in enumerate(q_table[state]) if abs(q-optimal_q) < 1e-10]
            if len(optimal_actions) == 1: action = optimal_actions[0]
            else: action = random.choice(optimal_actions) 
            next_state, reward, done = self.step(state, action) 
            # we reject loops and assign a negative reward equal to the cliff
            total_reward += reward
            state = next_state
        return total_reward



def q_step(q_table, state, action, reward, next_state, next_actions, next_coefficient, lr = 0.5):
    next_action = next_actions[0]
    td_error = next_coefficient*q_table[next_state][next_action] + reward - q_table[state][action]
    loss = td_error**2
    q_table[state][action] += lr*td_error
    return loss

def rg_step(q_table, state, action, reward, next_state, next_actions, next_coefficient, lr = 0.5):
    td_error = next_coefficient*q_table[next_state][next_actions[0]] + reward - q_table[state][action]
    loss = td_error**2
    q_table[state][action] += lr*td_error
    num_of_maximal_next_actions = len(next_actions)
    for next_action in next_actions: # in order to avoid disturbing the policy, we equally modify the actions that have the same Q value 
        q_table[next_state][next_action] -= lr/num_of_maximal_next_actions*next_coefficient*td_error
    return loss

def train_with_state(state, q_table, grid, strategy, gamma = 0.99, lr = 0.5, epsilon = 0.):
    if epsilon>0. and random.random() < epsilon:
        action = random.randrange(grid.num_action)
    else:
        optimal_q = max(q_table[state])
        optimal_actions = [idx for idx, q in enumerate(q_table[state]) if abs(q-optimal_q) < 1e-10]
        if len(optimal_actions) == 1: action = optimal_actions[0]
        else: action = random.choice(optimal_actions) 
    next_state, reward, done = grid.step(state, action)
    next_coefficient = (1.-done) * gamma
    next_q = max(q_table[next_state])
    next_actions = [idx for idx, q in enumerate(q_table[next_state]) if abs(q-next_q) < 1e-10]
    strategy(q_table, state, action, reward, next_state, next_actions, next_coefficient, lr = lr)
    return next_state, reward, done

def train_rand(q_table, grid, strategy, gamma = 0.99, lr = 0.5):
    state, action = grid.get_random()
    next_state, reward, done = grid.step(state, action)
    next_coefficient = (1.-done) * gamma
    next_action = np.argmax(q_table[next_state])
    strategy(q_table, state, action, reward, next_state, [next_action], next_coefficient, lr = lr)

grid = GridWorld(width=8)
gamma = 1. 
q_table = [[0. for j in range(grid.num_action)] for i in range(grid.width-1)]
for i in range(100000): # 30000
    if i % 1000 == 0:
        loss = grid.loss_bellman(q_table, gamma = gamma)
        if loss < 1e-25: break 
    train_rand(q_table, grid, q_step, gamma = gamma) 

truth = q_table
grid.render(truth, figsize=(5,2), name = "optimal")





num_of_trials = 100
q_performance = [[] for i in range(num_of_trials)]  
q_x = [[] for i in range(num_of_trials)]
for j in range(num_of_trials):
    q_table = [[0. for a in range(grid.num_action)] for i in range(grid.width-1)]
    log_scale_x_axis = 0.
    data_x_coordinate = 10 ** log_scale_x_axis
    state = grid.reset()
    for i in range(1, 10000+1):
        state, reward, done = train_with_state(state, q_table, grid, q_step, gamma = gamma, lr = 0.5, epsilon = 0.) 
        if i >= data_x_coordinate:
            q_performance[j].append(grid.greedy_total_reward(q_table))
            q_x[j].append(i)
            while i >= data_x_coordinate:
                log_scale_x_axis += 0.005
                data_x_coordinate = 10 ** log_scale_x_axis

q_performance = np.array(q_performance).mean(axis=0) 
q_x = q_x[0] 

num_of_trials = 100
rg_performance = [[] for i in range(num_of_trials)]  
rg_x = [[] for i in range(num_of_trials)]
_time = time.time()
for j in range(num_of_trials):
    q_table = [[0. for a in range(grid.num_action)] for i in range(grid.width-1)]
    log_scale_x_axis = 0.
    data_x_coordinate = 10 ** log_scale_x_axis
    #accu_performance = 0. 
    #ep_reward = 0.
    #accu_i = 0
    #last_x = 0
    state = grid.reset()
    for i in range(1, 100000+1): # 50000000
        state, reward, done = train_with_state(state, q_table, grid, rg_step, gamma = gamma, lr = 0.5, epsilon = 0.) 
        #ep_reward += reward 
        #if done:
        #    accu_performance += ep_reward; accu_i += 1
        #    ep_reward = 0. 
        if i % 1000000 == 0: print(j,i)
        if i >= data_x_coordinate:
            rg_performance[j].append(grid.greedy_total_reward(q_table))
            rg_x[j].append(i)
            while i >= data_x_coordinate:
                log_scale_x_axis += 0.005
                data_x_coordinate = 10 ** log_scale_x_axis
        #if i % 10 == 0:
            #rg_performance[j].append(accu_performance/accu_i) 
            #accu_performance = 0.; accu_i = 0
            #rg_x[j].append((i+last_x)/2.)
            #last_x = i
    print("time cost: {}".format(time.time()-_time))
    _time = time.time()

rg_performance = np.array(rg_performance).mean(axis=0) 
rg_x = rg_x[0]

rg_performance001 = [[] for i in range(num_of_trials)]  
_time = time.time()
for j in range(num_of_trials):
    q_table = [[0. for a in range(grid.num_action)] for i in range(grid.width-1)]
    log_scale_x_axis = 0.
    data_x_coordinate = 10 ** log_scale_x_axis
    #accu_performance = 0. 
    #ep_reward = 0.
    #accu_i = 0
    #last_x = 0
    state = grid.reset()
    for i in range(1, 100000+1): # 50000000
        state, reward, done = train_with_state(state, q_table, grid, rg_step, gamma = gamma, lr = 0.5, epsilon = 0.01) 
        #ep_reward += reward 
        #if done:
        #    accu_performance += ep_reward; accu_i += 1
        #    ep_reward = 0. 
        if i % 1000000 == 0: print(j,i)
        if i >= data_x_coordinate:
            rg_performance001[j].append(grid.greedy_total_reward(q_table))
            while i >= data_x_coordinate:
                log_scale_x_axis += 0.005
                data_x_coordinate = 10 ** log_scale_x_axis
        #if i % 10 == 0:
            #rg_performance001[j].append(accu_performance/accu_i) 
            #rg_performance001[j].append(grid.greedy_total_reward(q_table))
            #accu_performance = 0.; accu_i = 0
            #last_x = i
    print("time cost: {}".format(time.time()-_time))
    _time = time.time()

rg_performance001 = np.array(rg_performance001).mean(axis=0) 

rg_performance01 = [[] for i in range(num_of_trials)]  
_time = time.time()
for j in range(num_of_trials):
    q_table = [[0. for a in range(grid.num_action)] for i in range(grid.width-1)]
    log_scale_x_axis = 0.
    data_x_coordinate = 10 ** log_scale_x_axis
    #accu_performance = 0. 
    #ep_reward = 0.
    #accu_i = 0
    #last_x = 0
    state = grid.reset()
    for i in range(1, 100000+1): # 50000000
        state, reward, done = train_with_state(state, q_table, grid, rg_step, gamma = gamma, lr = 0.5, epsilon = 0.1) 
        #ep_reward += reward 
        #if done:
        #    accu_performance += ep_reward; accu_i += 1
        #    ep_reward = 0. 
        if i % 1000000 == 0: print(j,i)
        if i >= data_x_coordinate:
            rg_performance01[j].append(grid.greedy_total_reward(q_table))
            while i >= data_x_coordinate:
                log_scale_x_axis += 0.005
                data_x_coordinate = 10 ** log_scale_x_axis
        #if i % 10 == 0:
            #rg_performance01[j].append(accu_performance/accu_i) 
            #rg_performance01[j].append(grid.greedy_total_reward(q_table))
            #accu_performance = 0.; accu_i = 0
            #last_x = i
    print("time cost: {}".format(time.time()-_time))
    _time = time.time()

rg_performance01 = np.array(rg_performance01).mean(axis=0) 



plt.figure(figsize=(4,3))
plt.plot(q_x, q_performance, color = "C0", label = "Q-table $\epsilon=0$")
plt.plot(rg_x, rg_performance, color = "C1", label = "RG $\epsilon=0$")
plt.plot(rg_x, rg_performance001, color = "C1", label = "RG $\epsilon=0.01$", linestyle = "--")
plt.plot(rg_x, rg_performance01, color = "C1", label = "RG $\epsilon=0.1$", linestyle = ":")
plt.xscale("log")
plt.yscale("linear")
#plt.xlim(left=1, right=1e3)#
#plt.ylim(bottom=-2)
plt.xlabel("number of steps")
plt.ylabel("total reward")
plt.legend(loc="lower right")
plt.tight_layout()

plt.savefig("greedy performance.pdf".format(gamma))
plt.close()

