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
    height = 4 
    num_action = 4
    action_meaning = {0: "up", 1: "down", 2: "right", 3: "left"}
    cliff_color, goal_color = colors.to_rgb(colors.CSS4_COLORS["dimgrey"]), colors.to_rgb(colors.CSS4_COLORS["darkorange"])
    def __init__(self, width = 10):
        self.width = width
        self.low_bound, self.right_bound = self.height-1, self.width-1
        self.render_grid = np.ones((self.height,self.width,3))
        self.render_grid[-1,1:] = self.cliff_color
        self.render_grid[-1,-1] = self.goal_color
    def step(self, state, action):
        if action == 0 and state[0] > 0: state = (state[0]-1, state[1])
        elif action == 1 and state[0] < self.low_bound: state = (state[0]+1, state[1])
        elif action == 2 and state[1] < self.right_bound: state = (state[0], state[1]+1)
        elif action == 3 and state[1] > 0: state = (state[0], state[1]-1)
        else: assert False, "the action is invalid!"
        # prepare reward and done
        done = 0.
        reward = -1.
        # only if the state is at the lowest row, done may be True
        if state[0] == self.low_bound:
            if state[1] > 0:
                done = 1.
                reward = 0. if state[1] == self.right_bound else -100.
        return state, reward, done
    def reset(self):
        return (self.height-1, 0)
    def action_invalid(self, state, action):
        return (action == 0 and state[0] == 0) or (action == 1 and state[0] == self.low_bound) or (action == 2 and state[1] == self.right_bound) or (action == 3 and state[1] == 0)
    def flatten_state(self, state):
        return state[0]*self.width + state[1]
    def flatten_state_action(self, state, action):
        return state[0]*self.width*self.num_action + state[1]*self.num_action + action
    def unflatten_state(self, state):
        return state // self.width, state % self.width
    def get_random(self):
        state_action = random.randrange(self.height*self.width*self.num_action)
        state, action = state_action // self.num_action, state_action % self.num_action
        state = state // self.width, state % self.width
        if not ((state[0] == self.low_bound and state[1] > 0) or self.action_invalid(state, action)): return state, action
        else: return self.get_random() # resample by recursively calling itself
    def render(self, q_table, name = ""):
        fig, ax = plt.subplots()
        ax.imshow(self.render_grid, extent = (0, self.width, self.height,0), interpolation = "none" )
        ax.grid(color='black', linewidth = 2.5)
        ax.set_xticks(list(range(0, self.width+1)))
        ax.set_yticks(list(range(0, self.height+1)))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_frame_on(True)
        actions = self.to_action_table(q_table)
        for i in range(self.height):
            for j in range(self.width):
                if i == self.low_bound and j > 0: continue
                displacement = 0.4
                qs = q_table[self.flatten_state((i,j))]
                max_q = max(qs)
                # note that variable names 'i' and 'j' are already in use
                actions = [idx for idx, q in enumerate(qs) if abs(q-max_q) < 1e-10]
                for action in actions:
                    center = (j+0.5, i+0.5)
                    if action == 0: x_y_dx_dy = (center[0], center[1]+0.05, 0, -1.*displacement) # +displacement
                    elif action == 1: x_y_dx_dy = (center[0], center[1]-0.05, 0, 1.*displacement) # -displacement
                    elif action == 2: x_y_dx_dy = (center[0]-0.05, center[1], 1.*displacement, 0) # -displacement
                    elif action == 3: x_y_dx_dy = (center[0]+0.05, center[1], -1.*displacement, 0) # +displacement
                    ax.arrow(*x_y_dx_dy, width = 0.05, head_width = 0.15, head_length = 0.15, length_includes_head = True, color = "tab:red")
        ax.text(self.width-0.5, self.height-0.5, "Goal", horizontalalignment = "center", verticalalignment = "center", fontsize = 13)
        plt.tight_layout()
        plt.savefig(datetime.datetime.now().strftime("%m-%d %H:%M:%S") + name + ".pdf") # the file name
        plt.close()
    def to_action_table(self, q_table):
        q_table = np.array(q_table).reshape(self.height, self.width, self.num_action)
        return np.argmax(q_table, axis = 2)
    def loss_to_truth(self, q_table, truth):
        loss = 0.
        for state, qs in enumerate(q_table):  
            unflattened_state = self.unflatten_state(state)
            if unflattened_state[0] == self.low_bound and unflattened_state[1] > 0 : continue
            for action, q in enumerate(qs):
                if self.action_invalid(unflattened_state, action): continue
                else: loss += (q - truth[state][action])**2
        return loss
    def policy_diff_truth(self, q_table, truth):
        n, n_wrong = 0, 0
        for state, qs in enumerate(q_table):  
            unflattened_state = self.unflatten_state(state)
            if unflattened_state[0] == self.low_bound and unflattened_state[1] > 0 : continue
            n += 1
            # find the optimal actions 
            optimal_q = max(truth[state])
            optimal_actions = [idx for idx, q in enumerate(truth[state]) if abs(q-optimal_q) < 1e-10]
            # find the actions of the current q_table
            max_q = max(qs)
            current_actions = [idx for idx, q in enumerate(qs) if abs(q-max_q) < 1e-10]
            num_action = len(current_actions)
            for action in current_actions:
                if action not in optimal_actions:
                    n_wrong += 1./num_action
        return n_wrong/n 
    def loss_bellman(self, q_table, gamma): 
        loss = 0. 
        for state, qs in enumerate(q_table):  
            state = self.unflatten_state(state)  
            for action, q in enumerate(qs): 
                if (state[0] == self.low_bound and state[1] > 0) or self.action_invalid(state, action): continue
                next_state, reward, done = self.step(state, action) 
                target_q_value = reward + (1.-done) * gamma  * max(q_table[self.flatten_state(next_state)]) 
                loss += (q - target_q_value)**2
        if not loss < 1e20: assert False, "The optimization diverged"
        return loss
    def greedy_total_reward(self, q_table):
        previous_states = []
        state = self.reset()
        total_reward, done = 0., 0.
        while not done:
            # to keep a memory of states that have been previously seen
            previous_states.append(state)
            flattened_state = grid.flatten_state(state)
            optimal_q = max(q_table[flattened_state])
            optimal_actions = [idx for idx, q in enumerate(q_table[flattened_state]) if abs(q-optimal_q) < 1e-10]
            if len(optimal_actions) == 1: action = optimal_actions[0]
            else: action = random.choice(optimal_actions) 
            next_state, reward, done = self.step(state, action) 
            # we reject loops and assign a negative reward equal to the cliff
            if next_state in previous_states: 
                reward = -100.; done = 1.
            total_reward += reward
            state = next_state
        return total_reward
    def disable_invalid_action_q(self, q_table):
        for state, qs in enumerate(q_table):   
            for action, q in enumerate(qs): 
                if self.action_invalid(self.unflatten_state(state), action):
                    q_table[state][action] = -float("inf")
        return q_table



def q_step(q_table, states, actions, rewards, next_states, next_actions, next_coefficients, lr = 0.5):
    td_errors = []
    loss = 0.
    for state, action, reward, next_state, next_action, next_coefficient in zip(states, actions, rewards, next_states, next_actions, next_coefficients):
        td_error = next_coefficient*q_table[next_state][next_action] + reward - q_table[state][action]
        loss += td_error**2
        td_errors.append(td_error)
    batch_size = len(states)
    for td_error, state, action, reward, next_state, next_action, next_coefficient in zip(td_errors, states, actions, rewards, next_states, next_actions, next_coefficients):
        q_table[state][action] += lr/batch_size*td_error
    return loss/batch_size

def rg_step(q_table, states, actions, rewards, next_states, next_actions, next_coefficients, lr = 0.5):
    td_errors = []
    loss = 0.
    for state, action, reward, next_state, next_action, next_coefficient in zip(states, actions, rewards, next_states, next_actions, next_coefficients):
        td_error = next_coefficient*q_table[next_state][next_action] + reward - q_table[state][action]
        loss += td_error**2
        td_errors.append(td_error)
    batch_size = len(states)
    for td_error, state, action, reward, next_state, next_action, next_coefficient in zip(td_errors, states, actions, rewards, next_states, next_actions, next_coefficients):
        q_table[state][action] += lr/batch_size*td_error; q_table[next_state][next_action] -= lr/batch_size*next_coefficient*td_error
    return loss/batch_size

def train_rand(q_table, grid, strategy, gamma = 0.99, lr = 0.5, batch_size = 1):
    states, actions, next_states, next_actions, next_coefficients, rewards = [], [], [], [], [], []
    for i in range(batch_size): 
        state, action = grid.get_random()
        next_state, reward, done = grid.step(state, action)
        next_coefficient = (1.-done) * gamma
        state, next_state = grid.flatten_state(state), grid.flatten_state(next_state)
        next_action = np.argmax(q_table[next_state])
        states.append(state), actions.append(action), next_states.append(next_state), next_actions.append(next_action), next_coefficients.append(next_coefficient), rewards.append(reward)
    strategy(q_table, states, actions, rewards, next_states, next_actions, next_coefficients, lr = lr)

grid = GridWorld(width=10)
gamma = 0.9 
q_table = grid.disable_invalid_action_q([[0. for j in range(grid.num_action)] for i in range(grid.height * grid.width)])
for i in range(100000): # 30000
    if i % 1000 == 0:
        loss = grid.loss_bellman(q_table, gamma = gamma)
        if loss < 1e-25: break 
    train_rand(q_table, grid, q_step, gamma = gamma) 

truth = q_table
grid.render(truth, name = "optimal")

num_of_trials = 10
q_MSBE_loss, q_true_loss, q_policy_error = [[] for i in range(num_of_trials)], [[] for i in range(num_of_trials)], [[] for i in range(num_of_trials)]  
q_x = [[] for i in range(num_of_trials)]
for j in range(num_of_trials):
    q_table = grid.disable_invalid_action_q([[0. for a in range(grid.num_action)] for i in range(grid.height * grid.width)])
    log_scale_x_axis = 0.
    data_x_coordinate = 10 ** log_scale_x_axis
    for i in range(1, 10000+1):
        train_rand(q_table, grid, q_step, gamma = gamma, lr = 0.5, batch_size = 1) 
        if i >= data_x_coordinate:
            loss = grid.loss_bellman(q_table, gamma = gamma)
            q_MSBE_loss[j].append(loss); q_true_loss[j].append(grid.loss_to_truth(q_table, truth)); q_policy_error[j].append(grid.policy_diff_truth(q_table, truth))
            q_x[j].append(i)
            while i >= data_x_coordinate:
                log_scale_x_axis += 0.005
                data_x_coordinate = 10 ** log_scale_x_axis

length = min(*[len(l) for l in q_MSBE_loss])
q_MSBE_loss, q_true_loss, q_policy_error = [l[:length] for l in q_MSBE_loss], [l[:length] for l in q_true_loss], [l[:length] for l in q_policy_error] 
q_MSBE_loss, q_true_loss, q_policy_error = np.array(q_MSBE_loss).mean(axis=0), np.array(q_true_loss).mean(axis=0), np.array(q_policy_error).mean(axis=0)
q_x = q_x[0][:length]
for q_length, l in enumerate(q_MSBE_loss):
    if l < 1e-5: break

q_x = q_x[:q_length]
q_MSBE_loss, q_true_loss, q_policy_error = q_MSBE_loss[:q_length], q_true_loss[:q_length], q_policy_error[:q_length]


num_of_trials = 10
rg_MSBE_loss, rg_true_loss, rg_policy_error = [[] for i in range(num_of_trials)], [[] for i in range(num_of_trials)], [[] for i in range(num_of_trials)] 
rg_x = [[] for i in range(num_of_trials)]
_time = time.time()
for j in range(num_of_trials):
    q_table = grid.disable_invalid_action_q([[0. for a in range(grid.num_action)] for i in range(grid.height * grid.width)])
    log_scale_x_axis = 0.
    data_x_coordinate = 10 ** log_scale_x_axis
    for i in range(1, 400000+1): # 50000000
        train_rand(q_table, grid, rg_step, gamma = gamma, lr = 0.5, batch_size = 1) 
        if i % 1000000 == 0: print(j,i)
        if i >= data_x_coordinate:
            loss = grid.loss_bellman(q_table, gamma = gamma)
            rg_MSBE_loss[j].append(loss); rg_true_loss[j].append(grid.loss_to_truth(q_table, truth)); rg_policy_error[j].append(grid.policy_diff_truth(q_table, truth)) 
            rg_x[j].append(i)
            while i >= data_x_coordinate:
                log_scale_x_axis += 0.005
                data_x_coordinate = 10 ** log_scale_x_axis
    print("time cost: {}".format(time.time()-_time))
    _time = time.time()

length = min(*[len(l) for l in rg_MSBE_loss])
rg_MSBE_loss, rg_true_loss, rg_policy_error = [l[:length] for l in rg_MSBE_loss], [l[:length] for l in rg_true_loss], [l[:length] for l in rg_policy_error] 
rg_MSBE_loss, rg_true_loss, rg_policy_error = np.array(rg_MSBE_loss).mean(axis=0), np.array(rg_true_loss).mean(axis=0), np.array(rg_policy_error).mean(axis=0)

rg_x = rg_x[0][:length]
for rg_length, l in enumerate(rg_MSBE_loss):
    if l < 1e-5: break

rg_x = rg_x[:rg_length]
rg_MSBE_loss, rg_true_loss, rg_policy_error = rg_MSBE_loss[:rg_length], rg_true_loss[:rg_length], rg_policy_error[:rg_length]

plt.figure(figsize=(8,3))
plt.subplot(1,2,1)
plt.plot(q_x, q_MSBE_loss, color = "C0", label = "MSBE of Q-table")
plt.plot(q_x, q_true_loss, color = "C0", linestyle = "--", label = "$|Q-Q^*|^2$ of Q-table")
plt.plot(rg_x, rg_MSBE_loss, color = "C1", label = "MSBE of RG")
plt.plot(rg_x, rg_true_loss, color = "C1", linestyle = "--", label = "$|Q-Q^*|^2$ of RG")
plt.xscale("log")
plt.yscale("log")
plt.xlim(left=10, right=5e5)#
plt.ylim(bottom=1e-1)
plt.xlabel("update steps")
#plt.ylabel("loss")
plt.legend(loc="lower left")
plt.tight_layout()

plt.subplot(1,2,2)
plt.scatter(q_MSBE_loss, q_true_loss, marker = "+", s=20, label = "Q-table")
plt.scatter(rg_MSBE_loss, rg_true_loss, marker = "x", s=20, label = "RG")
plt.xscale("log")
plt.yscale("log")
plt.xlim(left = 1e-2)
plt.ylim(bottom = 1e-2)
plt.legend()
plt.xlabel("MSBE")
plt.ylabel("$|Q-Q^*|^2$")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("cliffWalking{}.pdf".format(gamma))
plt.close()


plt.scatter(q_true_loss, q_policy_error, marker = "+", label = "Q Learning")
plt.scatter(rg_true_loss, rg_policy_error, marker = "x", label = "Residual Gradient Q Learning")
plt.xscale("log")
plt.yscale("linear")
plt.xlim(left = 1e-2, right = 2e5)
plt.legend()
plt.xlabel("$|Q-Q^*|^2$")
plt.ylabel("Policy Error")
plt.tight_layout()
plt.savefig("ideal_policy_quality{}.pdf".format(gamma))
plt.close()

plt.scatter(q_MSBE_loss, q_policy_error, marker = "+", label = "Q Learning")
plt.scatter(rg_MSBE_loss, rg_policy_error, marker = "x", label = "Residual Gradient Q Learning")
plt.xscale("log")
plt.yscale("linear")
plt.xlim(left = 1e-2)
plt.legend()
plt.xlabel("MSBE")
plt.ylabel("Policy Error")
plt.tight_layout()
plt.savefig("policy_quality{}.pdf".format(gamma))
plt.close()

plt.plot(q_x, q_policy_error, color = "C0", label = "Q learning")
plt.plot(rg_x, rg_policy_error, color = "C1", label = "RG original")
plt.xscale("log")
plt.yscale("linear")
plt.xlim(left=1, right=5e7)
#plt.ylim(bottom=1e-1)
plt.xlabel("update steps")
plt.ylabel("policy error")
plt.legend()
plt.tight_layout()

plt.savefig("policy_error{}.pdf".format(gamma))
plt.close()




grid = GridWorld(width=20)
gamma = 0.95
q_table = grid.disable_invalid_action_q([[0. for j in range(grid.num_action)] for i in range(grid.height * grid.width)])
for i in range(100000): # 30000
    if i % 1000 == 0:
        loss = grid.loss_bellman(q_table, gamma = gamma)
        if loss < 1e-25: break 
    train_rand(q_table, grid, q_step, gamma = gamma) 

truth = q_table

num_of_trials = 10
q_MSBE_loss, q_true_loss, q_policy_error = [[] for i in range(num_of_trials)], [[] for i in range(num_of_trials)], [[] for i in range(num_of_trials)]  
q_x = [[] for i in range(num_of_trials)]
for j in range(num_of_trials):
    q_table = grid.disable_invalid_action_q([[0. for a in range(grid.num_action)] for i in range(grid.height * grid.width)])
    log_scale_x_axis = 0.
    data_x_coordinate = 10 ** log_scale_x_axis
    for i in range(1, 20000+1):
        train_rand(q_table, grid, q_step, gamma = gamma, lr = 0.5, batch_size = 1) 
        if i >= data_x_coordinate:
            loss = grid.loss_bellman(q_table, gamma = gamma)
            q_MSBE_loss[j].append(loss); q_true_loss[j].append(grid.loss_to_truth(q_table, truth)); q_policy_error[j].append(grid.policy_diff_truth(q_table, truth))
            q_x[j].append(i)
            while i >= data_x_coordinate:
                log_scale_x_axis += 0.005
                data_x_coordinate = 10 ** log_scale_x_axis

length = min(*[len(l) for l in q_MSBE_loss])
q_MSBE_loss, q_true_loss, q_policy_error = [l[:length] for l in q_MSBE_loss], [l[:length] for l in q_true_loss], [l[:length] for l in q_policy_error] 
q_MSBE_loss, q_true_loss, q_policy_error = np.array(q_MSBE_loss).mean(axis=0), np.array(q_true_loss).mean(axis=0), np.array(q_policy_error).mean(axis=0)
q_x = q_x[0][:length]
for q_length, l in enumerate(q_MSBE_loss):
    if l < 1e-5: break

q_x = q_x[:q_length]
q_MSBE_loss, q_true_loss, q_policy_error = q_MSBE_loss[:q_length], q_true_loss[:q_length], q_policy_error[:q_length]


num_of_trials = 10
rg_MSBE_loss, rg_true_loss, rg_policy_error = [[] for i in range(num_of_trials)], [[] for i in range(num_of_trials)], [[] for i in range(num_of_trials)] 
rg_x = [[] for i in range(num_of_trials)]
_time = time.time()
for j in range(num_of_trials):
    q_table = grid.disable_invalid_action_q([[0. for a in range(grid.num_action)] for i in range(grid.height * grid.width)])
    log_scale_x_axis = 0.
    data_x_coordinate = 10 ** log_scale_x_axis
    for i in range(1, 6000000+1): # 50000000
        train_rand(q_table, grid, rg_step, gamma = gamma, lr = 0.5, batch_size = 1) 
        if i % 1000000 == 0: print(j,i)
        if i >= data_x_coordinate:
            loss = grid.loss_bellman(q_table, gamma = gamma)
            rg_MSBE_loss[j].append(loss); rg_true_loss[j].append(grid.loss_to_truth(q_table, truth)); rg_policy_error[j].append(grid.policy_diff_truth(q_table, truth)) 
            rg_x[j].append(i)
            while i >= data_x_coordinate:
                log_scale_x_axis += 0.005
                data_x_coordinate = 10 ** log_scale_x_axis
    print("time cost: {}".format(time.time()-_time))
    _time = time.time()

length = min(*[len(l) for l in rg_MSBE_loss])
rg_MSBE_loss, rg_true_loss, rg_policy_error = [l[:length] for l in rg_MSBE_loss], [l[:length] for l in rg_true_loss], [l[:length] for l in rg_policy_error] 
rg_MSBE_loss, rg_true_loss, rg_policy_error = np.array(rg_MSBE_loss).mean(axis=0), np.array(rg_true_loss).mean(axis=0), np.array(rg_policy_error).mean(axis=0)

rg_x = rg_x[0][:length]
for rg_length, l in enumerate(rg_MSBE_loss):
    if l < 1e-5: break

rg_x = rg_x[:rg_length]
rg_MSBE_loss, rg_true_loss, rg_policy_error = rg_MSBE_loss[:rg_length], rg_true_loss[:rg_length], rg_policy_error[:rg_length]

plt.figure(figsize=(8,3))
plt.subplot(1,2,1)
plt.plot(q_x, q_MSBE_loss, color = "C0", label = "MSBE of Q-table")
plt.plot(q_x, q_true_loss, color = "C0", linestyle = "--", label = "$|Q-Q^*|^2$ of Q-table")
plt.plot(rg_x, rg_MSBE_loss, color = "C1", label = "MSBE of RG")
plt.plot(rg_x, rg_true_loss, color = "C1", linestyle = "--", label = "$|Q-Q^*|^2$ of RG")
plt.xscale("log")
plt.yscale("log")
plt.xlim(left=10, right=7e6)#
plt.ylim(bottom=1e-1)
plt.xlabel("update steps")
#plt.ylabel("loss")
plt.legend(loc="lower left")
plt.tight_layout()

plt.subplot(1,2,2)
plt.scatter(q_MSBE_loss, q_true_loss, marker = "+", s=15, label = "Q-table")
plt.scatter(rg_MSBE_loss, rg_true_loss, marker = "x", s=15, label = "RG")
plt.xscale("log")
plt.yscale("log")
plt.xlim(left = 1e-2)
plt.ylim(bottom = 1e-2)
plt.legend()
plt.xlabel("MSBE")
plt.ylabel("$|Q-Q^*|^2$")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("cliffWalking{}.pdf".format(gamma))
plt.close()


plt.scatter(q_true_loss, q_policy_error, marker = "+", label = "Q Learning")
plt.scatter(rg_true_loss, rg_policy_error, marker = "x", label = "Residual Gradient Q Learning")
plt.xscale("log")
plt.yscale("linear")
plt.xlim(left = 1e-2, right = 2e5)
plt.legend()
plt.xlabel("$|Q-Q^*|^2$")
plt.ylabel("Policy Error")
plt.tight_layout()
plt.savefig("ideal_policy_quality{}.pdf".format(gamma))
plt.close()

plt.scatter(q_MSBE_loss, q_policy_error, marker = "+", label = "Q Learning")
plt.scatter(rg_MSBE_loss, rg_policy_error, marker = "x", label = "Residual Gradient Q Learning")
plt.xscale("log")
plt.yscale("linear")
plt.xlim(left = 1e-2)
plt.legend()
plt.xlabel("MSBE")
plt.ylabel("Policy Error")
plt.tight_layout()
plt.savefig("policy_quality{}.pdf".format(gamma))
plt.close()

plt.plot(q_x, q_policy_error, color = "C0", label = "Q learning")
plt.plot(rg_x, rg_policy_error, color = "C1", label = "RG original")
plt.xscale("log")
plt.yscale("linear")
plt.xlim(left=1, right=5e7)
#plt.ylim(bottom=1e-1)
plt.xlabel("update steps")
plt.ylabel("policy error")
plt.legend()
plt.tight_layout()

plt.savefig("policy_error{}.pdf".format(gamma))
plt.close()


