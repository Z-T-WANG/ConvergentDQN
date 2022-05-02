import numpy as np
import os
from collections import deque
import math
import torch
import itertools

class Agent_History:
    def __init__(self, args, n_best = 5, num_avg_episode = 40):
        self.n_best_agents = [] # it saves data in the form of (model_dict, training performance, number of steps)
        assert n_best>0 and type(n_best)==int
        assert num_avg_episode>0 and type(num_avg_episode)==int
        self.num_of_bests = n_best
        self.recent_agents = [] # it saves agents in the form of (model_dict, number of steps)
        self.episodic_performance_previous = deque([], maxlen = num_avg_episode) # (performance, number of steps)
        self.episodic_performance_recent = [] # (performance, number of steps) 
        self.args = args
        self.num_avg_episode = num_avg_episode
    def add_agent(self, model_dict, num_step):
        if len(self.episodic_performance_previous) < self.num_avg_episode: return
        self.recent_agents.append((model_dict, num_step))

    def add_training_performance(self, performance, num_step):
        if len(self.recent_agents) == 0:
            self.episodic_performance_previous.append((performance, num_step)) 
        else:
            self.episodic_performance_recent.append((performance, num_step)) 
            # if we can evaluate the performance of the oldest agent stored in self.recent_agents
            while self.find_n_included_prev(2*self.recent_agents[0][1] - num_step) + len(self.episodic_performance_recent) >= self.num_avg_episode:
                # find average performance
                n_included_prev = self.find_n_included_prev(2*self.recent_agents[0][1] - num_step)
                #print(n_included_prev, type(n_included_prev))
                #print(len(self.episodic_performance_recent))
                #print(self.episodic_performance_previous)
                #print(self.episodic_performance_previous[-17:])
                #print(self.episodic_performance_previous[-n_included_prev:])

                p = ( sum([_p for _p, _i in itertools.islice(self.episodic_performance_previous, len(self.episodic_performance_previous)-n_included_prev, None)]) \
                     + sum([_p for _p, _i in self.episodic_performance_recent]) ) \
                      / (n_included_prev + len(self.episodic_performance_recent))
                m, s = self.recent_agents.pop(0)
                if len(self.n_best_agents) < self.num_of_bests:
                    self.n_best_agents.append((m, p, s))
                    self.n_best_agents.sort(reverse=True, key=lambda x: x[1])
                    if len(self.n_best_agents) == self.num_of_bests: self.save_agents()
                elif p > self.n_best_agents[-1][1]:
                    self.n_best_agents[-1] = (m, p, s)
                    self.n_best_agents.sort(reverse=True, key=lambda x: x[1])
                    self.save_agents()
                # move data from self.episodic_performance_recent to self.episodic_performance_previous
                if len(self.recent_agents) > 0:
                    i = self.find_idx_until(self.recent_agents[0][1])
                    self.episodic_performance_previous.extend(self.episodic_performance_recent[:i])
                    self.episodic_performance_recent = self.episodic_performance_recent[i:]
                else:
                    self.episodic_performance_previous.extend(self.episodic_performance_recent)
                    self.episodic_performance_recent.clear()

    def find_n_included_prev(self, from_step):
        i = 0
        for p, s in self.episodic_performance_previous:
            if s >= from_step: break
            i += 1
        return len(self.episodic_performance_previous) - i
    def find_idx_until(self, until_step):
        i = 0
        for p, s in self.episodic_performance_recent:
            if s > until_step: break
            i += 1
        return i 
    def save_agents(self):
        for i, (m, p, s) in enumerate(self.n_best_agents):
            torch.save((m, s), os.path.join(self.args.env, '{}_{}_{}.pth'.format(self.args.currentTask, self.args.comment, i+1)))

