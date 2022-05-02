import matplotlib.pyplot as plt
import matplotlib
import matplotlib.gridspec as gridspec
import numpy as np
from math import sqrt

plt.style.use('seaborn-whitegrid')
matplotlib.rc('legend', frameon=True, fontsize='medium', loc='upper left', framealpha=0.6)
#matplotlib.rc('text', usetex=True)
matplotlib.rc('font', family='serif', serif='CMU Serif', monospace='Computer Modern Typewriter', size=14)

grid1 = gridspec.GridSpec(nrows=3, ncols=1, hspace=0.1, height_ratios=[4, 1.2, 4]) #, bottom=0.45
#grid2 = gridspec.GridSpec(nrows=2, ncols=1, hspace=0., height_ratios=[3, 1.2], top=0.42)

def set_parameters(**kwargs):
    global x, x_max, dt, num_of_episodes, probability, reward_multiply, read_length, controls_per_half_period
    x = kwargs["x"]
    x_max = kwargs["x_max"]
    dt = kwargs["dt"]
    num_of_episodes = kwargs["num_of_episodes"]
    probability = kwargs["probability"]
    reward_multiply = kwargs["reward_multiply"]
    read_length = kwargs["read_length"]
    controls_per_half_period = kwargs["controls_per_half_period"]

class Plot(object):
    def __init__(self, time_of_episode):
        self.t_max = time_of_episode
        plt.close()
        plt.ion()
        self.figure = plt.figure(figsize=(6.5,8.5))
        # the wave function plot
        self.ax1 = plt.subplot(grid1[0], xlim=(-x_max, x_max),
            ylim = (-0.75,0.75), yticks=[-0.6,-0.3,0,0.3,0.6])
        ax1 = self.ax1
        # the density plot
        self.ax2 = plt.subplot(grid1[1], sharex=ax1, ylim = (-0.08,0.8))
        ax2 = self.ax2
        ax2.tick_params(axis='y', labelleft=False)
        # the stochastic measurement plot
        self.ax3 = plt.subplot(grid1[2], ylim=(-8, 8), xlim = (0,6))
        ax3 = self.ax3
        ax3.set_ylabel(r'$\bf{x}$', rotation=0)
        ax3.set_xlabel(r'$\bf{t}$', rotation=0)
        # the RL training plot with training time
        #self.ax4 = plt.subplot(grid2[0], xlim=(0.,num_of_episodes), ylim = (-16,0.35))
        #ax4 = self.ax4
        #ax4.yaxis.get_ticklabels()[0].set_visible(False)
        # the RL training loss
        #self.ax5 = plt.subplot(grid2[1], sharex=ax4, xlabel='episodes', ylim = (0,5.5))
        #ax5 = self.ax5
        # adjust the layout
        self.figure.subplots_adjust(top=0.95,bottom=0.05,left=0.10,right=0.97)
        ax1.tick_params(axis='x', labeltop=True)
        ax3.tick_params(axis='x', labelbottom=True)
        #ax4.tick_params(axis='x', labelbottom=False)
        # the real part of wave
        self.line11, = ax1.plot(x, np.zeros_like(x), zorder = 2., label=r'Re$(\psi)$')
        # the imaginary
        self.line12, = ax1.plot(x, np.zeros_like(x), zorder = 1., label=r'Im$(\psi)$')
        # the potential
        self.line13, = ax1.plot(x, np.zeros_like(x), zorder = 0., color = 'lightslategray', label=r'$V$')
        # the density
        self.line21, = ax2.plot(x, np.zeros_like(x), zorder = 1., color = 'firebrick', label=r'$|\psi|^2$')
        # potential
        self.line22, = ax2.plot(x, np.zeros_like(x), zorder = 0., color = 'lightslategray', label=r'$V$')
        # the stochastic measurements
        #self.line31, = ax3.plot([],[], linestyle='', marker='o',
        #    markeredgewidth=0.,markersize=3., markerfacecolor='tomato',alpha=0.6, zorder = 0., label='measurements')
        # the controlled potential minimum position
        self.line32, = ax3.plot([],[], color = 'lightslategray', zorder = 1., linewidth = 1.5, label=r'$F_{con}$')
        # the x position expectation
        self.line33, = ax3.plot([],[], color = 'darkorange', zorder = 2., linewidth = 1.5, label=r'$\langle\hat{x}\rangle$')
        # set the time discretization coordinates
        self.graph3_xdata = np.arange(0, self.ax3.get_xlim()[1], step=dt)#[::-1]
        self.graph3_num = self.graph3_xdata.size
        #self.line31.set_ydata(self.graph3_ydata)
        self.line32.set_xdata(self.graph3_xdata)
        self.line33.set_xdata(self.graph3_xdata)
        graph3_empty = [None for i in range(self.graph3_num)]
        #self.line31.set_xdata(graph3_empty)
        self.line32.set_ydata(graph3_empty)
        self.line33.set_ydata(graph3_empty)
        # the RL random action events
        #self.line44, = ax4.plot([],[], linestyle='',marker='o',
        #    markeredgewidth=0.,markersize=3., markerfacecolor='red', zorder = 3., label='rnd action')
        #self.rnd_actions = []
        # the RL results
        #self.line41, = ax4.plot([],[], color = 'cornflowerblue', zorder = 0., label='final <n>s')
        #self.results = []
        # the RL reward
        #self.line42, = ax4.plot([],[], color = 'darkturquoise', zorder = 1., linewidth = 1.5, label='<n>')
        #self.rewards = []
        # the RL estimated value
        #self.line43, = ax4.plot([],[], color = 'gold', zorder = 2., linewidth = 1.5, label='value')
        #self.estimations = []
        #self.graph4_xdata = np.linspace(0, num_of_episodes/1.5, num = controls_per_half_period*(self.t_max+8.))
        # the RL loss
        #self.line51, = ax5.plot([],[], color = 'cornflowerblue', zorder = 0., label='loss')
        #self.loss = [] 
        for i in range(1,4): # (1, 6):
            self.__dict__['ax'+str(i)].legend()
        self.lines = [self.line11, self.line12, self.line13, self.line21, self.line22,
            #self.line31, 
            self.line32, self.line33#, self.line41, self.line42, self.line43, self.line44, self.line51
            ]
    def __del__(self):
        plt.close(self.figure)
        plt.ioff()
    ######
    def store_RL_stats(self, reward, estimation, rnd_action):
        # do the statistics of RL agent separately since this is on a separate graph and coordinate
        self.rewards.append(reward/reward_multiply)
        self.estimations.append(estimation/reward_multiply) if estimation != None else self.estimations.append(None)
        self.rnd_actions.append(0.2) if rnd_action else self.rnd_actions.append(None)
    def finish(self, loss):
        self.results.append(self.rewards[-1])
        self.line41.set_xdata(np.arange(len(self.results)))
        self.line41.set_ydata(self.results)
        self.loss.append(loss)
        self.line51.set_xdata(np.arange(len(self.loss)))
        self.line51.set_ydata(self.loss)
        self.rewards = []
        self.estimations = []
        self.rnd_actions = []
        stop_point = min(num_of_episodes, len(self.results)+num_of_episodes/1.5)
        self.graph4_xdata = np.linspace(len(self.results), stop_point, num = controls_per_half_period*self.t_max+1)
    ######
    def __call__(self, state, x_mean, forces, choice = None): #, measurements
        # plot potential
        potential=1/2*x*x*x*x/25+x*forces[-1]+1.74*(forces[-1]**4)**(1/3)
        self.line13.set_ydata(potential/110. - 0.7)
        self.line22.set_ydata(potential/55.)
        # plot wave function
        self.line12.set_ydata(np.imag(state))
        self.line11.set_ydata(np.real(state))
        # plot density distribution
        self.line21.set_ydata(probability(state))
        # finish plotting first two graphs
        # plot stochastic measurements and control
        if len(x_mean) >= self.graph3_num:
            #measurements[measurements==0.]=None
            #self.line31.set_ydata(self.graph3_ydata)
            self.line32.set_xdata(self.graph3_xdata)
            self.line33.set_xdata(self.graph3_xdata)
            #self.line31.set_xdata(measurements[-self.graph3_num:])
            self.line32.set_ydata(np.array(forces[-self.graph3_num:]))
            self.line33.set_ydata(x_mean[-self.graph3_num:])
        else:
            length = len(x_mean)
            #measurements[measurements==0.]=None
            #self.line31.set_ydata(self.graph3_ydata[-length:])
            self.line32.set_xdata(self.graph3_xdata[-length:])
            self.line33.set_xdata(self.graph3_xdata[-length:])
            #self.line31.set_xdata(measurements)
            self.line32.set_ydata(np.array(forces[-length:]))
            self.line33.set_ydata(x_mean[-length:])
        # plot the RL process graph
        #self.line42.set_xdata(self.graph4_xdata[:len(self.rewards)])
        #self.line43.set_xdata(self.graph4_xdata[:len(self.estimations)])
        #self.line44.set_xdata(self.graph4_xdata[:len(self.rnd_actions)])
        #self.line42.set_ydata(self.rewards)
        #self.line43.set_ydata(self.estimations)
        #self.line44.set_ydata(self.rnd_actions)
        self.figure.canvas.draw()
        plt.pause(0.002)
