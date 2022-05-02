import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import math, os
from scipy.ndimage import gaussian_filter as gaussian
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

plt.rcParams["mathtext.fontset"]="cm"
plt.rcParams["axes.formatter.use_mathtext"]=True



algs = {"C-DQN": "CDQN", "DQN": "DQN"}
file_suffix_meaning = {"_.txt": "reward", "mse_.txt": "loss"} #

game = "SpaceInvaders"
gaussian_smoothing = {"reward": 60, "loss": 1}

plt.figure(figsize=(3,2))
for i, (suffix, ylabel) in enumerate(file_suffix_meaning.items()):
    if ylabel == "loss":
        ax = plt.gca()
        inset_axes(ax, width="100%", height="100%", bbox_to_anchor=(0.722,0.365,0.31,0.35), bbox_transform=ax.transAxes) #, loc=
        plt.yscale("log"); plt.xlim(left=3e5, right=7.3e7); plt.ylim(top=1e7, bottom=1e-3)
        plt.gca().axes.get_xaxis().set_visible(False)
        plt.gca().yaxis.set_tick_params(pad=0, labelsize=6)
        plt.ylabel(ylabel, labelpad=-3)
    else: plt.xlim(left=0, right=7.3e7); plt.ylim(top=900); plt.xlabel("frames"); plt.ylabel(ylabel)
    for alg, filename in algs.items():
        with open(os.path.join(game+"NoFrameskip-v4", filename+suffix) , 'r') as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines]
            x = np.array([float(line.split()[0]) for line in lines])
            y = np.array([float(line.split()[1]) for line in lines])
            plt.plot(x, gaussian(y, sigma=gaussian_smoothing[ylabel], mode="mirror"), label=alg) #/1e6
    if ylabel == "reward":
        plt.legend(loc="lower left")
    plt.tight_layout()

plt.savefig("{} half data discarded.pdf".format(game))
plt.close()





file_suffix_meaning = {"_RandReplace": "reward", "mse_RandReplace": "loss"} #

plt.figure(figsize=(3,2.5))
for i, (suffix, ylabel) in enumerate(file_suffix_meaning.items()):
    if ylabel == "loss":
        ax = plt.gca()
        inset_axes(ax, width="100%", height="100%", bbox_to_anchor=(0.722,0.689,0.31,0.35), bbox_transform=ax.transAxes) #, loc=
        plt.yscale("log"); plt.xlim(left=3e5, right=6e7); plt.ylim(top=1e7, bottom=1e-3)
        plt.gca().axes.get_xaxis().set_visible(False)
        plt.gca().yaxis.set_tick_params(pad=0, labelsize=6)
        plt.ylabel(ylabel, labelpad=-3)
    else: plt.xlim(left=0, right=6e7); plt.ylim(top=900); plt.xlabel("frames"); plt.ylabel(ylabel)
    with open(os.path.join(game+"NoFrameskip-v4", "DQN"+suffix+".txt") , 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
        x = np.array([float(line.split()[0]) for line in lines])
        y = np.array([float(line.split()[1]) for line in lines])
        plt.plot(x, gaussian(y, sigma=gaussian_smoothing[ylabel], mode="mirror"), label="DQN", color="C1") #/1e6
    if ylabel == "reward":
        plt.legend(loc="lower left")
    plt.tight_layout()

plt.savefig("{} random memory replacement.pdf".format(game))
plt.close()




gaussian_smoothing = {"reward": 60, "loss": 2}

algs = {"C-DQN random": ("CDQN", "smallerRandReplace"), "DQN random": ("DQN", "smallerRandReplace"), "DQN FIFO": ("DQN", "smallerFIFO")}
file_suffix_meaning = {"_": "reward", "mse_": "loss"} #

plt.figure(figsize=(6,2.5))
for i, (suffix, ylabel) in enumerate(file_suffix_meaning.items()):
    plt.subplot(1,2,i+1)
    if ylabel == "loss":
        ax = plt.gca()
        plt.xlim(left=3e5, right=1e8)#; plt.ylim(top=1e7, bottom=1e-3)
    else: plt.xlim(left=0, right=1e8); #; plt.ylim(top=900)
    for alg, (filename_pre, filename_suf) in algs.items():
        with open(os.path.join(game+"NoFrameskip-v4", filename_pre+suffix+filename_suf+".txt") , 'r') as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines]
            x = np.array([float(line.split()[0]) for line in lines])
            y = np.array([float(line.split()[1]) for line in lines])
            plt.plot(x, gaussian(y, sigma=gaussian_smoothing[ylabel], mode="mirror"), label=alg) #/1e6
    plt.xlabel("frames")
    if ylabel == "reward":
        plt.legend(loc="upper left")
        plt.ylabel(ylabel, labelpad=-3)
    else:
        plt.ylabel(ylabel, labelpad=1)
    plt.tight_layout()

plt.subplots_adjust(wspace=0.34)
plt.savefig("{} smaller memory.pdf".format(game))
plt.close()


