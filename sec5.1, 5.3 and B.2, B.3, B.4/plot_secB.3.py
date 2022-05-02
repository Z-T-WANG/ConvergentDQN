import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import math, os
from scipy.ndimage import gaussian_filter as gaussian
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.interpolate import griddata

plt.rcParams["mathtext.fontset"]="cm"
plt.rcParams["axes.formatter.use_mathtext"]=True

algs = {"C-DQN": "CDQN", "DQN": "DQN"}
comments = {r"$N_{\tilde{\theta}}=200$": "target800", r"$N_{\tilde{\theta}}=20$": "target80"}
file_suffix_meaning = {"_": "reward", "mse_": "loss"}


game = "SpaceInvaders"
gaussian_smoothing = {"reward": 70, "loss": 5}

plt.figure(figsize=(6,2.5))
for i, (suffix, ylabel) in enumerate(file_suffix_meaning.items()):
    plt.subplot(1,2,i+1)
    for alg, filename in algs.items():
        for j, (label, comm) in enumerate(comments.items()):
            with open(os.path.join(game+"NoFrameskip-v4", filename+suffix+comm+".txt") , 'r') as f:
                lines = f.readlines()
                lines = [line.strip() for line in lines]
                x = np.array([float(line.split()[0]) for line in lines])
                y = np.array([float(line.split()[1]) for line in lines])
                if j == 0: label = alg +" "+label; linestyle = "-"
                else: linestyle = "--"
                color = "C0" if filename=="CDQN" else "C1"
                plt.plot(x, gaussian(y, sigma=gaussian_smoothing[ylabel], mode="mirror"), color=color, linestyle=linestyle, label=label) #/1e6
        plt.xlabel("frames")
        plt.ylabel(ylabel)
        
        if ylabel=="loss": 
            plt.yscale("log"); plt.xscale("linear"); plt.xlim(left=3e5, right=1e8)
        else: plt.xlim(left=0, right=1e8); plt.ylim(bottom=0); plt.legend() #loc="lower left"
        plt.tight_layout()

plt.subplots_adjust(wspace=0.3)
plt.savefig("{}_updatePeriod.pdf".format(game))
plt.tight_layout()
plt.close()



game = "Hero"
gaussian_smoothing = {"reward": 70, "loss": 5}

plt.figure(figsize=(6,2.5))
for i, (suffix, ylabel) in enumerate(file_suffix_meaning.items()):
    plt.subplot(1,2,i+1)
    for alg, filename in algs.items():
        for j, (label, comm) in enumerate(comments.items()):
            with open(os.path.join(game+"NoFrameskip-v4", filename+suffix+comm+".txt") , 'r') as f:
                lines = f.readlines()
                lines = [line.strip() for line in lines]
                x = np.array([float(line.split()[0]) for line in lines])
                y = np.array([float(line.split()[1]) for line in lines])
                if j == 0: label = alg +" "+label; linestyle = "-"

                else: linestyle = "--"
                color = "C0" if filename=="CDQN" else "C1"
                plt.plot(x, gaussian(y, sigma=gaussian_smoothing[ylabel], mode="mirror"), color=color, linestyle=linestyle, label=label) #/1e6
        plt.xlabel("frames")
        plt.ylabel(ylabel)
        
        if ylabel=="loss": 
            plt.yscale("log"); plt.xscale("linear"); plt.xlim(left=3e5, right=4e7)
        else: plt.xlim(left=0, right=4e7); plt.ylim(bottom=0); plt.legend() #loc="lower left"
        plt.tight_layout()

plt.subplots_adjust(wspace=0.3)
plt.savefig("{}_updatePeriod.pdf".format(game))
plt.tight_layout()
plt.close()
