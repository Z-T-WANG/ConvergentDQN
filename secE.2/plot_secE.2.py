import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import math, os
from scipy.ndimage import gaussian_filter as gaussian
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.interpolate import griddata

plt.rcParams["mathtext.fontset"]="cm"
plt.rcParams["axes.formatter.use_mathtext"]=True

comments = {r"DQN $\epsilon_p=10^{-8}$": "pEps-8", r"$\epsilon_p=10^{-6}$": "pEps-6", r"$\epsilon_p=10^{-4}$": "pEps-4"}
file_suffix_meaning = {"_": "reward", "mse_": "loss"}


game = "SpaceInvaders"
gaussian_smoothing = {"reward": 30, "loss": 1}

plt.figure(figsize=(6,2.5))
for i, (suffix, ylabel) in enumerate(file_suffix_meaning.items()):
    plt.subplot(1,2,i+1)
    for label, comm in comments.items():
        with open(os.path.join(game+"NoFrameskip-v4", "DQN"+suffix+comm+".txt") , 'r') as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines]
            x = np.array([float(line.split()[0]) for line in lines])
            y = np.array([float(line.split()[1]) for line in lines])
            plt.plot(x, gaussian(y, sigma=gaussian_smoothing[ylabel], mode="mirror"), label=label) #/1e6
    plt.xlabel("frames")
    plt.ylabel(ylabel)
    plt.legend() #loc="lower left"
    if ylabel=="loss": 
        plt.yscale("log"); plt.xlim(left=0, right=1.5e8)
    else: plt.xlim(left=0, right=1.5e8); plt.ylim(bottom=0)
    plt.tight_layout()

plt.subplots_adjust(wspace=0.3)
plt.savefig("{}_pEpsilon.pdf".format(game))
plt.tight_layout()
plt.close()
