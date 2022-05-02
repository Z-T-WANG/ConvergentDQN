import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import math, os
from scipy.ndimage import gaussian_filter as gaussian
plt.rcParams["mathtext.fontset"]="cm"
plt.rcParams["axes.formatter.use_mathtext"]=True



algs = {"C-DQN": "CDQN", "DQN": "DQN", "RG": "Residual"}
file_suffix_meaning = {"_.txt": "reward", "mse_.txt": "loss"}


game = "Pong"
gaussian_smoothing = {"reward": 15, "loss": 3}

plt.figure(figsize=(6,2.5))
for i, (suffix, ylabel) in enumerate(file_suffix_meaning.items()):
    plt.subplot(1,2,i+1)
    for alg, filename in algs.items():
        with open(os.path.join(game+"NoFrameskip-v4", filename+suffix) , 'r') as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines]
            x = np.array([float(line.split()[0]) for line in lines])
            y = np.array([float(line.split()[1]) for line in lines])
            plt.plot(x, gaussian(y, sigma=gaussian_smoothing[ylabel], mode="mirror"), label=alg) #/1e6
    plt.xlabel("frames")
    plt.ylabel(ylabel)
    plt.legend() #loc="lower left"
    if ylabel=="loss": 
        plt.yscale("log"); plt.xscale("log"); plt.xlim(left=3e5, right=1e8)
    else: plt.xlim(left=0, right=1e8)
    plt.tight_layout()

plt.subplots_adjust(wspace=0.3)
plt.savefig("{}.pdf".format(game))
plt.tight_layout()
plt.close()


game = "SpaceInvaders"
gaussian_smoothing = {"reward": 60, "loss": 2}

plt.figure(figsize=(6,2.5))
for i, (suffix, ylabel) in enumerate(file_suffix_meaning.items()):
    plt.subplot(1,2,i+1)
    for alg, filename in algs.items():
        with open(os.path.join(game+"NoFrameskip-v4", filename+suffix) , 'r') as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines]
            x = np.array([float(line.split()[0]) for line in lines])
            y = np.array([float(line.split()[1]) for line in lines])
            plt.plot(x, gaussian(y, sigma=gaussian_smoothing[ylabel], mode="mirror"), label=alg) #/1e6
    plt.xlabel("frames")
    plt.ylabel(ylabel)
    plt.legend() #loc="lower left"
    if ylabel=="loss": 
        plt.yscale("log"); plt.xscale("log"); plt.xlim(left=3e5, right=2e8)
    else: plt.xlim(left=0, right=2e8)
    plt.tight_layout()

plt.subplots_adjust(wspace=0.3)
plt.savefig("{}.pdf".format(game))
plt.tight_layout()
plt.close()

