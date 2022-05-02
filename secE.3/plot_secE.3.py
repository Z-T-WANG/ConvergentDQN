import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import math, os
from scipy.ndimage import gaussian_filter as gaussian
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.interpolate import griddata

plt.rcParams["mathtext.fontset"]="cm"
plt.rcParams["axes.formatter.use_mathtext"]=True



algs = {"C-DQN": "CDQN"}

game_smoothing = {"Skiing": 70, "Tennis": 30, "Bowling": 50, "SpaceInvaders": 50, "Hero": 20, "MsPacman": 100}
human = {"Skiing": -4336.9, "Tennis": -8.3, "Bowling": 160.7, "SpaceInvaders": 1668.7, "Hero": 30826.4, "MsPacman": 6951.6}
DDQN = {"Skiing": -9021.8, "Tennis": -22.8, "Bowling": 68.1, "SpaceInvaders": 2525.5, "Hero": 20130.2, "MsPacman": 2711.4}
title_append = {"Skiing": r" ($\gamma\approx0.9997$)", "Tennis": r" ($\gamma\approx 0.9972$)", "Bowling": r" ($\gamma\approx 0.9995$)", "SpaceInvaders": r" ($\gamma\approx 0.9969$)", "Hero": r" ($\gamma\approx 0.9976$)", "MsPacman": r" ($\gamma\approx 0.9948$)"}

plt.figure(figsize=(9,5))
for i, (game, smoothing) in enumerate(game_smoothing.items()):
    plt.subplot(2,3,i+1)
    plt.xlim(left=0, right=2e8) 
    plt.title(game+title_append[game])
    for alg, filename in algs.items():
        with open(os.path.join(game+"NoFrameskip-v4", filename+"_time10.txt") , 'r') as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines]
            x = np.array([float(line.split()[0]) for line in lines])
            y = np.array([float(line.split()[1]) for line in lines])
            y = gaussian(y, sigma=smoothing, mode="mirror")
            plt.plot(x, y, label=alg) #/1e6
        plt.axhline(y=human[game], c="grey", linestyle="--", label="Human")
        plt.axhline(y=DDQN[game], c="orange", linestyle="--", label="Double DQN")
    if i == 0:
        plt.legend(loc="lower right")
        plt.ylabel("reward", labelpad=0)
        plt.xlabel("frames", labelpad=2)
    plt.tight_layout()


plt.subplots_adjust(wspace=0.27)
plt.savefig("other games.pdf")
plt.close()


