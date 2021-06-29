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
seeds = "_1.txt", "_2.txt", "_3.txt"

game_smoothing = {"Skiing": 120, "Tennis": 30, "PrivateEye": 100, "Venture": 100}
title_append = {"Skiing": r" ($\gamma\approx0.9998$)", "Tennis": r" ($\gamma\approx 0.9982$)", "PrivateEye": r" ($\gamma\approx0.9998$)", "Venture": r" ($\gamma=0.9998$)"}

plt.figure(figsize=(12,2.5))
for i, (game, smoothing) in enumerate(game_smoothing.items()):
    plt.subplot(1,4,i+1)
    plt.xlim(left=0, right=2e8) 
    plt.title(game+title_append[game])
    for alg, filename in algs.items():
        color = "C0" if filename=="CDQN" else "C1"
        smoothed_data = []
        x_grid = np.arange(5e5,2e8,5e5)
        for j, seed in enumerate(seeds):
            with open(os.path.join(game+"NoFrameskip-v4", filename+seed) , 'r') as f:
                lines = f.readlines()
                lines = [line.strip() for line in lines]
                x = np.array([float(line.split()[0]) for line in lines])
                y = np.array([float(line.split()[1]) for line in lines])
                y = gaussian(y, sigma=smoothing, mode="mirror")
                plt.plot(x, y, label=alg if j==0 else None, color=color, alpha=0.6) #/1e6
                smoothed_data.append(griddata(x, y, x_grid, method="linear"))
        smoothed_data = np.array(smoothed_data)
        mean = np.mean(smoothed_data, axis=0)
        std = np.std(smoothed_data, axis=0, ddof=1)
        plt.fill_between(x_grid, mean+std, mean-std, facecolor=color, alpha = 0.3)
    if i == 0:
        plt.legend(loc="upper left")
        plt.ylabel("reward", labelpad=0)
        plt.xlabel("frames", labelpad=2)
    plt.tight_layout()

plt.subplots_adjust(wspace=0.25)
plt.savefig("difficult games.pdf")
plt.close()


