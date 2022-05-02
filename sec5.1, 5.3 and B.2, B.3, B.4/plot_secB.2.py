import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import math, os
from scipy.ndimage import gaussian_filter as gaussian
plt.rcParams["mathtext.fontset"]="cm"
plt.rcParams["axes.formatter.use_mathtext"]=True



algs = {"C-DQN": "CDQN", "DQN": "DQN"}
file_suffix_meaning = {"_.txt": "reward", "mse_.txt": "loss"}
linewidth = 1.2


game = "Breakout"
gaussian_smoothing = {"reward": 120, "loss": 1}

fig=plt.figure(figsize=(6,2.7))

for i, (suffix, ylabel) in enumerate(file_suffix_meaning.items()):
    plt.subplot(1,2,i+1)
    for alg, filename in algs.items():
        with open(os.path.join(game+"NoFrameskip-v4", filename+suffix) , 'r') as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines]
            x = np.array([float(line.split()[0]) for line in lines])
            y = np.array([float(line.split()[1]) for line in lines])
            plt.plot(x, gaussian(y, sigma=gaussian_smoothing[ylabel], mode="mirror"), label=alg, linewidth=linewidth) #/1e6
    plt.xlabel("frames")
    plt.ylabel(ylabel)
    #if i==0: 
    #    plt.legend() #loc="lower left"
    if ylabel=="loss": 
        plt.yscale("log"); plt.xlim(left=3e5, right=2e8)
    else: plt.xlim(left=0, right=2e8)
    plt.tight_layout()


fig.suptitle("Breakout")
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.subplots_adjust(wspace=0.3)
plt.savefig("{}.pdf".format(game))
plt.close()


game = "Freeway"
gaussian_smoothing = {"reward": 20, "loss": 1}

fig=plt.figure(figsize=(6,2.7))
for i, (suffix, ylabel) in enumerate(file_suffix_meaning.items()):
    plt.subplot(1,2,i+1)
    for alg, filename in algs.items():
        with open(os.path.join(game+"NoFrameskip-v4", filename+suffix) , 'r') as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines]
            x = np.array([float(line.split()[0]) for line in lines])
            y = np.array([float(line.split()[1]) for line in lines])
            plt.plot(x, gaussian(y, sigma=gaussian_smoothing[ylabel], mode="mirror"), label=alg, linewidth=linewidth) #/1e6
    plt.xlabel("frames")
    plt.ylabel(ylabel)
    #plt.legend() #loc="lower left"
    if ylabel=="loss": 
        plt.yscale("log"); plt.xlim(left=3e5, right=2e8)
    else: plt.xlim(left=0, right=2e8)
    plt.tight_layout()

fig.suptitle("Freeway")
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.subplots_adjust(wspace=0.3)
plt.savefig("{}.pdf".format(game))
plt.close()


game = "VideoPinball"
gaussian_smoothing = {"reward": 20, "loss": 1}

fig=plt.figure(figsize=(6,2.7))
for i, (suffix, ylabel) in enumerate(file_suffix_meaning.items()):
    plt.subplot(1,2,i+1)
    for alg, filename in algs.items():
        with open(os.path.join(game+"NoFrameskip-v4", filename+suffix) , 'r') as f:
            lines = f.readlines()

            lines = [line.strip() for line in lines]
            x = np.array([float(line.split()[0]) for line in lines])
            y = np.array([float(line.split()[1]) for line in lines])
            plt.plot(x, gaussian(y, sigma=gaussian_smoothing[ylabel], mode="mirror"), label=alg, linewidth=linewidth) #/1e6
    plt.xlabel("frames")
    plt.ylabel(ylabel)
    #plt.legend() #loc="lower left"
    if ylabel=="loss": 
        plt.yscale("log"); plt.xlim(left=3e5, right=2e8)
    else: plt.xlim(left=0, right=2e8)
    plt.tight_layout()

fig.suptitle("Video Pinball")
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.subplots_adjust(wspace=0.3)
plt.savefig("{}.pdf".format(game))
plt.close()



game = "FishingDerby"
gaussian_smoothing = {"reward": 50, "loss": 1}

fig=plt.figure(figsize=(6,2.7))
for i, (suffix, ylabel) in enumerate(file_suffix_meaning.items()):
    plt.subplot(1,2,i+1)
    for alg, filename in algs.items():
        with open(os.path.join(game+"NoFrameskip-v4", filename+suffix) , 'r') as f:
            lines = f.readlines()

            lines = [line.strip() for line in lines]
            x = np.array([float(line.split()[0]) for line in lines])
            y = np.array([float(line.split()[1]) for line in lines])
            plt.plot(x, gaussian(y, sigma=gaussian_smoothing[ylabel], mode="mirror"), label=alg, linewidth=linewidth) #/1e6
    plt.xlabel("frames")
    plt.ylabel(ylabel)
    #plt.legend() #loc="lower left"
    if ylabel=="loss": 
        plt.yscale("log"); plt.xlim(left=3e5, right=2e8)
    else: plt.xlim(left=0, right=2e8)
    plt.tight_layout()

fig.suptitle("Fishing Derby")
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.subplots_adjust(wspace=0.3)
plt.savefig("{}.pdf".format(game))
plt.close()



game = "Atlantis"
gaussian_smoothing = {"reward": 8, "loss": 1}

fig=plt.figure(figsize=(6,2.7))
for i, (suffix, ylabel) in enumerate(file_suffix_meaning.items()):
    plt.subplot(1,2,i+1)
    for alg, filename in algs.items():
        with open(os.path.join(game+"NoFrameskip-v4", filename+suffix) , 'r') as f:
            lines = f.readlines()

            lines = [line.strip() for line in lines]
            x = np.array([float(line.split()[0]) for line in lines])
            y = np.array([float(line.split()[1]) for line in lines])
            plt.plot(x, gaussian(y, sigma=gaussian_smoothing[ylabel], mode="mirror"), label=alg, linewidth=linewidth) #/1e6
    plt.xlabel("frames")
    plt.ylabel(ylabel)
    if i==0: 
        plt.legend()
    #plt.legend() #loc="lower left"
    if ylabel=="loss": 
        plt.yscale("log"); plt.xlim(left=3e5, right=2e8)
    else: plt.xlim(left=0, right=2e8)
    plt.tight_layout()

fig.suptitle("Atlantis")
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.subplots_adjust(wspace=0.3)
plt.savefig("{}.pdf".format(game))
plt.close()

game = "TimePilot"
gaussian_smoothing = {"reward": 100, "loss": 1}

fig=plt.figure(figsize=(6,2.7))
for i, (suffix, ylabel) in enumerate(file_suffix_meaning.items()):
    plt.subplot(1,2,i+1)
    for alg, filename in algs.items():
        with open(os.path.join(game+"NoFrameskip-v4", filename+suffix) , 'r') as f:
            lines = f.readlines()

            lines = [line.strip() for line in lines]
            x = np.array([float(line.split()[0]) for line in lines])
            y = np.array([float(line.split()[1]) for line in lines])
            plt.plot(x, gaussian(y, sigma=gaussian_smoothing[ylabel], mode="mirror"), label=alg, linewidth=linewidth) #/1e6
    plt.xlabel("frames")
    plt.ylabel(ylabel)
    #plt.legend() #loc="lower left"
    if ylabel=="loss": 
        plt.yscale("log"); plt.xlim(left=3e5, right=2e8)
    else: plt.xlim(left=0, right=2e8)
    plt.tight_layout()

fig.suptitle("Time Pilot")
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.subplots_adjust(wspace=0.3)
plt.savefig("{}.pdf".format(game))
plt.close()



