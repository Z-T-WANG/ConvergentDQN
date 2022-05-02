import numpy as np 
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import math, os
from scipy.ndimage import gaussian_filter as gaussian
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.interpolate import griddata

plt.rcParams["mathtext.fontset"]="cm"
plt.rcParams["axes.formatter.use_mathtext"]=True
plt.rcParams['axes.unicode_minus'] = False
#plt.rcParams['text.usetex'] = True
#plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'




files = {"C-DQN": "CDQN", "DQN (NFQ)": "DQN", "RG": "RG"}

plt.figure(figsize=(8,3.5))
#plt.title("wet chicken")
plt.subplot(1,2,1)
for _i, (label, file_suffix) in enumerate(files.items()):
    smoothed_data = []
    data = np.array(torch.load("results_{}.pth".format(file_suffix)))
    mean = np.mean(data,axis=0)
    ste = np.std(data, axis=0, ddof=1)/math.sqrt(data.shape[0])
    x = (np.arange(len(mean))+1)*5
    #y = gaussian(y, sigma=smoothing, mode="mirror")
    plt.plot(x, gaussian(mean, sigma=1, mode="mirror"), label=label, color="C"+str(_i)) #/1e6
    plt.fill_between(x, gaussian(mean+ste, sigma=1, mode="mirror"), gaussian(mean-ste, sigma=2, mode="mirror"), facecolor="C"+str(_i), alpha = 0.3)

plt.xlim(left=-50, right=2050) 
plt.ylabel("average reward", labelpad=1)
plt.xlabel("epoch", labelpad=2)
plt.legend()#loc="upper left"
plt.tight_layout()

#plt.savefig("wet chicken.pdf")
#plt.close()



plt.subplot(1,2,2)
Q_DQN, Q_CDQN, Q_RG = torch.load("Qs_DQN.pth"), torch.load("Qs_CDQN.pth"), torch.load("Qs_RG.pth")
Q_DQN, Q_CDQN, Q_RG = np.array(Q_DQN), np.array(Q_CDQN), np.array(Q_RG)
Q_0 = np.expand_dims(Q_DQN[:,0,:,:], 1)


#plt.figure(figsize=(3,2.5))
smoothing = 2
dist_DQN_CDQN = np.sqrt(np.sum((Q_DQN-Q_CDQN)**2, axis=(2,3)))
dist_RG_CDQN = np.sqrt(np.sum((Q_RG-Q_CDQN)**2, axis=(2,3)))
dist_DQN_RG = np.sqrt(np.sum((Q_DQN-Q_RG)**2, axis=(2,3)))
dist_DQN_CDQN_plus_RG_CDQN_minus_DQN_RG = dist_DQN_CDQN+dist_RG_CDQN-dist_DQN_RG

dist_DQN_CDQN_mean = np.mean(dist_DQN_CDQN, axis=0)
dist_RG_CDQN_mean = np.mean(dist_RG_CDQN, axis=0)
dist_DQN_RG_mean = np.mean(dist_DQN_RG, axis=0)
dist_DQN_CDQN_plus_RG_CDQN_minus_DQN_RG_mean = np.mean(dist_DQN_CDQN_plus_RG_CDQN_minus_DQN_RG, axis=0)
dist_DQN_CDQN_std = np.std(dist_DQN_CDQN, axis=0, ddof=1)
dist_RG_CDQN_std = np.std(dist_RG_CDQN, axis=0, ddof=1)
dist_DQN_RG_std = np.std(dist_DQN_RG, axis=0, ddof=1)
dist_DQN_CDQN_plus_RG_CDQN_minus_DQN_RG_std = np.std(dist_DQN_CDQN_plus_RG_CDQN_minus_DQN_RG, axis=0, ddof=1)

x = np.arange(len(dist_DQN_CDQN_mean))
plt.plot(x, gaussian(dist_DQN_CDQN_mean, sigma=smoothing, mode="mirror"), label=r"$|Q_\mathrm{CDQN}-Q_\mathrm{DQN}|$")
plt.plot(x, gaussian(dist_RG_CDQN_mean, sigma=smoothing, mode="mirror"), label=r"$|Q_\mathrm{CDQN}-Q_\mathrm{RG}|$")
plt.plot(x, gaussian(dist_DQN_RG_mean, sigma=smoothing, mode="mirror"), label=r"$|Q_\mathrm{DQN}-Q_\mathrm{RG}|$")
plt.plot(x, gaussian(dist_DQN_CDQN_plus_RG_CDQN_minus_DQN_RG_mean, sigma=smoothing, mode="mirror"), label=r"$|Q_\mathrm{CDQN}-Q_\mathrm{DQN}|+|Q_\mathrm{CDQN}-Q_\mathrm{RG}|$"+"\n"+ r"$- |Q_\mathrm{DQN}-Q_\mathrm{RG}|$", linestyle= "--")
for _i, (mean, std) in enumerate([[dist_DQN_CDQN_mean, dist_DQN_CDQN_std], [dist_RG_CDQN_mean, dist_RG_CDQN_std], [dist_DQN_RG_mean, dist_DQN_RG_std], [dist_DQN_CDQN_plus_RG_CDQN_minus_DQN_RG_mean, dist_DQN_CDQN_plus_RG_CDQN_minus_DQN_RG_std]]):
    plt.fill_between(x, gaussian(mean+std, sigma=smoothing, mode="mirror"), gaussian(mean-std, sigma=smoothing, mode="mirror"), facecolor="C"+str(_i), alpha = 0.3)

plt.xlim(left=-50, right=2050) 
plt.ylim(bottom=-2) 
#plt.title("wet chicken")
plt.ylabel("distance", labelpad=1)
plt.xlabel("epoch", labelpad=2)
plt.legend()#loc="upper left"
plt.tight_layout()

#plt.savefig("wet chicken Qs.pdf")
#plt.close()
plt.savefig("wet_chicken.pdf")
plt.close()


#plt.subplot(1,3,2)
#dist_DQN_0 = np.sqrt(np.sum((Q_DQN-Q_0)**2, axis=(2,3)))
#dist_RG_0 = np.sqrt(np.sum((Q_RG-Q_0)**2, axis=(2,3)))
#dist_CDQN_0 = np.sqrt(np.sum((Q_CDQN-Q_0)**2, axis=(2,3)))
#dist_DQN_0_mean = np.mean(dist_DQN_0, axis=0)
#dist_RG_0_mean = np.mean(dist_RG_0, axis=0)
#dist_CDQN_0_mean = np.mean(dist_CDQN_0, axis=0)
#dist_DQN_0_std = np.std(dist_DQN_0, axis=0, ddof=1)
#dist_RG_0_std = np.std(dist_RG_0, axis=0, ddof=1)
#dist_CDQN_0_std = np.std(dist_CDQN_0, axis=0, ddof=1)

#plt.plot(x, gaussian(dist_CDQN_0_mean, sigma=smoothing, mode="mirror"), label=r"$|Q_\mathrm{CDQN}-Q_{0}|$")
#plt.plot(x, gaussian(dist_DQN_0_mean, sigma=smoothing, mode="mirror"), label=r"$|Q_{\mathrm{DQN}}-Q_{0}|$")
#plt.plot(x, gaussian(dist_RG_0_mean, sigma=smoothing, mode="mirror"), label=r"$|Q_\mathrm{RG}-Q_{0}|$")
#for _i, (mean, std) in enumerate([[dist_CDQN_0_mean, dist_CDQN_0_std], [dist_DQN_0_mean, dist_DQN_0_std], [dist_RG_0_mean, dist_RG_0_std]]): 
#    plt.fill_between(x, gaussian(mean+std, sigma=smoothing, mode="mirror"), gaussian(mean-std, sigma=smoothing, mode="mirror"), facecolor="C"+str(_i), alpha = 0.3)

#plt.xlim(left=-50, right=2050) 
#plt.ylim(bottom=0) 
#plt.title("wet chicken")
#plt.ylabel("distance to initialization", labelpad=1)
#plt.xlabel("epoch", labelpad=2)
#plt.legend()#loc="upper left"
#plt.tight_layout()

#plt.savefig("wet chicken Q progress.pdf")
#plt.close()






