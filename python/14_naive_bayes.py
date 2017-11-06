# Clint Ferrin
# Mon Sep 25, 2017
# Parzen Density Estimate

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

p_0 = 0.8
p_1 = 0.2
mu_0 = 5
mu_1 = 10
sig_0 = 1
sig_1 = 1

lam = .5 

f, ax = plt.subplots(4,1)
f.subplots_adjust(hspace=.5)
f.suptitle("Bayes Classifier",fontsize=14)

ax[0].set_title("True Density",fontsize=12)
ax[0].set_ylim(0, 0.5)
ax[0].yaxis.set_ticks(np.arange(0,0.6,0.2))
ax[1].set_title("Empirical Distribution",fontsize=12)
ax[1].set_ylim(0, 1.5)
ax[1].yaxis.set_ticks(np.arange(0,2,1))
ax[2].set_title("Kernel Functions",fontsize=12)
ax[2].set_ylim(0, .65)
ax[2].yaxis.set_ticks(np.arange(0,.64,.3))
ax[3].set_title("Parzen Density Estimate",fontsize=12)
ax[3].set_ylim(0, 0.35)
ax[3].yaxis.set_ticks(np.arange(0,0.6,0.2))
for i in range(4):
    ax[i].set_xlim(0, 14)

pts = 100
x = np.linspace(0,14, pts)
pdf = p_0*mlab.normpdf(x, mu_0, sig_0)+p_1*mlab.normpdf(x, mu_1, sig_1)
ax[0].plot(x,pdf) 
cdf = np.cumsum(pdf)/sum(pdf)



emperical = np.interp(np.random.rand(pts,1),cdf,x)
emperical = np.sort(emperical,axis=None)

ax[1].stem(emperical,np.ones(pts),'b',markerfmt=' ')
ax[1].set_xlim(0,14)

kernel = np.empty([emperical.shape[0],pts])
for i in range(emperical.shape[0]):
    kernel[i] = mlab.normpdf(x, emperical[i], lam)
    ax[2].plot(x,kernel[i],color='c')

parzen = np.empty(100)
for i in range(pts):
    parzen[i] = sum(kernel[:,i])/pts

ax[3].plot(x,parzen,color='forestgreen')

plt.show()
