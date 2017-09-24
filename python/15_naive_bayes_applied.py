import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib.ticker as ticker
import math

class data_frame:
    def __init__(self, data):
        self.x0 = data[:,0:2].T
        self.x1 = data[:,2:4].T
        self.xtot = np.c_[self.x0,self.x1]
        self.N0 = self.x0.shape[1]
        self.N1 = self.x1.shape[1]
        self.N = self.N0 + self.N1
        self.xlim = [np.min(self.xtot[0,:]),np.max(self.xtot[0,:])]
        self.ylim = [np.min(self.xtot[1,:]),np.max(self.xtot[1,:])]

def gendata2(class_type,N):
    m0 = np.array(
         [[-0.132,0.320,1.672,2.230,1.217,-0.819,3.629,0.8210,1.808, 0.1700],
          [-0.711,-1.726,0.139,1.151,-0.373,-1.573,-0.243,-0.5220,-0.511,0.5330]])

    m1 = np.array(
          [[-1.169,0.813,-0.859,-0.608,-0.832,2.015,0.173,1.432,0.743,1.0328],
          [ 2.065,2.441,0.247,1.806,1.286,0.928,1.923,0.1299,1.847,-0.052]])

    x = np.array([[],[]])
    for i in range(N):
        idx = np.random.randint(10)
        if class_type == 0:
            m = m0[:,idx]
        elif class_type == 1:
            m = m1[:,idx]
        else:
            print("not a proper classifier")
            return 0 
        x = np.c_[x, [[m[0]],[m[1]]] + np.random.randn(2,1)/np.sqrt(5)]
    return x 

def getParzen(data,pts,lam):
    # give it x/y pairs and it returns linspaces and Parzen pdf
    x = np.linspace(min(data)-3*lam,max(data)+3*lam, pts)
    kernel = np.empty([data.size,pts])
    for i in range(data.size):
        kernel[i] = mlab.normpdf(x, data[i], lam)
        # plt.plot(x,kernel[i],color='b')

    parzen = np.empty(data.size)
    for i in range(data.size):
        parzen[i] = sum(kernel[:,i])/pts
   
    return x, parzen

data = np.loadtxt("../data/classasgntrain1.dat",dtype=float)
data = data_frame(data)

pts = 100
lam = 0.8

x0,parzen_x0 = getParzen(data.x0[0,:],pts,lam)
y0,parzen_y0= getParzen(data.x0[1,:],pts,lam)

x1,parzen_x1 = getParzen(data.x1[0,:],pts,lam)
y1,parzen_y1= getParzen(data.x1[1,:],pts,lam)

plt.plot(x1,parzen_x1)
plt.plot(y1,parzen_y1)

prob = np.interp(-10,x,parzen)
# print(prob)
plt.show()

# emperical = np.interp(np.random.rand(pts,1),cdf,x)
# emperical = np.sort(emperical,axis=None)
#
# ax[1].stem(emperical,np.ones(pts),'b',markerfmt=' ')
# ax[1].set_xlim(0,14)
#
#
# parzen = np.empty(100)
# for i in range(pts):
#     parzen[i] = sum(kernel[:,i])/pts
#
# ax[3].plot(x,parzen)
#
# plt.show()
#
