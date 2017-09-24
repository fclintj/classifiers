import sys 
import numpy as np
import matplotlib.pyplot as plt

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

def plotData(data):
    fig = plt.figure() # make handle to save plot 
    plt.scatter(data.x0[0,:],data.x0[1,:],c='red',label='$x_0$')
    plt.scatter(data.x1[0,:],data.x1[1,:],c='blue',label='$y_0$')
    plt.xlabel('X Coordinate') 
    plt.ylabel('Y Coordinate') 
    plt.legend()

def get_phat(data,X,beta):
    phat = np.zeros([data.N,1])
    for i in range(data.N):
        phat[i] = (np.power(np.e,np.dot(beta.T,X[i,:]).T))/(1 + np.power(np.e,np.dot(beta.T,X[i,:]).T))
    return phat

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

data = np.loadtxt("../data/classasgntrain1.dat",dtype=float)
data = data_frame(data)
y = np.r_[np.ones([data.N0,1]),np.zeros([data.N1,1])] 
X = np.r_[np.ones([1,data.xtot.shape[1]]), data.xtot].T 
beta = np.zeros([3,1])

for i in range(500):
    phat = get_phat(data,X,beta)
    Xhat = X*phat
    beta = beta + np.dot(np.dot(np.linalg.inv(np.dot(X.T,Xhat)),X.T),(y-phat))
    print(beta)

phat_hard = phat > 0.5
num_err = (sum(abs(phat_hard - y)))
print("Percent of errors: %.2f"%(float(num_err)/data.N))

Ntest0 = 10000; 
Ntest1 = 10000;

xtest0 = gendata2(0,Ntest0)
xtest1 = gendata2(1,Ntest1) 

for i in range(Ntest0):
    prob = np.dot(beta.T,np.r_[1,xtest0[:,i]]) 
    if prob < 0.5: 
        num_err = num_err + 1;

for i in range(Ntest1):
    prob = np.dot(beta.T,np.r_[1,xtest1[:,i]]) 
    if prob > 0.5: 
        num_err = num_err + 1;

print("Number of errors: %d"%(num_err))
err_rate_linregress_test = float(num_err) / (Ntest0 + Ntest1);
print("Percent of errors: %.3f"%(err_rate_linregress_test))


# create colored graph above/below line
xp1 = np.linspace(data.xlim[0],data.xlim[1], num=100)
yp1 = np.linspace(data.ylim[0],data.ylim[1], num=100) 

red_pts = np.array([[],[]])
green_pts= np.array([[],[]])

for x in xp1:
    for y in yp1:
        prob = np.dot(beta.T,np.r_[1,x,y])
        if prob > 0.5: 
            green_pts = np.c_[green_pts,[x,y]]
        else:
            red_pts = np.c_[red_pts,[x,y]]

plotData(data)
plt.scatter(green_pts[0,:],green_pts[1,:],color='blue',s=0.25)
plt.scatter(red_pts[0,:],red_pts[1,:],color='red',s=0.25)
plt.xlim(data.xlim)
plt.ylim(data.ylim)
plt.show()

