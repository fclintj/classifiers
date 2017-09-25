import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib.ticker as ticker
import math

class data_frame:
    def __init__(self, data0, data1):
        self.x0 = data0 
        self.x1 = data1 
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

def gen_test_df(num0,num1):

    return data_frame

def get_parzen(data,pts,lam):
    x = np.linspace(min(data)-3*lam,max(data)+3*lam, pts)
    kernel = np.empty([data.size,pts])
    for i in range(data.size):
        kernel[i] = mlab.normpdf(x, data[i], lam)
        # plt.plot(x,kernel[i],color='b')

    parzen = np.empty(data.size)
    for i in range(data.size):
        parzen[i] = sum(kernel[:,i])/pts
   
    return x, parzen

def class_parzen(data0,pts,lam):
    x0,parzen_x0 = get_parzen(data0[0,:],pts,lam)
    y0,parzen_y0= get_parzen(data0[1,:],pts,lam)
    return [x0,y0],[parzen_x0,parzen_y0]

def prob2d(point,linspace0,parzen0):
    prob_x = np.interp(point[0],linspace0[0],parzen0[0])
    prob_y = np.interp(point[1],linspace0[1],parzen0[1])
    # print("prob x: %f, prob y: %f"%(prob_x, prob_y))
    return prob_x*prob_y     

def run_bayes_test(data_tot,linspace,parzen):
    y = np.r_[np.zeros([data_tot.shape[1],1]),np.ones([data_tot.shape[1],1])] 
    y_hat = np.zeros([data_tot.shape[1],1])

    for i in range(data_tot.shape[1]):
        prob0 = prob2d(data_tot[:,i],linspace[0],parzen[0])
        prob1 = prob2d(data_tot[:,i],linspace[1],parzen[0])
        if prob1 > prob0:
            y_hat[i] = 1 

    return y_hat


data = np.loadtxt("../data/classasgntrain1.dat",dtype=float)
x0 = data[:,0:2].T
x1 = data[:,2:4].T
data = data_frame(x0,x1)

pts = 100
lam = 0.8

linspace0,parzen0 = class_parzen(data.x0,pts,lam)
linspace1,parzen1 = class_parzen(data.x1,pts,lam)

linspace = np.array([linspace0,linspace1])
parzen = np.array([parzen0,parzen1])

y = np.r_[np.zeros([data.N1,1]),np.ones([data.N0,1])] 
y_hat = run_bayes_test(data.xtot,linspace,parzen)

num_err = sum(abs(y_hat - y))
print("Percent of errors: %.2f"%(float(num_err)/data.N))


xtest0 = gendata2(0,10000)
xtest1 = gendata2(1,10000) 
test_data = data_frame(xtest0,xtest1)
y = np.r_[np.zeros([test_data.N1,1]),np.ones([test_data.N0,1])] 

y_hat = run_bayes_test(test_data.xtot,linspace,parzen)

num_err = sum(abs(y_hat - y))
print("Percent of errors: %.2f"%(float(num_err)/data.N))

xp1 = np.linspace(data.xlim[0],data.xlim[1], num=100)
yp1 = np.linspace(data.ylim[0],data.ylim[1], num=100) 



red_pts = np.array([[],[]])
green_pts= np.array([[],[]])


for x in xp1:
    for y in yp1:
        prob0 = prob2d([x,y],linspace[0],parzen[0])
        prob1 = prob2d([x,y],linspace[1],parzen[0])
        if prob1 > prob0: 
            green_pts = np.c_[green_pts,[x,y]]
        else:
            red_pts = np.c_[red_pts,[x,y]]

plt.scatter(green_pts[0,:],green_pts[1,:],color='blue',s=0.25)
plt.scatter(red_pts[0,:],red_pts[1,:],color='red',s=0.25)
plt.xlim(data.xlim)
plt.ylim(data.ylim)
plt.show()
