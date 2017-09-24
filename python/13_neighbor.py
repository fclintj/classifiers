import sys 
import numpy as np
import matplotlib.pyplot as plt

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

class data_frame:
    def __init__(self, data):
        self.x0 = data[:,0:2]
        self.x1 = data[:,2:4]
        self.xtot = np.r_[self.x0,self.x1]
        self.N0 = self.x0.shape[0]
        self.N1 = self.x1.shape[0]
        self.N = self.N0 + self.N1
        self.xlim = [np.min(self.xtot[:,0]),np.max(self.xtot[:,0])]
        self.ylim = [np.min(self.xtot[:,1]),np.max(self.xtot[:,1])]

def plot_data(data):
    fig = plt.figure() # make handle to save plot 
    plt.scatter(data.x0[:,0],data.x0[:,1],c='blue',label='$x_0$')
    plt.scatter(data.x1[:,0],data.x1[:,1],c='red',label='$x_1$')
    plt.xlabel('X Coordinate') 
    plt.ylabel('Y Coordinate') 
    plt.legend()

def get_distance_matrix(X,point):
    dist_mat = np.empty([X.shape[0],2]) 
    for i in range(X.shape[0]):
        dist_mat[i] = [np.linalg.norm(point - X[i,0:2]),X[i,2]]
    return dist_mat

def find_class(dist_mat,k):
    neighbors = dist_mat[0:k]
    for i in range(k,X.shape[0]):
        if np.max(neighbors[:,0]) > dist_mat[i,0]:
            neighbors[np.argmax(neighbors[:,0])] = dist_mat[i]
    prob = sum(neighbors[:,1])/k    
    if prob > 0.5:
        class_type = 1
    else:
        class_type = 0
    return class_type 

data = np.loadtxt("../data/classasgntrain1.dat",dtype=float)
data = data_frame(data)
k = 1
y = np.r_[np.zeros([data.N0,1]),np.ones([data.N1,1])] 
X = np.c_[data.xtot,y] 
yhat = np.empty([data.N,1])

for i in range(data.N):
    dist_mat = get_distance_matrix(X,data.xtot[i])
    yhat[i] = find_class(dist_mat,k)
    
num_err = (sum(abs(yhat - y)))
print("Percent of errors: %.2f"%(float(num_err)/data.N))

Ntest0 = 10000; 
Ntest1 = 10000;

xtest0 = gendata2(0,Ntest0)
xtest1 = gendata2(1,Ntest1) 
num_err = 0

for i in range(Ntest0):
    dist_mat = get_distance_matrix(X,xtest0[:,i])
    class_type = find_class(dist_mat,k)
    if class_type == 1:
        num_err = num_err + 1

for i in range(Ntest1):
    dist_mat = get_distance_matrix(X,xtest1[:,i])
    class_type = find_class(dist_mat,k)
    if class_type == 0: 
        num_err = num_err + 1

print("Number of errors: %d"%(num_err))
err_rate_linregress_test = float(num_err) / (Ntest0 + Ntest1);
print("Percent of errors: %.3f"%(err_rate_linregress_test))

# create colored graph above/below line
xp1 = np.linspace(data.xlim[0],data.xlim[1], num=100)
yp1 = np.linspace(data.ylim[0],data.ylim[1], num=100) 

red_pts = np.array([[],[]])
blue_pts= np.array([[],[]])

for x in xp1:
    for y in yp1:
        dist_mat = get_distance_matrix(X,[x,y])
        class_type = find_class(dist_mat,k)
        if class_type == 0: 
            blue_pts = np.c_[blue_pts,[x,y]]
        else:
            red_pts = np.c_[red_pts,[x,y]]

plot_data(data)
plt.scatter(blue_pts[0,:],blue_pts[1,:],color='blue',s=0.25)
plt.scatter(red_pts[0,:],red_pts[1,:],color='red',s=0.25)
plt.xlim(data.xlim)
plt.ylim(data.ylim)
plt.show()


