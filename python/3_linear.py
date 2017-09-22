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

def plotData(x0,x1):
    fig = plt.figure() # make handle to save plot 
    plt.scatter(x0[0,:],x0[1,:],c='red',label='$x_0$')
    plt.scatter(x1[0,:],x1[1,:],c='blue',label='$y_0$')
    plt.xlabel('X Coordinate') 
    plt.ylabel('Y Coordinate') 
    plt.legend()


data = np.loadtxt("../data/classasgntrain1.dat",dtype=float)
x0 = data[:,0:2].T
x1 = data[:,2:4].T
data_tot = np.c_[x0,x1]

N0 = x0.shape[1]
N1 = x1.shape[1];
N = N0 + N1

# linear regression classifier
X = np.r_[np.c_[np.ones((N0,1)),x0.T],
          np.c_[np.ones((N1,1)),x1.T]]

Y = np.r_[np.c_[np.ones((N0,1)),np.zeros((N0,1))],
          np.c_[np.zeros((N1,1)),np.ones((N1,1))]]

# find parameter matrix
Bhat = np.dot(np.linalg.inv(np.dot(X.T,X)),np.dot(X.T,Y))

# find approximate response
Yhat = np.dot(X,Bhat)
Yhathard = Yhat > 0.5    

num_err = sum(sum(abs(Yhathard - Y)))/2
print("Number of errors: %d"%(num_err))

Ntest0 = 10000; 
Ntest1 = 10000;

err_rate_linregress_train = float(num_err) / N
print("Percent of errors: %.2f"%(err_rate_linregress_train))

# generate the test data for class O
xtest0 = gendata2(0,Ntest0)
xtest1 = gendata2(1,Ntest1) 
num_err = 0;
 
for i in range(Ntest0):
    yhat = np.dot(np.r_[1,xtest0[:,i]],Bhat)
    if yhat[1] > yhat[0]:
        num_err = num_err + 1;

for i in range(Ntest1):
    yhat = np.dot(np.r_[1,xtest1[:,i]],Bhat)
    if yhat[1] < yhat[0]: 
        num_err = num_err + 1;

print("Number of errors: %d"%(num_err))
err_rate_linregress_test = float(num_err) / (Ntest0 + Ntest1);
print("Percent of errors: %.2f"%(err_rate_linregress_test))


# find max and min of sets
x_tot = np.r_[x0[0,:],x1[0,:]]
y_tot = np.r_[x0[1,:],x1[1,:]]
xlim = [np.min(x_tot),np.max(x_tot)]
ylim = [np.min(y_tot),np.max(y_tot)]

# find x,y coordinate of separating line
x_cor_lin = [xlim[0],xlim[1]]
y_cor_lin = [
    (Bhat[0,0]-Bhat[0,1]+(Bhat[1,0]-Bhat[1,1])*xlim[0])
             /(Bhat[2,1]-Bhat[2,0]),

    (Bhat[0,0]-Bhat[0,1]+(Bhat[1,0]-Bhat[1,1])*xlim[1])
             /(Bhat[2,1]-Bhat[2,0]) 
]

# create colored graph above/below line
xp1 = np.linspace(xlim[0],xlim[1], num=100)
yp1 = np.linspace(ylim[0],ylim[1], num=100) 

red_pts = np.array([[],[]])
green_pts= np.array([[],[]])

for x in xp1:
    for y in yp1:
        yhat = np.dot(np.r_[1,x,y],Bhat)
        if yhat[1] > yhat[0]: 
            green_pts = np.c_[green_pts,[x,y]]
        else:
            red_pts = np.c_[red_pts,[x,y]]

plotData(x0,x1)
plt.plot(x_cor_lin,y_cor_lin,color='black')
plt.scatter(green_pts[0,:],green_pts[1,:],color='blue',s=0.25)
plt.scatter(red_pts[0,:],red_pts[1,:],color='red',s=0.25)
plt.xlim(xlim)
plt.ylim(ylim)
plt.show()

