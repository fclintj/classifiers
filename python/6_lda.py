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

def getRhat(x0,x1):
    N0 = x0.shape[1]
    N1 = x1.shape[1]
    N = N0 + N1

    mu = np.array([[np.mean(x0[0,:]),np.mean(x0[1,:])],
                   [np.mean(x1[0,:]),np.mean(x1[1,:])]])

    Rhat = np.empty([2,2])
    for i in range(N0):
        Rhat = Rhat + np.outer(x0[:,i]-mu[0],x0[:,i]+mu[0])

    for i in range(N1):
        Rhat = Rhat + np.outer(x1[:,i]-mu[1],x1[:,i]+mu[1])

    return Rhat/(N-2)

def calcDel(data,Rhat,mu,N0,N):
     return np.dot(np.dot(data,np.linalg.inv(Rhat)),mu) 
     - 0.5*np.dot(np.dot(mu,np.linalg.inv(Rhat)),mu)
     + np.log(N0)-np.log(N)

def getDel(x0,x1,Rhat):
    N0 = x0.shape[1]
    N1 = x1.shape[1]
    N = N0 + N1
    data_tot = np.c_[x0,x1]
    num_errors = 0
    mu = [[np.mean(x0[0,:]),np.mean(x0[1,:])],
          [np.mean(x1[0,:]),np.mean(x1[1,:])]]

    del_l = np.array([[],[]])

    for i in range(N):
        del_l = np.c_[del_l,np.array(
                [calcDel(data_tot[:,i],Rhat,mu[0],N0,N) , 
                 calcDel(data_tot[:,i],Rhat,mu[1],N0,N)]).T ]

    for i in range(N0):
        if del_l[0,i]<del_l[1,i]:
            num_errors=num_errors+1

    for i in range(N1):
        if del_l[0,N0+i]>del_l[1,N0+i]:
            num_errors=num_errors+1

    # return an array of 2 values for every point. Larger=class
    return del_l, num_errors

    
data = np.loadtxt("../data/classasgntrain1.dat",dtype=float)
x0 = data[:,0:2].T
x1 = data[:,2:4].T

N0 = x0.shape[1]
N1 = x1.shape[1]
N = N0 + N1

mu = np.array([[np.mean(x0[0,:]),np.mean(x0[1,:])],
               [np.mean(x1[0,:]),np.mean(x1[1,:])]])


fig = plt.figure() # make handle to save plot 
plt.scatter(x0[0,:],x0[1,:],c='red',label='$x_0$')
plt.scatter(x1[0,:],x1[1,:],c='blue',label='$y_0$')
plt.xlabel('X Coordinate') 
plt.ylabel('Y Coordinate') 
plt.legend()

# find parameter matrix
Rhat = getRhat(x0,x1)

del_l,num_err = getDel(x0,x1,Rhat)
print("Number of Errors: %d"%(num_err))
print("Percent errors: %.2f"%(float(num_err)/N))
Ntest0 = 10000; 
Ntest1 = 10000;

# generate the test data for class O
xtest0 = gendata2(0,Ntest0)
xtest1 = gendata2(1,Ntest1) 

del_l,num_err = getDel(xtest0,xtest1,Rhat)

np.savetxt('output.out', del_l)

print("Number of Errors: %d"%(num_err))
print("Percent errors: %.2f"%(float(num_err)/(Ntest0 + Ntest1)))

# find max and min of sets
x_tot = np.r_[x0[0,:],x1[0,:]]
y_tot = np.r_[x0[1,:],x1[1,:]]
xlim = [np.min(x_tot),np.max(x_tot)]
ylim = [np.min(y_tot),np.max(y_tot)]

# create colored graph above/below line
xp1 = np.linspace(xlim[0],xlim[1], num=100)
yp1 = np.linspace(ylim[0],ylim[1], num=100) 

red_pts = np.array([[],[]])
green_pts= np.array([[],[]])

for x in xp1:
    for y in yp1:
        del_l =  np.array(
                [calcDel([x,y],Rhat,mu[0],N0,N),
                 calcDel([x,y],Rhat,mu[1],N1,N)])

        if del_l[0]<del_l[1]:
            green_pts = np.c_[green_pts,[x,y]]
        else:
            red_pts = np.c_[red_pts,[x,y]]

plt.scatter(green_pts[0,:],green_pts[1,:],color='blue',s=0.25)
plt.scatter(red_pts[0,:],red_pts[1,:],color='red',s=0.25)
plt.xlim(xlim)
plt.ylim(ylim)
plt.show()
