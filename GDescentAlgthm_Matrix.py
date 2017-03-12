# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 14:53:16 2016

@author: Ganesh
"""
#%%
from numpy import *

import numpy as np
import random
from sklearn.datasets.samples_generator import make_regression 
import pylab
from scipy import stats
import pandas as pd
import os
#%%

#%%
path = "S:\\ANALYTICS\\R_Programming\\Gradient_Descent"
os.chdir(path)

points = genfromtxt("Data_Python_testing.csv", delimiter=",")


#%%
# Y = mX+c


#%%
a= points[:,0]
b= points[:,1]

x= np.reshape(a,(len(a),1))
x = np.c_[ np.ones(len(a)), x] # insert column
y = b

#%%


#%%
m = y.shape[0] 
hypothesis = np.dot(x, theta)
loss = hypothesis - y
x_transpose = x.transpose()
#%%


#%%
def Dev_grad(x, y, theta):
    m = y.shape[0]
    hypothesis = np.dot(x, theta)
    loss = hypothesis - y
    x_transpose = x.transpose()
    gradient = (1/m) * np.dot(x_transpose, loss) 
    return(gradient) 

#%%

#%%
theta = np.ones(2)
Dev_grad(x,y, theta)

#%%



#with separate functions 
#%%
def gradient_descent_2(alpha, x, y, numIterations):
    m = y.shape[0] # number of samples
    theta = np.zeros(2)
    x_transpose = x.transpose()
    for iter in range(0, numIterations):
        m = x.shape[0]
        hypothesis = np.dot(x, theta)
        loss = hypothesis - y
        J = np.sum(loss ** 2) / (2 * m)  # cost
       # print "iter %s | J: %3f" % (iter, J)    
        theta = theta - alpha * Dev_grad(x,y, theta)  # update
    return theta
    
#%%    



# calling the function.

#%%
gradient_descent_2 (0.0001,x,y,1000)    
#%%



#calling function 
#%%
#%%
import numpy as np
import random
from sklearn.datasets.samples_generator import make_regression 
import pylab
from scipy import stats

path = "S:\\ANALYTICS\\R_Programming\\Gradient_Descent"
os.chdir(path)

points = genfromtxt("Data_Python.csv", delimiter=",")

a= points[:,0]
b= points[:,1]

x= np.reshape(a,(len(a),1))
x = np.c_[ np.ones(len(a)), x] # insert column
y = b


def gradient_descent(alpha, x, y, numIterations):
    m = x.shape[0] # number of samples
    theta = np.ones(2)
    x_transpose = x.transpose()
    for iter in range(0, numIterations):
        hypothesis = np.dot(x, theta)
        loss = hypothesis - y
        J = np.sum(loss ** 2) / (2 * m)  # cost
        #print "iter %s | J: %.3f" % (iter, J)      
        gradient = np.dot(x_transpose, loss) / m         
        theta = theta - alpha * gradient  # update
    return theta

    
    
gradient_descent (0.0001,x,y,1000)    
#%%






#Code a

#%%

import numpy as np
import random
import sklearn
from sklearn.datasets.samples_generator import make_regression 
import pylab
from scipy import stats

def gradient_descent(alpha, x, y, ep=0.0001, max_iter=10000):
    converged = False
    iter = 0
    m = len(x) # number of samples

    # initial theta
    t0 = 1
    t1 = 1
    #t0 = np.random.random(x.shape[1])
    #t1 = np.random.random(x.shape[1])

    # total error, J(theta)
    J = sum([(t0 + t1*x[i] - y[i])**2 for i in range(m)])

    # Iterate Loop
    while not converged:
        # for each training sample, compute the gradient (d/d_theta j(theta))
        grad0 = 1.0/m * sum([(t0 + t1*x[i] - y[i]) for i in range(m)]) 
        grad1 = 1.0/m * sum([(t0 + t1*x[i] - y[i])*x[i] for i in range(m)])

        # update the theta_temp
        temp0 = t0 - alpha * grad0
        temp1 = t1 - alpha * grad1
    
        # update theta
        t0 = temp0
        t1 = temp1

        # mean squared error
        e = sum( [ (t0 + t1*x[i] - y[i])**2 for i in range(m)] ) 

        if abs(J-e) <= ep:
            print ('Converged, iterations: '), iter, ('!!!')
            converged = True
    
        J = e   # update error 
        iter += 1  # update iter
    
        if iter == max_iter:
            print ('Max interactions exceeded!')
            converged = True

    return t0,t1
#%%





#%%
path = "S:\\ANALYTICS\\R_Programming\\Gradient_Descent"
os.chdir(path)

points = genfromtxt("Data_Python.csv", delimiter=",")

alpha = 0.0001
x = [(i[0]) for i in points]
y = [(i[1]) for i in points]

gradient_descent(alpha, x, y, ep=0.0001, max_iter=1000)

#%%


