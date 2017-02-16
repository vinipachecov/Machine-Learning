#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 10:17:53 2017

@author: Vinícius Pacheco Vieira
"""


import pandas as pd
import matplotlib.pylab as plt
import numpy as np
import scipy.optimize as opt  


def sigmoid(Z):
    '''Compute the sigmoid function '''
    return 1.0 / (1.0 + np.exp( -1.0 * Z))

    
def compute_cost(theta,X,y):
    '''compute cost given '''
            
    m = y.size
    theta.shape = (1,3)
    
    Z = X.dot(theta.T)
    
    hypothesis = sigmoid(Z)  
    
    J = ((-y.T.dot(np.log(hypothesis))) - ((1.0 - y).T.dot(np.log(1.0 - hypothesis)))) / m 
        
    return J.sum()


def grad(theta,X,y,):    
    
        
    X = np.matrix(X)
    y = np.matrix(y)
    m = y.size
   # theta0 = np.zeros(X.shape[1])  
    #reg = np.dot(learningRate / m, theta0)
    
    Z = X.dot(theta.T)    
    hypothesis = sigmoid(Z)      
    error = hypothesis - y    
    grad =  ((1/m) * X.T.dot(error)).flatten()     
    
    return grad          

def compute_grad(theta,X,y):    
    
    
    m = y.size
    theta.shape = (1, 3)    
    grad = np.zeros(3)    
    Z = X.dot(theta.T)    
    hypothesis = sigmoid(Z)      
    error = hypothesis - y    
    grad =  ((1/m) * X.T.dot(error)).flatten()
    theta.shape = (3,) 
    
    return grad
    
####################################
## Running settings

data= np.loadtxt('ex2data1.txt', delimiter=',')


X = data[:, 0:2]
y = data[:, -1]

pos = plt.where(y == 1)
neg = plt.where(y == 0)

m,n = X.shape
y.shape = (m,1)


## + 1 interception term
it = np.ones(shape=(m,n +1))
##add intercept term to X, i.e it variable
it[:,1:] = X

##Initialize theta parameters
theta = np.zeros(3)
  
## test functions
print (compute_cost(theta,it,y))
##compute cost should be at 0.693
print("Gradient at initial theta")
print (grad(theta,it,y))
##gradient for initial theta (0, 0 ,0 ) should be  [ -0.1,   -12.00921659, -11.26284221]



##Uncomment for random theta values
#theta = 0.1* np.random.randn(3)

## fmin_tnc gets optimal values for alpha so we don't have to choose it randomly 
## or by gut feeling
result = opt.fmin_tnc(func=compute_cost, x0=theta, fprime=grad, args=(it, y))  
##First value returned is the optimal theta parameters for the model
theta = result[0]

#Plotting the decision boundary
plot_x = np.array([min(it[:, 1]) - 2, max(it[:, 2]) + 2])
plot_y = (- 1.0 / theta[2]) * (theta[1] * plot_x + theta[0])
plt.plot(plot_x, plot_y)
plt.legend(['Decision Boundary', 'Not admitted', 'Admitted'])
#show()


def predict(theta,X):
    '''predict whether the label
    is 0 or 1 using learned logistic
    regression parameters'''
    m,n = X.shape
    p = np.zeros(shape=(m,1))
    
    h = sigmoid(X.dot(theta.T))
    
    for it in range(0,h.shape[0]):
        if h[it] > 0.5:
            p[it,0] = 1
        else:
            p[it,0] = 0
    
    return p

p = predict(np.array(theta), it)
print ('Train Accuracy: %f' % ((y[np.where(p == y)].size / float(y.size)) * 100.0))

##VIsualizing data X[pos,0] -> first exam note (admitted)
##VIsualizing data X[pos,1] -> second exam note (admitted)
plt.scatter(X[pos, 0], X[pos, 1], marker='o', c='r')
#Visualizing data[neg,0] -> first exam note (not admitted)
#VIzualigin data[neg,1] -> second exam note (not admitted)
plt.scatter(X[neg, 0], X[neg, 1], marker='x', c='g')
plt.xlabel('Exam 1 Score')
plt.ylabel('Exam 2 Score')
plt.legend(['Not Admitted', 'Admitted'])
plt.show()