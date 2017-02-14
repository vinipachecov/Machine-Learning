#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 15:46:30 2017

@author: root
"""
from numpy import loadtxt, zeros, ones, array, linspace, logspace, shape, repeat,sum,dot
import numpy as np
from pylab import scatter, show, title, xlabel, ylabel, plot, contour
import matplotlib.pyplot as plt



def feature_normalize(X):
    '''
    Returns a normalized version of X Where
    the mean value of each feature is 0 and the standard deviation is 1. 
    THis is often a good preprocessing step do do when working with learning 
    algorithms
    '''
    
    mean_r =[]
    std_r = []
    
    X_norm = X
    
    n_c = X.shape[1]
    for i in range(n_c):
        m = np.mean(X[:,i])
        s = np.std(X[:,i])
        mean_r.append(m)
        std_r.append(s)
        X_norm[:,i] = (X_norm[:,i] - m) / s
          
    return X_norm, mean_r, std_r
    
    
def compute_cost(X, y, theta):
    '''
    COmpute cost for linear regression
    '''
    
    #number of training examples
    m = y.size
     
    predictions = X.dot(theta)
    
    loss = (predictions - y)
    
    J = np.sum(loss ** 2) / (2 *m)      
                        
    return J
    

def gradient_descent(X, y, theta, alpha, num_iters):
    '''
    Performs gradient descent to learn theta
    by taking num_items gradient steps with learning
    rate alpha
    '''
    J_history = zeros(shape=(num_iters, 1))    
    for i in range(num_iters):

        hypothesis = X.dot(theta)
        
        loss = (hypothesis - y)
        
        grad = np.dot(X.transpose(),loss) / m              
        
        theta = theta - alpha * grad

        J_history[i, 0] = compute_cost(X, y, theta)

    return theta, J_history




data = loadtxt('ex1data2.txt', delimiter=',')

X = data[:, :-1]
y = data[:,2]
y.shape = (y.size,1)


#number of dataset examples
m = y.size
#number of features
n = len(X[1,:]) + 1


#Scale features
x, mean_r, std_r = feature_normalize(X)

## Add intercept term
it = ones(shape=(m, n))
it[:, 1:3] = x
  
#Gradient settings
theta = zeros((n,1))
iterations = 100
alpha = 0.01
  
print (compute_cost(it,y,theta))

theta, J_history = gradient_descent(it, y, theta, alpha, iterations)

print (theta, J_history)
plt.figure(1)
plot( range(iterations), J_history, color = 'purple')
xlabel('Iterations')
ylabel('Cost Function')
show()


##Predicting

predict1 = array([1.0,(1650.0 - mean_r[0] )/ (std_r[0]), (3 - mean_r[1])/ (std_r[1])]).dot(theta)
print ('For a house with 1650m² and 3 bedrooms we predict a price of %f' % (predict1))























