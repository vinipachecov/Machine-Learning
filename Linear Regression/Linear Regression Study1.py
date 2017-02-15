#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 10:27:49 2017

@author: root
"""

# Y = B0 + B1*X1 + B2*X2 + ... Bn*Xn

# Study case:
# For example, suppose you are the CEO of a big company of 
# shoes franchise and are considering different cities for opening
#a new store. The chain already has stores in various cities and you have 
#data for profits and populations from the cities.  You would like to use 
# this data to help you select which city to expand next. You could use linear
#regression for evaluating the parameters of a function that predicts profits 
#for the new store.

from numpy import loadtxt, zeros, ones, array, linspace, logspace, shape, repeat,sum,dot
import numpy as np
from pylab import scatter, show, title, xlabel, ylabel, plot, contour
import matplotlib.pyplot as plt


def compute_cost(X, y, theta):
    '''
    COmpute cost for linear regression
    '''
    
    #number of training examples
    m = y.size
     
    predictions = X.dot(theta)
    
    loss = predictions - y
    
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
        
        loss = hypothesis - y
        
        grad = np.dot(X.transpose(),loss) / m              
        
        theta = theta - alpha * grad

        J_history[i, 0] = compute_cost(X, y, theta)

    return theta, J_history
 
#Load dataset

data = loadtxt('ex1data1.txt', delimiter=',')

X = data[:,0]
y = data[:,1]
y.shape = (y.size,1)


#Plot the data

plt.scatter(X,y, color= 'blue')
plt.title("Profits distribuition")
plt.xlabel('POpulation of City in 10,000s')
plt.ylabel('Profits in 10,000s')
plt.show()

#Hypothesis Htheta= (theta)T * X = theta0 + theta1x1
# alpha 0.01

#number of training examples                    
m = y.size


#Add a column of ones to X (interception term)
it = ones(shape=(m,2))
it[:,1] = X
  
#Initialize theta parameters and gradient descent parameters
theta = zeros(shape=(2,1))
iterations = 1500
alpha = 0.01
  
#compute and display initial cost
print (compute_cost(it,y,theta))
  
theta, J_history = gradient_descent(it,y,theta,alpha,iterations)          


#Predict values for population sizes of 35,000 and 70,000

predict1 = array([1, 3.5]).dot(theta).flatten()
print ('For population = 35,000, we predict a profit of %f' % (predict1 * 10000))
predict2 = array([1,7.0]).dot(theta).flatten()
print ('For population = 70,000, we predict a profit of %f' % (predict2 * 10000))

result = it.dot(theta)

#Printing new results 35k and 70k
plt.figure(0)
plt.scatter(X,y, color= 'blue')
plt.title("Profits distribuition")
plt.xlabel('POpulation of City in 10,000s')
plt.ylabel('Profits in 10,000s')

plt.plot(X, result, color = 'red')
plt.show()


theta0_vals = linspace(-10,10,100)
theta1_vals = linspace(-1,4,100)



#initialize j_vals to a matrix of 0's
J_vals = zeros(shape=(theta0_vals.size, theta1_vals.size))

#Fiil ou J_vals
for t1, element in enumerate(theta0_vals):
    for t2, element2 in enumerate(theta1_vals):
        thetaT = zeros(shape=(2,1))
        thetaT[0][0] = element
        thetaT[1][0] = element2
        J_vals[t1][t2] = compute_cost(it,y,thetaT)
        
J_vals = J_vals.T

#Plot J_vals as a 15 contours spaced logarithmically between 0.01 and 100
plt.figure(1)
contour(theta0_vals, theta1_vals,J_vals, logspace(-2.3,20))
xlabel('theta_0')
ylabel('theta_1')
scatter(theta[0][0],theta[1][0])
show()
