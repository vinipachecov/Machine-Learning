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

    
def compute_cost(theta,X,y, learningRate):
    '''compute cost given '''
            
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    m = y.size
    theta0 = np.zeros((1,X.shape[1]))
    theta0[0,1:] = theta[0,1:]    
    
    reg = np.dot((learningRate/2*m),(theta0.T.dot(theta0))) 
    
    Z = X.dot(theta.T)
    
    hypothesis = sigmoid(Z)  
    exp1 = (-y.dot(np.log(hypothesis)))
    exp2 = ((1.0 - y).dot(np.log(1.0 - hypothesis)))    
    J = (exp1  - exp2) / m 
        
    return J.sum() + reg.sum() 

          
def compute_grad(theta,X,y,learningRate):    
    
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    m = y.size
    theta0 = np.zeros(X.shape[1])  
    reg = np.dot(learningRate / m, theta0)
    
    Z = X.dot(theta.T)    
    hypothesis = sigmoid(Z)      
    error = hypothesis - y    
    grad =  ((1/m) * X.T.dot(error)).flatten() + reg
    theta.shape = (3,) 
    
    return grad
    
####################################
## Running settings
# Ipython
data2 = pd.read_csv('ex2data2.txt', header=None, names=['Test 1', 'Test 2', 'Accepted'])

positive = data2[data2['Accepted'].isin([1])]
negative = data2[data2['Accepted'].isin([0])]

fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(positive['Test 1'], positive['Test 2'], s=50, c='b', marker='o', label='Accepted')
ax.scatter(negative['Test 1'], negative['Test 2'], s=50, c='r', marker='x', label= 'Rejected')
ax.legend()
ax.set_xlabel('Test 1 Score')
ax.set_ylabel('Test 2 Score')

degree = 5

x1 = data2['Test 1']
x2 = data2['Test 2']

data2.insert(0,'Ones',1)

##inserting more features
for i in range(1,degree):
    for j in range(0,i):
#        data2['F' + str(i) + str(j)] = np.power(x1,i-j) * np.power(x2,j)
        data2.insert(1,'F'+str(i) + str(j),np.power(x1,i-j) * np.power(x2,j))
        

##setting up our parameters for cost and gradient calculation
X = data2.iloc[:,:-1]
y = data2.iloc[:, -1]
theta = np.zeros(X.shape[1])
learningRate = 1

print( compute_cost(theta, X,y,learningRate= learningRate))
        

##############################################
## Running settings
#Normal Editor
data= loadtxt('ex2data1.txt', delimiter=',')


X = data[:, 0:2]
y = data[:, -1]

pos = where(y == 1)
neg = where(y == 0)

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
print (compute_grad(theta,it,y))
##gradient for initial theta (0, 0 ,0 ) should be  [ -0.1,   -12.00921659, -11.26284221]



##Uncomment for random theta values
#theta = 0.1* np.random.randn(3)

## fmin_tnc gets optimal values for alpha so we don't have to choose it randomly 
## or by gut feeling
result = opt.fmin_tnc(func=compute_cost, x0=theta, fprime=compute_grad, args=(it, y))  
##First value returned is the optimal theta parameters for the model
theta = result[0]

#Plotting the decision boundary
plot_x = array([min(it[:, 1]) - 2, max(it[:, 2]) + 2])
plot_y = (- 1.0 / theta[2]) * (theta[1] * plot_x + theta[0])
plot(plot_x, plot_y)
legend(['Decision Boundary', 'Not admitted', 'Admitted'])
#show()


def predict(theta,X):
    '''predict whether the label
    is 0 or 1 using learned logistic
    regression parameters'''
    m,n = X.shape
    p = zeros(shape=(m,1))
    
    h = sigmoid(X.dot(theta.T))
    
    for it in range(0,h.shape[0]):
        if h[it] > 0.5:
            p[it,0] = 1
        else:
            p[it,0] = 0
    
    return p

p = predict(array(theta), it)
print ('Train Accuracy: %f' % ((y[where(p == y)].size / float(y.size)) * 100.0))

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