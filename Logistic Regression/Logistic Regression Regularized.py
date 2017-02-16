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
##########################################
def costReg(theta, X, y, learningRate):  
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    first = np.multiply(-y, np.log(sigmoid(X * theta.T)))
    second = np.multiply((1 - y), np.log(1 - sigmoid(X * theta.T)))
    reg = (learningRate / 2 * len(X)) * np.sum(np.power(theta[:,1:theta.shape[1]], 2))
    return np.sum(first - second) / (len(X)) + reg




def gradientReg(theta, X, y, learningRate):  
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)

    parameters = int(theta.ravel().shape[1])
    grad = np.zeros(parameters)

    error = sigmoid(X * theta.T) - y

    for i in range(parameters):
        term = np.multiply(error, X[:,i])

        if (i == 0):
            grad[i] = np.sum(term) / len(X)
        else:
            grad[i] = (np.sum(term) / len(X)) + ((learningRate / len(X)) * theta[:,i])

    return grad
###########################################

    
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
    exp1 = (-y.T.dot(np.log(hypothesis)))
    exp2 = ((1.0 - y).T.dot(np.log(1.0 - hypothesis)))    
    J = (exp1  - exp2).dot(1/m) 
        
    return J.sum() + reg.sum() 



def grad(theta,X,y,learningRate):    
    
    theta = theta.T          
    X = np.matrix(X)
    y = np.matrix(y)
    m = y.shape[0]
    theta0 = np.zeros(X.shape[1])      
    theta0[1:] = theta[1:]    
    theta = np.matrix(theta)    
    theta0 = np.matrix(theta0)
    
    reg = np.dot(learningRate / m, theta)
    
    Z = X.dot(theta.T)    
    hypothesis = sigmoid(Z)      
    error = hypothesis - y    
    print(reg)
    grad =  np.dot((X.T.dot(error).flatten()),1/m)  + reg
                  
#    theta.shape = (3,) 
    
    return grad          
    
##
def predict(theta, X):    
    probability = sigmoid(X * theta.T)
    return [1 if x >= 0.5 else 0 for x in probability]          
    
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

data2.insert(3, 'Ones', 1)

for i in range(1, degree):  
    for j in range(0, i):
        data2['F' + str(i) + str(j)] = np.power(x1, i-j) * np.power(x2, j)

data2.drop('Test 1', axis=1, inplace=True)  
data2.drop('Test 2', axis=1, inplace=True)



# set X and y (remember from above that we moved the label to column 0)
cols = data2.shape[1]  
X2 = data2.iloc[:,1:cols]  
y2 = data2.iloc[:,0:1]

# convert to numpy arrays and initalize the parameter array theta
X2 = np.array(X2.values)  
y2 = np.array(y2.values)  
theta2 = np.zeros(11)

learningRate = 1

costReg(theta2, X2, y2, learningRate)        
result2 = opt.fmin_tnc(f=compute_cost,x0=theta2,fprime=grad,args=(X2,y2,learningRate))



theta_min = np.matrix(result2[0])  
predictions = predict(theta_min, X2)  
correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(predictions, y2)]  
accuracy = (sum(map(int, correct)) % len(correct))  
print ('accuracy = {0}%'.format(accuracy))


#grad(theta2,X2,y2,learningRate)
###########################################



print( compute_cost(theta2, X2,y2,learningRate= lr))
result = opt.fmin_tnc(func=compute_cost, x0=theta2,fprime=grad, args=(X2,y2,lr))
print(result[0])
opt.fmin_ncg(f=compute_cost,x0=theta2, fprime=grad, args=(X2,y2,lr))
        


