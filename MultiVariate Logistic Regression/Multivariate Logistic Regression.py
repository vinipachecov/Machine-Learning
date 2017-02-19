#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 22:21:38 2017

@author: root
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.optimize import minimize
import scipy.optimize as opt  


def sigmoid(Z):
    return 1.0 / (1.0 + np.exp( -1.0 * Z))


## My const Function
#def compute_cost(theta,X,y, learningRate):
#
#            
#    theta = np.matrix(theta)
#    X = np.matrix(X)
#    y = np.matrix(y)
#    m = y.size
#    theta0 = np.zeros((1,X.shape[1]))
#    theta0[0,1:] = theta[0,1:]    
#    
#    reg = np.multiply((learningRate/(2.0*m)),(theta0.T.dot(theta0))) 
#    
##    Z = X.dot(theta.T)
#    Z = X * theta.T
#    
#    hypothesis = sigmoid(Z)  
#    exp1 = (-y.T.dot(np.log(hypothesis)))
#    exp2 = ((1.0 - y).T.dot(np.log(1.0 - hypothesis)))    
#    J = (exp1  - exp2)/ m
#        
#    return J.sum() + reg.sum() 


##Given Cost FUnction
def cost(theta, X, y, learningRate):  
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    first = np.multiply(-y, np.log(sigmoid(X * theta.T)))
    second = np.multiply((1 - y), np.log(1 - sigmoid(X * theta.T)))
    reg = (learningRate / 2 * len(X)) * np.sum(np.power(theta[:,1:theta.shape[1]], 2))
    return np.sum(first - second) / (len(X)) + reg

# GIVEN GRADIENT FUNCTION
#def gradient(theta, X, y, learningRate):  
#    theta = np.matrix(theta)
#    X = np.matrix(X)
#    y = np.matrix(y)
#
#    parameters = int(theta.ravel().shape[1])
#    error = sigmoid(X * theta.T) - y
#
#    grad = ((X.T * error) / len(X)).T + (learningRate / len(X) * theta)
#
#    # intercept gradient is not regularized
#    grad[0, 0] = np.sum(np.multiply(error, X[:,0])) / len(X)
#
#    return np.array(grad).ravel()



def grad(theta,X,y,learningRate):    
    
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    m = y.shape[0]
    theta0 = np.zeros((1,X.shape[1]))      
    theta0[0,1:] = theta[0,1:]  
    theta = np.matrix(theta)    
    theta0 = np.matrix(theta0)
    
#    reg = np.dot(learningRate / m, theta0)
    reg = np.multiply(learningRate / m,  theta0)
    
    Z = X.dot(theta.T)    
    hypothesis = sigmoid(Z)      
    error = hypothesis - y        
    grad =  (X.T.dot(error)/m)  + reg.sum()                    
    
    return grad          


def one_vs_all(X,y, num_labels, learning_rate):
    rows = X.shape[0]
    params = X.shape[1]
    
    all_theta = np.zeros((num_labels,params+1))
    
    X = np.insert(X, 0, values=np.ones(rows), axis=1)
    for i in range(1, num_labels + 1):
        theta = np.zeros(params + 1)
        ##Turns y_i into a binary column vector with the 1 when i==label
        y_i = np.array([1 if label == i else 0 for label in y])
        y_i = np.reshape(y_i, (rows, 1))

        # minimize the objective function
        fmin = minimize(fun=cost, x0=theta, args=(X, y_i, learning_rate), method='TNC', jac=grad)
## fmin_tnc
#        fmin = opt.fmin_bfgs(func=compute_cost,x0=theta,fprime=grad,args=(X,y_i,learning_rate))
##fmin_bfgs        
#        fmin = opt.fmin_bfgs(f=compute_cost,x0=theta,fprime=grad,args=(X,y_i,learning_rate))
        all_theta[i-1,:] = fmin.x

    return all_theta


def predict_all(X, all_theta):
    rows = X.shape[0]
    params = X.shape[1]
    num_labels = all_theta.shape[0]
    
    # same as before, insert ones to match the shape
    X = np.insert(X, 0, values=np.ones(rows), axis=1)
    
    # convert to matrices
    X = np.matrix(X)
    all_theta = np.matrix(all_theta)
    
    # compute the class probability for each class on each training instance
    h = sigmoid(X * all_theta.T)
    
    # create array of the index with the maximum probability
    h_argmax = np.argmax(h, axis=1)
    
    # because our array was zero-indexed we need to add one for the true label prediction
    h_argmax = h_argmax + 1
    
    return h_argmax
########################################################    
# Setting Up
data = loadmat('ex3data1.mat')
weights = loadmat('ex3weights.mat')
theta1, theta2 = weights['Theta1'], weights['Theta2']

rows = data['X'].shape[0]
params = data['X'].shape[1]

all_theta = np.zeros((10, params + 1))

X = np.insert(data['X'], 0, values=np.ones(rows), axis=1)
y = data['y']

theta = np.zeros(params + 1)

y_0 = np.array([1 if label == 0 else 0 for label in data['y']])
y_0 = np.reshape(y_0, (rows, 1))

all_theta = one_vs_all(data['X'], data['y'], 10 ,1)

##Pŕedictions

y_pred = predict_all(data['X'], all_theta)
correct = [1 if a == b else 0 for (a, b) in zip(y_pred, data['y'])]
accuracy = (sum(map(int, correct)) / float(len(correct)))
print 'accuracy = {0}%'.format(accuracy * 100)

####Expected results

'''Using the given cost function with my own gradient function (grad) 
   I was able to get to 97.44% of accuracy. Using both cost and grad function
   I have written I got 92.5. His Cost function is giving some zero-division 
   errors and some invalid values on element-wise np.multiply funtcion '''
   
   
### Neural Network attempt

def predict_neural_network(theta_1, theta_2, features):
    z2 = theta_1.dot(features.T)
    a2 = np.c_[np.ones((data['X'].shape[0],1)), sigmoid(z2).T]
    
    z3 = a2.dot(theta_2.T)
    a3 = sigmoid(z3)
        
    return(np.argmax(a3, axis=1)+1)


pred = predict_neural_network(theta1, theta2, X)
print('Training set accuracy: {} %'.format(np.mean(pred == y.ravel())*100))