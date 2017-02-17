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

data = loadmat('ex3data1.mat')
data['X'].shape, data['y'].shape

def sigmoid(Z):
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
    exp1 = (-y.T.dot(np.log(hypothesis)))
    exp2 = ((1.0 - y).T.dot(np.log(1.0 - hypothesis)))    
    J = (exp1  - exp2).dot(1/m) 
        
    return J.sum() + reg.sum() 

def gradient(theta, X, y, learningRate):  
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)

    parameters = int(theta.ravel().shape[1])
    error = sigmoid(X * theta.T) - y

    grad = ((X.T * error) / len(X)).T + ((learningRate / len(X)) * theta)

    # intercept gradient is not regularized
    grad[0, 0] = np.sum(np.multiply(error, X[:,0])) / len(X)

    return np.array(grad).ravel()

def one_vs_all(X, num_labels, learning_rate):
    rows = X.shape[0]
    params = X.shape[1]
    
    all_theta = np.zeros((num_labels,params+1))
    
    X = np.insert(X, 0, values=np.ones(rows), axis=1)
    for i in range(1, num_labels + 1):
        theta = np.zeros(params + 1)
        y_i = np.array([1 if label == i else 0 for label in y])
        y_i = np.reshape(y_i, (rows, 1))

        # minimize the objective function
        fmin = minimize(fun=compute_cost, x0=theta, args=(X, y_i, learning_rate), method='TNC', jac=gradient)
        all_theta[i-1,:] = fmin.x

    return all_theta
    