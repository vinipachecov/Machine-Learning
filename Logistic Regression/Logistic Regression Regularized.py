#USING PYTHON 2.7
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
    exp1 = (-y.T.dot(np.log(hypothesis)))
    exp2 = ((1.0 - y).T.dot(np.log(1.0 - hypothesis)))    
    J = (exp1  - exp2)/m
        
    return J.sum() + reg.sum() 


def grad(theta,X,y,learningRate):    
    
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    m = y.shape[0]
    theta0 = np.zeros((1,X.shape[1]))      
    theta0[0,1:] = theta[0,1:]  
    theta = np.matrix(theta)    
    theta0 = np.matrix(theta0)
    
    reg = np.dot(learningRate / m, theta0)
    
    Z = X.dot(theta.T)    
    hypothesis = sigmoid(Z)      
    error = hypothesis - y        
    grad =  (X.T.dot(error)/m)  + reg.sum()                    
    
    return grad.T          
    
##
def predict(theta, X):    
    probability = sigmoid(X * theta.T)
    return [1 if (x >= 0.5) else 0 for x in probability]          
    
####################################
## Running settings
data2 = pd.read_csv('ex2data2.txt', header=None, names=['Test 1', 'Test 2', 'Accepted'])

positive = data2[data2['Accepted'].isin([1])]
negative = data2[data2['Accepted'].isin([0])]

fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(positive['Test 1'], positive['Test 2'], s=50, c='b', marker='o', label='Accepted')
ax.scatter(negative['Test 1'], negative['Test 2'], s=50, c='r', marker='x', label= 'Rejected')
ax.legend()
ax.set_xlabel('Test 1 Score')

##########################################
### Minizing with all features that Andrew's uses
##Trying  with all 28 high order features
#y = data2[data2.columns[-1]].as_matrix()
#m = len(y)
#y = y.reshape(m, 1)
#X = data2[data2.columns[:-1]]
#X = X.as_matrix()
#
#from sklearn.preprocessing import PolynomialFeatures
#
#feature_mapper = PolynomialFeatures(degree=6)
#X = feature_mapper.fit_transform(X)


degree = 5
x1 = data2['Test 1']
x2 = data2['Test 2']

data2.insert(3, 'Ones', 1)

for i in range(1, degree):
    for j in range(0, i):
        data2['F' + str(i) + str(j)] = np.power(x1, i-j) * np.power(x2, j)

data2.drop('Test 1', axis=1, inplace=True)
data2.drop('Test 2', axis=1, inplace=True)

#########################3
# set X and y (remember from above that we moved the label to column 0)
cols = data2.shape[1]
X = data2.iloc[:,1:cols]
y = data2.iloc[:,0:1]

# convert to numpy arrays and initalize the parameter array theta
X = np.array(X.values)
y = np.array(y.values)
theta = np.zeros(11)
##############################3

# convert to numpy arrays and initalize the parameter array theta

theta = np.zeros(X.shape[1])

learningRate = 1

compute_cost(theta, X, y, learningRate)        
grad(theta,X,y,learningRate)

result = opt.fmin_tnc(func=compute_cost,x0=theta,fprime=grad,args=(X,y,learningRate))
## With bfgs and not using my gradient function
#result = opt.fmin_bfgs(f=compute_cost,x0=theta,args=(X,y,learningRate))


theta_min = np.matrix(result[0])  
predictions = predict(theta_min, X)  
correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(predictions, y)]  
accuracy = (sum(map(int, correct)) % len(correct))  
print 'accuracy = {0}%'.format(accuracy)
#############################################