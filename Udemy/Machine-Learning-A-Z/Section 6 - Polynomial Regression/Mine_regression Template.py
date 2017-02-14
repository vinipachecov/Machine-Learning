#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 11 19:39:03 2017

@author: root
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#import dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[: , 1:2].values
Y = dataset.iloc[:,2].values

                
##Splitting the dataset into the tranning set and test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size = 0.2, random_state = 0)"""

#Feature Scalling making all features in a same range
#so they don't have relative weights based on their original scale
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""

# Fitting regression to the dataset


#Visualizing the regression results (for higher resolution and smoother curve)
#X_grid = np.arange(min(X), max(X), 0.1)
#X_grid =X_grid.reshape((len(X_grid)),1)
plt.scatter(X,Y, color= 'red')
plt.plot(X, regressor.predict(X), color= 'blue')
plt.title("Truth of bluff (Polynomial Regression)")
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
   


#Predicting a new result with polynomial Regression
y_pred = regressor.predict(6.5)

#visualising the polunomial regression results

plt.scatter(X,Y, color= 'red')
plt.plot(X, regressor.predict(X), color= 'blue')
plt.title("Truth of bluff (Polynomial Regression)")
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
   