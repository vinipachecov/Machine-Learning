#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 11 18:46:37 2017

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

                       
# Fitting Linear regression to the dataset                       
from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
linreg.fit(X,Y)
                       
# Fitting polynomial regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)
#poly_reg.fit(X_poly,Y)
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly,Y)

# VIsualising the linear regression results
plt.scatter(X,Y, color= 'red')
plt.plot(X,linreg.predict(X), color= 'blue')
plt.title("Truth of bluff (Linear Regression)")
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()


#visualising the polunomial regression results
#X_grid = np.arange(min(X), max(X), 0.1)
#X_grid =X_grid.reshape((len(X_grid)),1)
plt.scatter(X,Y, color= 'red')
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)), color= 'blue')
plt.title("Truth of bluff (Polynomial Regression)")
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()


#Predicting a new result with Linear Regression
linreg.predict(6.5)

#Predicting a new result with polynomial Regression
lin_reg2.predict(poly_reg.fit_transform(6.5))
   