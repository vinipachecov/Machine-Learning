#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 11 14:08:42 2017

@author: root
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#import dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:,4].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder() 
X[:,3] = labelencoder_X.fit_transform(X[:,3]) 
onehotencoder = OneHotEncoder(categorical_features= [3])
X = onehotencoder.fit_transform(X).toarray()

#Avoiding the themmy variable trap
X = X[:, 1:]                    
                
##Splitting the dataset into the tranning set and test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size = 0.2, random_state = 0)

#Feature Scalling making all features in a same range
#so they don't have relative weights based on their original scale
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""

 ## fitting multiple linear regression to the training set
 from sklearn.linear_model import LinearRegression
 regressor = LinearRegression()
 regressor.fit(X_train, y_train)
 
 # Predicting the Test set results
 y_pred = regressor.predict(X_test)
 
 #Building backward elimination the optimal model
 import statsmodels.formula.api as sm
 
 
 ## Add a column of ones
 X = np.append(arr = np.ones((50,1)).astype(int),values = X, axis = 1)
 X_opt = X[:, [0,1,2,3,4,5]]
 regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
 regressor_OLS.summary()
 #Second attempt to fit#
 X_opt = X[:, [0,1,3,4,5]]
 regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
 regressor_OLS.summary()
 X_opt = X[:, [0,3,4,5]]
 regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
 regressor_OLS.summary()
 X_opt = X[:, [0,3,5]]
 regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
 regressor_OLS.summary()
 X_opt = X[:, [0,3]]
 regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
 regressor_OLS.summary()
  
 
 