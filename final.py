#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 23:22:35 2020

@author: aditisaxena
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 00:01:15 2020

@author: aditisaxena
"""

# Random Forest Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
# Importing the dataset
dataset = pd.read_csv('DASS.csv')
X = dataset.iloc[:,0:21].values

y = dataset.iloc[:, 22:25].values

# Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 0:21])
X[:, 0:21] = imputer.transform(X[:, 0:21])




# Splitting the dataset into the Training set and Test seT
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Fitting Random Forest Regression to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X, y)

# Predicting a new result
y_pred = regressor.predict(X_test)

pickle.dump(regressor,open('Predictor1.pkl','wb'))
model = pickle.load(open('Predictor1.pkl','rb'))

