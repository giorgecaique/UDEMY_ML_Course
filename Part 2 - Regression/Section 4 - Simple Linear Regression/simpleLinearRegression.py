#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 14:46:12 2017

@author: giorgecaique
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler


dataset = pd.read_csv("Salary_Data.csv")


X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 1].values


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.33, random_state = 0)


"""sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)
sc_Y = StandardScaler()
Y_train = sc_Y.fit_transform(Y_train)
Y_test = sc_Y.fit_transform(Y_test)"""

# Simple Linear Regression

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()