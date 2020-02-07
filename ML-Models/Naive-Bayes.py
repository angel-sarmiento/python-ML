#%%
"Preamble for importing libraries"

import numpy as np
import pandas as pd 
from pandas import Series, DataFrame

from scipy import stats

import seaborn as sns

import matplotlib as mlib
import matplotlib.pyplot as plt 

import sklearn 
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_iris

#%%
# load the iris datasets
iris = load_iris()

# Grab features (X) and the Target (Y)
X = iris.data

Y = iris.target

# Show the Built-in Data Description
#print iris.DESCR

#%%
model = GaussianNB()

# Split the data into Trainging and Testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y)

model.fit(X_train,Y_train)


# Predicted outcomes
predicted = model.predict(X_test)

# Actual Expected Outvomes
expected = Y_test

print (accuracy_score(expected, predicted))

# %%
