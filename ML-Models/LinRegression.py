#%%
#%%
"Preamble for importing libraries"

import numpy as np
from numpy.random import randn
import pandas as pd 
from pandas import Series, DataFrame

from scipy import stats

import seaborn as sns

import matplotlib as mlib
import matplotlib.pyplot as plt 

import sklearn 
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston 

#%%

boston = load_boston()

boston_df = DataFrame(boston.data)
boston_df.columns = boston.feature_names
boston_df.head()
boston_df.describe()


boston_df['Price'] = boston.target 
sns.lmplot('RM', 'Price', data=boston_df)
# %% Single-variable Linear Regression

#X as mediam Root Values
X = boston_df.RM 

#need your values to be in a 2D array. How many values and how
#many attributes
X = np.vstack(boston_df.RM)

#list comprehension to turn X into an array format we need a = [X, 1]
X = np.array( [[value, 1] for value in X ], dtype=np.float64)

Y = boston_df.Price

m , b = np.linalg.lstsq(X,Y)[0]

# %% Getting the Error

result = np.linalg.lstsq(X, Y)

error_total = result[1]

rmse = np.sqrt(error_total/len(X))

# %%Constructing a linear model 
lreg = LinearRegression()

X_multi = boston_df.drop('Price', 1)

Y_target = boston_df.Price 

lreg.fit(X_multi, Y_target)

print("The number of coefficients was", len(lreg.coef_))
print("The approximate value of the intercept is", lreg.intercept_)

# %% What has the highest coefficients ? 

coef_df = DataFrame(boston_df.columns)
coef_df.columns = ['Features']

coef_df['Coefficient Estimates'] = Series(lreg.coef_)

coef_df


# %% Cross validation and Dataset split with Sklearn

X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, boston_df.Price)

print(X_train.shape, X_test.shape , Y_train.shape , Y_test.shape)

lreg.fit(X_train, Y_train)

pred_train = lreg.predict(X_train)
pred_test = lreg.predict(X_test)


print(" MSE with Y_train on fitted model with X_train:", np.mean((Y_train-pred_train)**2))
print(" MSE with X_test and Y_test on fitted model with X_train:", np.mean((Y_test - pred_test)**2))
# %% Residual Plot

train = plt.scatter(pred_train, (pred_train - Y_train), c='b', alpha=0.5)
test = plt.scatter(pred_test, (pred_test - Y_test), c='r', alpha = 0.5)

#horizontal line
plt.hlines(y=0, xmin=-10, xmax=50)

plt.legend((train, test), ('Training', 'Test'), loc='lower left')
plt.title('Residual Plots')
# %%
