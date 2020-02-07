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
#from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import statsmodels.api as sm 


#%%
df = sm.datasets.fair.load_pandas().data
df.head() 

# %%
def affair_check(x):
    if x != 0:
        return 1
    else:
        return 0

df['affairs'] = df['affairs'].apply(affair_check)
df.head()

df.groupby('affairs').mean() 

sns.catplot('age', hue='affairs', kind='count', data=df) 
sns.catplot('yrs_married', hue='affairs', kind='count', data=df) 
# %% New Dataframes for Categorical variables

occ_dummies = pd.get_dummies(df['occupation'])
hus_occ_dummies = pd.get_dummies(df['occupation_husb'])

occ_dummies.columns = ['occ1', 'occ2', 'occ3', 'occ4', 'occ5', 'occ6']
hus_occ_dummies.columns = ['hocc1', 'hocc2', 'hocc3', 'hocc4', 'hocc5', 'hocc6']

X = df.drop(['occupation', 'occupation_husb', 'affairs'], axis=1)

dummies = pd.concat([occ_dummies, hus_occ_dummies], axis=1)

X = pd.concat([X, dummies], axis=1)

X.head()

# %% Taking care of multicollinearity
#ran X.drop(['hocc1'], axis=1) here

Y = df.affairs

Y = np.ravel(Y)

# %%Logistic Regression Model

log_model = LogisticRegression()

log_model.fit(X, Y)
log_model.score(X, Y)

#Percentage of women that had an affair
#Y.mean()



# %% Coefficient

coeff_df = DataFrame(zip(X.columns, np.transpose(log_model.coef_)))


# %% Testing and Training sets

X_train, X_test, Y_train, Y_test = train_test_split(X, Y) 

print (X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)
log_model2 = LogisticRegression(max_iter=1000)

log_model2.fit(X_train, Y_train)

class_predict = log_model2.predict(X_test)
print (sklearn.metrics.accuracy_score(Y_test, class_predict))
