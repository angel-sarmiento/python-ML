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
#from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

#%% 
iris = load_iris()

X = iris.data

Y = iris.target

iris_data = DataFrame(X, columns=['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width'])

iris_target = DataFrame(Y, columns=['Species'])


def flower(num):
    if num == 0:
        return 'Setosa'
    elif num ==1:
        return 'Versicolour'
    else: 
        return 'Virginica'

iris_target['Species'] = iris_target['Species'].apply(flower)

iris = pd.concat([iris_data, iris_target], axis=1)
# %%
sns.pairplot(data=iris, hue='Species', height=2)

# %% Sci-kit learn for Multi-Class Classification

logreg = LogisticRegression(max_iter=1000)
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.4, random_state=3)

logreg.fit(X_train,Y_train)

from sklearn import metrics
Y_pred = logreg.predict(X_test)

print (accuracy_score(Y_test, Y_pred))


# %% K Nearest Neighbors

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(X_train, Y_train)

Y_pred = knn.predict(X_test)

print (accuracy_score(Y_test, Y_pred))

# %% Looping through to find optimal K values 

k_range = range(1,21)

accuracy = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, Y_train)
    Y_pred = knn.predict(X_test)    
    accuracy.append(accuracy_score(Y_test, Y_pred))

# %% Plotting it with Matplot lib

plt.plot(k_range, accuracy)
plt.xlabel('K Value')
plt.ylabel('Testing Accuracy')

