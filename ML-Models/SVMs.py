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
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

#%%
iris = sklearn.datasets.load_iris()
X = iris.data
Y = iris.target 


X=iris.data[:, :2]
Y=iris.target
C  = 1.0

from sklearn import svm 
model = SVC(max_iter=10000)

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.4, random_state=3)

model.fit(X_train, Y_train)


from sklearn import metrics 

predicted = model.predict(X_test)

expected = Y_test 

print (accuracy_score(expected, predicted))

# %%



#%%
#lin_svc uses OnevsAll while svc uses OnevsOne

svc = svm.SVC(kernel='linear', C=C).fit(X,Y)
rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(X,Y)
poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(X,Y)
lin_svc = svm.LinearSVC(C=C).fit(X,Y)



# %% Graphing
#step size 
h = 0.02

x_min = X[:,0].min() - 1
x_max = X[:,0].max() + 1

y_min = X[:,1].min() - 1
y_max = X[:,1].max() + 1

xx,yy =np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

#titles for plots
titles = ['SVC with linear kernal', 'LinearSVC (linear kernal)', 'SVC with RBF kernal', 'SVC with plynomial (degree 3) kernel']


# %% for loop for plots

for i,clf in enumerate((svc,lin_svc,rbf_svc,poly_svc)):
    
    plt.figure(figsize=(15,15))
    
    plt.subplot(2,2,i+1)
    
    plt.subplots_adjust(wspace=0.4,hspace=0.4)
    
    Z = clf.predict(np.c_[xx.ravel(),yy.ravel()])
    
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx,yy,Z,cmap=plt.cm.terrain,alpha=0.5,linewidths = 0)
    
    plt.scatter(X[:,0],X[:,1],c=Y,cmap=plt.cm.Dark2)
    
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.xlim(xx.min(),xx.max())
    plt.ylim(yy.min(),yy.max())
    
    plt.xticks(())
    plt.yticks(())
    plt.title(titles[i])

# %%
