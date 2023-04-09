from sklearn. linear_model import LogisticRegression
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import sklearn
from sklearn import linear_model


data = pd. read_csv('D:\\SSD_Python01012023\\SP1.csv')
s = data[[ 'HomeTeam','AwayTeam', 'FTHG', 'FTAG', 'FTR']]

print(s.head()) # (H=Home Win, D=Draw, A=Away Win)

# Visualization with scatter the number of goals
plt.scatter(s.FTHG, s.FTAG, s = 100, alpha = 0.05) 
# s, size of the points, alpha, blending value, between 0 (transparent) and 1 (opaque).
plt.xlabel('Home team goals (FTHG)')
plt.ylabel('Away team goals (FTAG)')

plt.show()

def logist(x,l):
    return 1/(1+np.exp(-l*x))

x = np.linspace(-10, 10) # 50 points equally spaced from -10 to 10
t = logist(x, 0.5)
y = logist(x, 1)
z = logist(x, 3)
plt.plot(x, t, label = 'lambda=0.5')
plt.plot(x, y, label = 'lambda=1')
plt.plot(x, z, label = 'lambda=3')
plt.legend(loc = 'upper left')
plt.savefig("D:\\SSD_Python01012023\\LogisticRegression.png", dpi = 300, bbox_inches = 'tight')
plt.show()


xmin, xmax = -10, 10
np.random.seed(0)
X = np.random.normal(size = 100) 

y = (X > 0).astype(float) 
X = X[:, np.newaxis] 

# Linear Regression:
ols = linear_model.LinearRegression()
ols.fit(X, y)
plt.plot(X_test, ols.coef_ * X_test + ols.intercept_, color = 'blue', linewidth = 2)

# Logistic Regression:
clf = linear_model.LogisticRegression(C=1e5)
clf.fit(X, y)

# Drawing:
X_test = np.linspace(-10, 10, 300)
loss = lr_model(X_test * clf.coef_ + clf.intercept_).ravel()
plt.plot(X_test, loss, color = 'red', linewidth = 2)

plt.axhline(0.5, color = 'black')

plt.scatter(X, y, color = 'black')

plt.legend(loc = 'lower right')
plt.ylabel('y')
plt.xlabel('x')

plt.ylim(-1, 2)
plt.xlim(-3, 3)

plt.savefig("D:\\SSD_Python01012023\\LinearvsLR.png", dpi = 300, bbox_inches = 'tight')
plt.show()

def lr_model(x):
    return 1 / (1 + np.exp(-x))
