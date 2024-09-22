import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt

reg = make_regression(n_samples=100, n_features=1, noise=10)

x = reg[0]
y = reg[1]
x = x.reshape(x.shape[0], 1)
y = y.reshape(y.shape[0], 1)

def showPlot(name="f(x)"):
	plt.scatter(x,y, label=name)
	plt.legend()
	print("data generated randomly", "X : {} ; Y : {}".format(x.shape, y.shape))
	plt.show()

##########################
#         MODEL          #
##########################

X = np.hstack((x, np.ones(x.shape)))
theta = np.random.randn(2, 1)

def resetTheta():
	global theta
	theta = np.random.randn(2, 1)

def model(X, theta):
	return X.dot(theta)

def showModel(name="f(x)"):
	plt.scatter(x,y, label=name)
	plt.legend()
	plt.plot(x, model(X,theta), c='red')
	plt.show()

##########################
#     COST FUNCTION      #
##########################

def costFunction(X, y, theta):
	m = len(y)
	return 1/(2*m) * np.sum((model(X,theta)-y)**2)

##########################
#    GRADIENT DESCENT    #
##########################

def grad(X, y, theta):
	m = len(y)
	return 1/m * X.T.dot(model(X,theta)-y)

def gradientDescent(X, y, theta, learning_rate, iterations):
	for i in range(0,iterations):
		theta = theta - learning_rate*grad(X,y,theta)
	return theta

##########################
#       LEARNING         #
##########################

def startLearning(learning_rate=0.001, iterations=1000):
	final_theta = gradientDescent(X,y,theta, learning_rate, iterations)
	


DISPATCHER_P = {'showp' :showPlot,
							 'showm' :showModel}

DISPATCHER_N = {'reset':resetTheta}



