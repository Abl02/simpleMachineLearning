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
	plt.scatter(x,y,label=name)
	plt.legend()
	print("data generated randomly", "X : {} ; Y : {}".format(x.shape, y.shape))
	plt.show()

##########################
#         MODEL          #
##########################

X = np.hstack((x, np.ones(x.shape)))
theta = np.random.randn(2, 1)
cost_history = np.zeros(1)
prev_predictions = []

def resetTheta():
	global theta, prev_predictions, cost_history
	prev_predictions = []
	cost_history = np.zeros(1)
	theta = np.random.randn(2, 1)

def model(X, theta):
	return X.dot(theta)

def showModel(show_predictions:bool=False):
	if show_predictions:
		for p in prev_predictions:
			plt.plot(x,model(X,p),c='yellow',linestyle='dashed',alpha=0.3,aa=1)
	plt.scatter(x,y,label='X')
	plt.plot(x,model(X,theta),c='red',aa=1,label='prediciton')
	plt.legend()
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
	cost_history = np.zeros(iterations)
	for i in range(0,iterations):
		theta = theta - learning_rate*grad(X,y,theta)
		cost_history[i] = costFunction(X,y,theta)
		if (i%100)==0:
			global prev_predictions
			prev_predictions.append(theta.copy())
	return theta, cost_history

##########################
#       LEARNING         #
##########################

def makePrediction(l, i):
	return gradientDescent(X,y,theta, l, i)

def showCost():
	plt.plot(cost_history)
	plt.legend()
	plt.show()

def startLearning(learning_rate=0.01, iterations=1000):
	global theta, cost_history
	theta, cost_history = makePrediction(learning_rate, iterations)

##########################
#      DETERMINATION     #
##########################
	
def determination(y, pred):
	u = ((y-pred)**2).sum()
	v = ((y-y.mean())**2).sum()
	return 1-u/v

def calcEfficiency():
	efficiency = determination(y,model(X,theta))
	print(efficiency)
	
DISPATCHER_P = {'showp'   :showPlot,
								'showm'   :showModel,
								'showcost':showCost}

DISPATCHER_N = {'reset'   :resetTheta,
								'start'   :startLearning,
								'effic'   :calcEfficiency}


