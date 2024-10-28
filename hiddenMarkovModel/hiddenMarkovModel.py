#! /home/abel/Documents/simulation/simpleNeuralNetwork/env/bin/python3

import numpy as np

transition = np.array([[0.7, 0.3],
											 [0.5, 0.5],
											 [0.4, 0.6]])
emission = np.array([[0.8, 0.1, 0.1],
										 [0.2, 0.3, 0.5]])

print(transition, emission, sep='\n')

colors = [1, 2, 0]

def maximizeProba(c_seq):
	p = 1
	moods = list()
	max_p = emission.argmax(0)
	for i in emission[max_p, range(len(max_p))]:
		p *= i
	moods = max_p[colors]
	ptr = 2
	for i in range(len(moods)):
		p *= transition.item(ptr, moods[i])
		ptr = moods[i]
	print(p)
	return moods

print(maximizeProba(colors))
