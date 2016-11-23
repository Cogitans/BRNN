import numpy as np 
import os, pickle
import matplotlib.pyplot as plt 

DIR = "../datasets/fixed/"

losses = [pickle.load(open(DIR + path, "rb"))[0] for path in os.listdir(DIR)]
fig, ax = plt.subplots()
for n in np.arange(len(losses)):
	loss = losses[n]
	path = os.listdir(DIR)[n]
	p = plt.plot(np.arange(len(loss)), loss, label=path)
legend = ax.legend()
plt.show()
