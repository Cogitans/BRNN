import numpy as np 
import os, pickle
import matplotlib.pyplot as plt 

DIR = "../results/moldau_discrete/"

for i in [0, 1]:
	losses = [pickle.load(open(DIR + path, "rb"))[i] for path in os.listdir(DIR) if path.split(".")[-1] != "weights"]
	for name in ["Clockwork", "BiasVRNN"]:
		plt.clf()
		fig, ax = plt.subplots()
		for n in np.arange(len(losses)):
			loss = losses[n]
			path = os.listdir(DIR)[n]
			if path.split("_")[1] == name:
				try:
					amount = 1#int(path.split("_")[3].split(".")[0])
				except:
					amount = 1
				xs = np.arange(0, len(loss) * amount, amount)
				if True:#path.split("_")[2].split(".")[0] == "GRU":
					p = plt.plot(xs, loss, label=path)
		legend = ax.legend()
		plt.show()
