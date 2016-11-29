import numpy as np 
import os, pickle
import matplotlib.pyplot as plt 

DIR = "../results/shakespeare_base2/"

for i in [0, 1]:
	losses = [pickle.load(open(DIR + path, "rb"))[i] for path in os.listdir(DIR)]
	fig, ax = plt.subplots()
	for n in np.arange(len(losses)):
		loss = losses[n]
		path = os.listdir(DIR)[n]
		amount = int(path.split("_")[3].split(".")[0])
		xs = np.arange(0, len(loss) * amount, amount)
		if True:#path.split("_")[2].split(".")[0] == "GRU":
			p = plt.plot(xs, loss, label=path)
	legend = ax.legend()
	plt.show()
