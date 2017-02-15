import numpy as np 
import os, pickle
import matplotlib.pyplot as plt 
import seaborn as sns
sns.set(palette="muted", color_codes=True)
DIR = "../results/scale_tests/"

names = {"Clockwork": ["CWRNN w/Contextual Discretization", "CWRNN w/Standard Binarization", "Full Precision CWRNN"], "BiasVRNN": ["VRNN w/Contextual Discretization", "VRNN w/Standard Binarization", "Full Precision VRNN"]}

for i in [1]:#, 1]:
	losses = [pickle.load(open(DIR + path, "rb"))[i] for path in os.listdir(DIR) if path.split(".")[-1] != "weights"]
	#for name in ["GRU", "ScaleGRU", "Clockwork", "BiasVRNN"]:
	fig, ax = plt.subplots()
	#j = 0
	print(len(losses))
	for n in np.arange(len(losses)):
		loss = losses[n]
		path = os.listdir(DIR)[n]
		if True:#len(path.split("_")) > 1:# and path.split("_")[1] == name:
			try:
				amount = 1#int(path.split("_")[3].split(".")[0])
			except:
				amount = 1
			xs = np.arange(0, len(loss) * amount, amount)
			if True:#path.split("_")[2].split(".")[0] == "GRU":
				p = plt.plot(xs, loss, label=path)#names[name][j])
			#j += 1
	legend = ax.legend(fontsize=13, markerscale=2)
	plt.show()
