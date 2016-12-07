import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

with open("../results/testing/weights.weights", "rb") as f:
	x = pickle.load(f)[0]

sns.set(style="white", palette="muted", color_codes=True)
fig, ax = plt.subplots(sharey=True)
for weight in x:

	plt.clf()
	# Generate a random univariate dataset
	d = np.log2(np.abs(weight[1])).ravel()
	d = np.abs(weight[1]).ravel()
	d[d == 0] = 1e-20
	d = np.log10(d)
	name = weight[0]
	K = name.split("_")
	color = None
	if "U:0" in K or "U" in K:
		color = 'blue'
	elif "W:0" in K or "W" in K:
		color = 'red'
	elif "b:0" in K or "b" in K:
		color = 'green'
	if True:#color is not None and color != "green":

		num_bins = 100
		# the histogram of the data

		n, bins, patches = plt.hist(d, num_bins, normed=1, facecolor=color, alpha=0.5)
		# add a 'best fit' line

		# Tweak spacing to prevent clipping of ylabel
		plt.subplots_adjust(left=0.15)
	ax.tick_params(axis='both', which='major', labelsize=15)
	ax.tick_params(axis='both', which='minor', labelsize=15)
	plt.xlabel(weight[0], fontsize = 25)
	plt.show()