import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

with open("../results/weights/weights.weights", "rb") as f:
	x = pickle.load(f)[0]

for weight in x:

	sns.set(style="white", palette="muted", color_codes=True)

	# Generate a random univariate dataset
	d = np.log(np.abs(weight[1])).ravel()

	num_bins = 100
	# the histogram of the data
	n, bins, patches = plt.hist(d, num_bins, normed=1, facecolor='green', alpha=0.5)
	# add a 'best fit' line
	plt.xlabel(weight[0])

	# Tweak spacing to prevent clipping of ylabel
	plt.subplots_adjust(left=0.15)
	plt.show()