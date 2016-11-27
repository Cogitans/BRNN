import numpy as np 
import os, pickle
import matplotlib.pyplot as plt 

<<<<<<< Updated upstream
DIR = "../results/timesteps/"
=======
DIR = "../old/early_losses_gru_identity/"
>>>>>>> Stashed changes

losses = [pickle.load(open(DIR + path, "rb"))[0] for path in os.listdir(DIR)]
fig, ax = plt.subplots()
for n in np.arange(len(losses)):
	loss = losses[n]
	path = os.listdir(DIR)[n]
<<<<<<< Updated upstream
	amount = int(path.split("_")[3].split(".")[0])
	xs = np.arange(0, len(loss) * amount, amount)
	if path.split("_")[2].split(".")[0] == "GRU":
		p = plt.plot(xs[::10], loss[::10], label=path)
=======
	if True:#path.split("_")[2].split(".")[0] == "Clockwork":
		p = plt.plot(np.arange(len(loss)), loss, label=path)
>>>>>>> Stashed changes
legend = ax.legend()
plt.show()
