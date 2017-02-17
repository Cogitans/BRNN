import pickle
import numpy as np
from Tkinter import *
from utils import *

DATA = "../datasets/"
TEXT = DATA + "shakespeare/s.txt"
num_classes, c_to_l, l_to_c = p_char_mapping(TEXT)


x = pickle.load(open("../results/shakeweight/layer_2.a", "rb"))
pairs, d = x
xs, acts = [], []
for act, x in pairs:
	xs.append(x)
	acts.append(act)

letters = ""
inds = []

template = """
<html>
<head>
</head>
<body>
<div>
{0}
</div>
</body>
</html>
"""


activations = np.vstack(acts)[:-1, :]
m = np.max(activations) 
activations = activations / m * 255

d = l_to_c

for x in xs:
	inds += np.argmax(x, axis=1).tolist()

for ind in inds:
	letters += d[ind]

for activation in np.arange(0, 1):
	text = ""
	for i in range(len(letters)):
		z = activations[i, activation]
		if z < 0:
			mycolor = (255 + z, 255 + z, 255) 
		else:
			mycolor = (255, 255 - z, 255 - z) 
		text += """<span "style=color: rgb({0},{1},{2});">{3}</span>""".format(mycolor[0], mycolor[1], mycolor[2], letters[i])
		if letters[i] == '\n':
			text += "</br>"
	# with open("myhtml.html", "wb") as f:
		# f.write(template.format(text))


for activation in np.arange(150, 170):
	root = Tk()
	text = Text(root)
	text.insert(INSERT, letters)
	line = 1
	col = 0
	for i in range(len(letters)):
		if letters[i] == "\n":
			line += 1
			col = 0
			continue
		z = activations[i, activation]
		if z < 0:
			mycolor = '#%02x%02x%02x' % (255 + z, 255 + z, 255) 
		else:
			mycolor = '#%02x%02x%02x' % (255, 255 - z, 255 - z) 
		text.tag_add(str(line)+"."+str(col), str(line)+"."+str(col))
		text.tag_config(str(line)+"."+str(col), background=mycolor)
		col += 1

	text.pack()
	root.mainloop()
