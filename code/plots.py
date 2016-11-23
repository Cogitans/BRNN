import pickle
import matplotlib.pyplot as plt 
import os
import re

numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

hessians, gradients, weights = [], [], []
for path in sorted(os.listdir("../datasets/graphing_checkpoints/"), key=numericalSort):
	with open("../datasets/graphing_checkpoints/" + path, "rb") as f:
		a, b, c = pickle.load(f)
		hessians.append(a[0])
		gradients.append(b[0])
		weights.append(c)

print weights[0]


