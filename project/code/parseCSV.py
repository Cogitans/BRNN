DIR = "../datasets/shakespeare/"
PATH = DIR + "will_play_text.csv"
OUTPATH = DIR + "s.txt"


import csv
import pickle

texts = {}

with open(PATH, "rb") as csvfile:
	read = csv.reader(csvfile, delimiter=';', )
	lines = []
	for row in read:
		linenum, play, charnum, line, char, line = row
		if len(line) == 0:
			continue
		lines.append(line)
f = open(OUTPATH, "wb")
pickle.dump(["\n".join(lines)], f)
f.close()


		
