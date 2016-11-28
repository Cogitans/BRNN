import csv
import numpy as np
import random

SHAKESPEARE_VALIDATION_PLAYS = ['Taming of the Shrew', 'The Tempest', 'Timon of Athens', 'Titus Andronicus', 'Troilus and Cressida', 'Twelfth Night', 'Two Gentlemen of Verona', 'A Winters Tale']

def shakespeare_raw_train_gen():
	with open('../datasets/shakespeare/will_play_text.csv', 'r') as f:
		parsed_csv = csv.reader(f, delimiter=';')
		prev_speaker = None
		for row in parsed_csv:
			linenum, play, charnum, line, char, line = row
			if play in SHAKESPEARE_VALIDATION_PLAYS:
				continue
			if len(line) == 0 or charnum == '':
				continue
			did_speaker_change = 1 if prev_speaker != charnum and prev_speaker else 0
			prev_speaker = charnum
			yield line, did_speaker_change

def shakespeare_raw_test_gen():
	with open('../datasets/shakespeare/will_play_text.csv', 'r') as f:
		parsed_csv = csv.reader(f, delimiter=';')
		prev_speaker = None
		for row in parsed_csv:
			linenum, play, charnum, line, char, line = row
			if play not in SHAKESPEARE_VALIDATION_PLAYS:
				continue
			if len(line) == 0 or charnum == '':
				continue
			did_speaker_change = 1 if prev_speaker != charnum and prev_speaker else 0
			prev_speaker = charnum
			yield line, did_speaker_change


def shakespeare_soft_train_gen():
	raw_gen = shakespeare_raw_train_gen()
	prev_line = next(raw_gen)[0]
	samples = []
	for line in raw_gen:
		samples.append((prev_line, line[0], line[1]))
		prev_line = line[0]
	random.shuffle(samples)
	for sample in samples:	
		yield sample

def shakespeare_soft_test_gen():
	raw_gen = shakespeare_raw_test_gen()
	prev_line = next(raw_gen)[0]
	for line in raw_gen:
		yield (prev_line, line[0], line[1])
		prev_line = line[0]

def shakespeare_window_gen(n):
	raw_gen = shakespeare_raw_gen()
	window = []
	window_change = []
	for i in xrange(n):
		line = next(raw_gen)
		window.append(line[0])
		window_change.append(line[1])
	for line in raw_gen:
		yield window, np.array(window_change)
		window.pop(0)
		window_change.pop(0)
		window.append(line[0])
		window_change.append(line[1])

def wilde_raw_gen():
	for i in xrange(1,4):
		with open('../datasets/wilde_{}_parsed.txt'.format(i), 'r') as f:
			parsed_csv = csv.reader(f, delimiter=';')
			prev_speaker = None
			for row in parsed_csv:
				if len(row) < 2:
					continue
				if len(row) > 2:
					speaker = row[0]
					line = '.'.join(row[1:])
				else:
					speaker, line = row
				did_speaker_change = 1 if prev_speaker != speaker and prev_speaker else 0
				prev_speaker = speaker
				yield line, did_speaker_change

def wilde_soft_gen():
	raw_gen = wilde_raw_gen()
	prev_line = next(raw_gen)[0]
	for line in raw_gen:
		yield (prev_line, line[0], line[1])
		prev_line = line[0]

def movie_raw_gen():
	with open('../datasets/movies.txt', 'r') as f:
		parsed_csv = csv.reader(f, delimiter=';')
		prev_speaker = None
		for row in parsed_csv:
			if len(row) == 1:
				prev_speaker = None
				continue
			speaker, _, line = row
			if len(line) == 0:
				continue
			did_speaker_change = 1 if prev_speaker != speaker and prev_speaker else 0
			prev_speaker = speaker
			yield line, did_speaker_change

def movie_soft_gen():
	raw_gen = movie_raw_gen()
	prev_line = next(raw_gen)[0]
	for line in raw_gen:
		yield (prev_line, line[0], line[1])
		prev_line = line[0]
