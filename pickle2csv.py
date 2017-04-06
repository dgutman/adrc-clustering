import csv
import pickle
import json

data = pickle.load(open("WordFluencyMultiTest.pickle", "rb"))

with open("WordFluencyMultiTest.csv", "w") as fh:

	w = csv.writer(fh, delimiter='\t')
	w.writerow(data[0].keys())

	for record in data:
		w.writerow(record.values())
