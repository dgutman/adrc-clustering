import csv

def file2dict(filename):
	with open(filename, "r") as fh:
		records = csv.DictReader(fh, delimiter = "\t")
		for record in records:
			print record
			break