import re
import csv
import itertools
import numpy as np
import networkx as nx
from autocorrect import spell
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from sklearn.preprocessing import normalize, scale
from utils import *

def read_data_from_file(filename):
	""" read_data_from_file()
	Read patient data from tab delimited file and converts
	into dictionary
	"""
	patients = []
	index = 0

	with open(filename, "r") as fh:
		lines = csv.DictReader(fh, delimiter="\t")
		for line in lines:
			p = lambda: None
			p.data = {k:get_float(v) for k,v in line.iteritems()}
			p.index = index
			p.features = {}
			patients.append(p)
			index += 1

	return patients

def aggregate_fluency_tests(patients):
	""" aggregate_fluency_tests()
	Given patient dictionary, this function will look at specific keys,
	namely the animal verbal fluency keys (avf_anm) and will aggregate
	all words across the 4 tests into one set	
	"""
	pattern = re.compile("^avf_anm\d+$")

	for p in patients:
		words = [val.split(",") for key, val in p.data.iteritems() if pattern.match(key) and isinstance(val, str)]
		words = list(itertools.chain(*words))
		p.words, p.data["errors"] = word_cleanup(words)
		p.data["error_count"] = len(p.data["errors"])
		patients[p.index] = p

	return patients

def word_graph(patients):
	""" word_graph()
	For each patient use the aggregated verbal fluency data
	to generate graph using NetworkX and generate the features
	based on the graph
	"""
	for p in patients:
		G = nx.Graph()

		if len(p.words) == 1:
			G.add_node(p.words[0])
		else:
			[G.add_edge(p.words[i-1], p.words[i])
				for i in range(1, len(p.words))]

		p.graph = G
		gf = graph_features(G)
		p.data["edge_count"] = gf["edgeCount"]
		p.data["node_count"] = gf["nodeCount"]
		p.data["diameter"] = gf["diameter"]
		p.data["cycle_count"] = gf["cycleCount"]
		p.data["longest_cycle"] = gf["longestCycle"]

		patients[p.index] = p

	return patients

def graph_features(G):
	""" graph_features()
	Given a graph compute some features including # edges,
	# nodes, # cycles, diameter and longest cycle
	"""
	cycles =  nx.cycle_basis(G)
	longestCycle = 0

	if len(cycles):
		longestCycle = max([len(cycle) for cycle in cycles])

	return {
		"edgeCount":    len(G.edges())  if len(G.nodes()) else 0,
		"nodeCount":    len(G.nodes())  if len(G.nodes()) else 0,
		"diameter":     nx.diameter(G)  if len(G.nodes()) else 0,
		"cycleCount":   len(cycles)     if len(G.nodes()) else 0,
		"longestCycle": longestCycle    if len(G.nodes()) else 0
	}

def word_cleanup(words):
	""" word_cleanup()
	Given list of words, try to clean them up and only return valid words.
	The function will return 2 lists: valid words and erred words
	"""
	valid_words = []
	errors = []

	for word in words:
		word, valid = is_word_valid(word)

		if valid == False:
			errors.append(word)
		else:
			valid_words.append(word)

	return valid_words, errors

def is_word_valid(word):
	""" is_word_valid()
	Given a word check if it valid.
		Lemmatize it
		Convert it to lower case
		Spell check it
	"""
	lmtzr = WordNetLemmatizer()
	word = re.sub("\s*-\s*p|\s*\(p\)", "", word).lower()
	lemmas = [lmtzr.lemmatize(spell(w)) for w in word.split() if re.match('\w{2,}\s*',w)]
	hasSynsets = min([len(wn.synsets(w)) for w in lemmas] or [0])

	if hasSynsets == 0:
		return " ".join(lemmas), False
	else:
		return " ".join(lemmas), True

def impute(vals, f=np.median, t=0.2):
	""" impute()
	Given list of value, look for missing values and imputate
	using user defined function and using threshold. list of missing
	values greater than t will not be ignored
	"""
	array_nans_size = np.size(vals[np.isnan(vals)])
	array_nonans = vals[~np.isnan(vals)]
	percent_missing = np.float(array_nans_size) / np.float(np.size(vals))

	if percent_missing < t:
		vals[np.isnan(vals)] = f(array_nonans)
		return vals
	else:
		return None

def build_feature_matrix(patients, feature_names, f=np.median, t=0.2):
	""" build_feature_matrix()
	Given list of patients and list of features generate feature matrix
	n x m (#patients x #features)
	"""
	for feature in feature_names:
		vals = [patient.data.get(feature, np.nan) for patient in patients]
		vals = np.asarray([np.nan if type(x) == str else x for x in vals])
		vals = impute(vals)
		if vals != None:
			for patient in patients:
				patient.features[feature] = vals[patient.index]

	n = len(patients)																			# number of patients
	m = len(patients[0].features)															# number of features
	X = np.zeros(shape=(n,m))																	# Feature matrix
	feature_names = patients[0].features.keys()

	for patient in patients:
		X[patient.index] = patient.features.values()

	X = normalize(X, norm='max', axis=0)
	return patients, feature_names, X
