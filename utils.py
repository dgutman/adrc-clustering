import re
import itertools
import networkx as nx
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial.distance import pdist, squareform
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter
from autocorrect import spell
import csv
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk import ngrams
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from tfidf import *

def getPatients(data):
	patients = []
	p = re.compile("^avf_anm\d+$")
	index = 0

	for i in range(0, len(data)):
		patient = lambda: None

		words = [val.split(",") for key, val in data[i].iteritems() if p.match(key) and val != ""]
		words = list(itertools.chain(*words))
		patient.words, patient.errors = word_cleanup(words)
		
		if len(words):
			patient.index = index
			patients.append(patient)
			index += 1

	return patients

def wordGraph(patients):
	for patient in patients:
		G = nx.Graph()

		if len(patient.words) == 1:
			G.add_node(patient.words[0])
		else:
			[G.add_edge(patient.words[i-1], patient.words[i]) 
				for i in range(1, len(patient.words))]

		patient.graph = G
		patient.features = graphFeatures(G)

	return patients

def graphFeatures(G):
	cycles =  nx.cycle_basis(G)
	longestCycle = 0

	if len(cycles):
		longestCycle = max([len(cycle) for cycle in cycles])

	if len(G.nodes()):
		features = {
			"edgeCount": len(G.edges()),
			"nodeCount": len(G.nodes()),
			"diameter": nx.diameter(G),
			"cycleCount": len(cycles),
			"longestCycle": longestCycle
		}
	else:
		features = {
			"edgeCount": 0,
			"nodeCount": 0,
			"diameter": 0,
			"cycleCount": 0,
			"longestCycle": 0
		}

	return features

def dunn_index(X, cluster_labels):
	D = euclidean_distances(X)
	labels = list(set(cluster_labels))
	inter_clust_dist = []
	intra_clust_dist = []

	for label1 in labels:
		idx1 = np.where(cluster_labels == label1)[0]
		for label2 in labels[label1+1:]:
			idx2 = np.where(cluster_labels == label2)[0]
			inter_clust_dist.append(np.min(D[idx1][:,idx2]))


		intra_clust_dist.append(np.max(D[idx1][:,idx1]))

	di = np.min(inter_clust_dist) / np.max(intra_clust_dist)
	return di

def sum_squared_error(X, centers, labels):
	sse = 0 
	ulabels = set(list(labels))

	for label in ulabels:
		c = centers[label]
		idx = np.where(labels == label)
		sse += sum([np.linalg.norm(c-x) for x in X[idx,:]])

	return sse

def reordered_dist_matrix(X, labels):
	index = np.empty(shape=(0,0))
	ulabels = list(set(labels))

	for label in ulabels:
		idx = np.where(labels == label)[0]
		index = np.append(index, idx)

	D = squareform(pdist(X, 'euclidean'))
	index = index.astype(int)
	reordered_d = D[index,:][:,index]
	plt.imshow(reordered_d, cmap='Greys_r')
	plt.colorbar()
	plt.savefig('results/clusters.png')
	plt.clf()

def cluster_centroids(X, patients, centers, labels):
	centroids = {}
	ulabels = list(set(labels))

	for label in labels:
		centroids[label] = []
		idx = np.where(labels == label)[0]
		clust_pats = [patients[i] for i in idx]
		center_patient_dist = euclidean_distances(centers[label].reshape(1,-1), X[idx,:])
		idx = np.argsort(center_patient_dist)[0]
		[centroids[label].append(clust_pats[i]) for i in idx[0:5]]

	return centroids

def cluster_words(patients, labels):
	ulabels = list(set(labels))
	clust = []

	for label in ulabels:
		idx = np.where(labels == label)[0]
		words = list(itertools.chain(*[set(patients[i].words) for i in idx]))
		c = Counter(words)

		for word, freq in c.iteritems():
			c[word] = round(freq/float(len(idx)), 2)
		
		clust.append(c)
		
	return clust

def cluster_ngrams(patients, labels):
	ulabels = list(set(labels))
	clust = []

	for label in ulabels:
		grams2 = []
		idx = np.where(labels == label)[0]
		for i in idx:
			for gram in ngrams(patients[i].words, 2):
				grams2.append(gram)

		c = Counter(grams2)
		for gram, freq in c.iteritems():
			c[gram] = round(freq/float(len(idx)), 2)

		clust.append(c)

	return clust

def cluster_word_importance(patients, labels):
	ulabels = list(set(labels))
	documents = []
	results =[]
	clust_size = []

	for label in ulabels:
		idx = np.where(labels == label)[0]
		clust_words = list(itertools.chain(*[patients[i].words for i in idx]))
		documents.append(" ".join(clust_words))
		clust_size.append(float(len(idx)))

	features, tfidf, tf = get_tfidf(documents)
	feature = np.array(features)

	for k in ulabels:
		idx = np.argsort(tfidf[k][::-1][0:30])
		results.append(np.array(features)[idx])
		results.append( (tf[k][idx] / clust_size[k]) * 100 )

	print np.stack(results, axis=1)
	np.savetxt("data/tfidf.csv", np.stack(results, axis=1), fmt="%s,%s,%s,%s,%s,%s,%s,%s")

def plot_dunn_index(di, n_clusters):
	plt.plot(di, marker='*')
	plt.xlabel("Number of Clusters")
	plt.ylabel("Dunn's Index")
	plt.xticks(range(0,len(n_clusters)), n_clusters)
	plt.savefig('data/dunn_index.jpg')
	plt.clf()

def plot_sse(sse, n_clusters):
	plt.plot(sse, marker='*')
	plt.xlabel("Number of Clusters")
	plt.ylabel("Sum of Squared Error")
	plt.xticks(range(0,len(n_clusters)), n_clusters)
	plt.savefig('data/sse.jpg')
	plt.clf()

def word_cleanup(words):
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
	lmtzr = WordNetLemmatizer()
	word = re.sub("\s*-\s*p|\s*\(p\)", "", word).lower() 
	lemmas = [lmtzr.lemmatize(spell(w)) for w in word.split() if re.match('\w{2,}\s*',w)] 
	hasSynsets = min([len(wn.synsets(w)) for w in lemmas] or [0])
 		
	if hasSynsets == 0:
		return " ".join(lemmas), False
	else:
		return " ".join(lemmas), True
