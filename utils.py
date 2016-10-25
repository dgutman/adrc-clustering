import re
import itertools
import networkx as nx
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial.distance import pdist, squareform
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

def getPatients(data):
	patients = []
	p = re.compile("^avf_anm\d+$")
	index = 0

	for i in range(0, len(data)):
		patient = lambda: None
		words = [re.sub("\s*\-\s*p|\s*\(p\)","", val).lower().split(",") 
				 for key, val in data[i].iteritems() 
				 if p.match(key) and val != ""]

		patient.words = list(itertools.chain(*words))
		patient.index = index

		if len(words):
			index += 1
			patients.append(patient)

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

	features = {
		"edgeCount": len(G.edges()),
		"nodeCount": len(G.nodes()),
		"diameter": nx.diameter(G),
		"cycleCount": len(cycles),
		"longestCycle": longestCycle
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
	plt.savefig('data/clusters.jpg')
	plt.clf()

def cluster_common_words(patients, labels, k=10):
	ulabels = list(set(labels))

	with open("data/cluster_common_words.csv", "w") as fh:
		for label in ulabels:
			idx = np.where(labels == label)[0]
			words = list(itertools.chain(*[patients[i].words for i in idx]))
			words = Counter(words)
			
			fh.write("Cluster %d\n" % label)
			fh.write('\n'.join('%s,%d' % x for x in words.most_common(k)) + '\n\n')



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
