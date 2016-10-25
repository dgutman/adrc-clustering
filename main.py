#!/usr/bin/python
import pickle
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
import itertools
from utils import *

with open("data/WordFluency.pickle", "r") as fh:
	data = pickle.load(fh)

patients = getPatients(data)
patients = wordGraph(patients)

n = len(patients)
m = len(patients[0].features)
A = np.zeros(shape=(n,m))

for patient in patients:
	A[patient.index] = patient.features.values()

sse = []
di = []
n_clusters = range(2,15)

for k in n_clusters:
	km = KMeans(n_clusters=k)
	clustering = km.fit(A)
	centers = clustering.cluster_centers_
	labels = list(set(km.labels_))
	index_order = np.empty(shape=(0,0))

	di.append(dunn_index(A, km.labels_))
	sse.append(sum_squared_error(A, centers, km.labels_))

plot_dunn_index(di, n_clusters)
plot_sse(sse, n_clusters)

km = KMeans(n_clusters=4)
clustering = km.fit(A)
centers = clustering.cluster_centers_
labels = list(set(km.labels_))
reordered_dist_matrix(A, km.labels_)
cluster_common_words(patients, km.labels_)

for label in labels:
	c = centers[label]
	print c
	idx = np.where(km.labels_ == label)
	print len(idx[0])
	clust_pats = [patients[i] for i in idx[0]]
	center_point_dist = euclidean_distances([c], A[idx[0],:])
	idx = np.argsort(center_point_dist)[0]
	
	tmp = [p.words for p in clust_pats]
	unique_words = set(list(itertools.chain(*tmp)))
	print len(unique_words)

	for i in idx[0:5]:
		nx.draw_spectral(clust_pats[i].graph)
		plt.savefig('data/cluster_%d_%d.jpg' % (label, i))
		plt.clf()

"""
G = nx.Graph()
for i in range(0, n):
	for j in range(i+1, n):
		G.add_edge(i,j)

nx.draw_spectral(G)
plt.show()
"""

ID = 930
tmp = patients[ID].words
labels = {}
for word in tmp:
	labels[word] = word

pos = nx.spectral_layout(patients[ID].graph)
nx.draw_networkx_nodes(patients[ID].graph,pos,node_size=2000)
nx.draw_networkx_labels(patients[ID].graph,pos,labels,font_size=16)

nx.draw_spectral(patients[ID].graph)
#plt.show()
