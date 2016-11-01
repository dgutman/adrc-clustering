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
clust_centroids = cluster_centroids(A, patients, centers, km.labels_)

print centers
for label, centroids in clust_centroids.iteritems():
	for centroid in centroids:
		nx.draw_spectral(centroid.graph)
		plt.savefig('data/cluster_%d_%d.jpg' % (label, centroid.index))
		plt.clf() 


ID = 397
tmp = patients[ID].words
labels = {}
for word in tmp:
	labels[word] = word

pos = nx.spectral_layout(patients[ID].graph)
nx.draw_networkx_nodes(patients[ID].graph,pos,node_size=2000)
nx.draw_networkx_labels(patients[ID].graph,pos,labels,font_size=16)

nx.draw_spectral(patients[ID].graph)
#print nx.average_degree_connectivity(patients[ID].graph)
#plt.show()
