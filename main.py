#!/usr/bin/python
import pickle
import numpy as np
from sklearn.cluster import KMeans
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
import itertools
from utils import *

with open("data/WordFluency.pickle", "r") as fh:
	data = pickle.load(fh)

patients = getPatients(data)
patients = wordGraph(patients)

n = len(patients)
m = len(patients[0].features) + 1
A = np.zeros(shape=(n,m))
features = patients[0].features.keys() + ["errors"]

for patient in patients:
	A[patient.index] = patient.features.values() + [len(patient.errors)]

"""
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
"""

km = KMeans(n_clusters=4)
clustering = km.fit(A)
centers = clustering.cluster_centers_
labels = list(set(km.labels_))
reordered_dist_matrix(A, km.labels_)

clust_centroids = cluster_centroids(A, patients, centers, km.labels_)
cluster_word_importance(patients, km.labels_)
#clust_words = cluster_words(patients, km.labels_)
#clust_grams = cluster_ngrams(patients, km.labels_)

for label, centroids in clust_centroids.iteritems():
	for centroid in centroids:
		nx.draw_spectral(centroid.graph)
		plt.savefig('results/cluster_%d_%d.png' % (label, centroid.index))
		plt.clf() 

with open("results/cluster_centers.csv", "w") as fh:
	w = csv.writer(fh)
	w.writerow(features)

	for center in centers:
		w.writerow(center)
