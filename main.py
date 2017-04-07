#!/home/mkhali8/anaconda2/envs/adrc-clustering/bin/python

import numpy as np
from sklearn.cluster import KMeans
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from features import *
from clustering import *

# Read patient data from CSV file and convert into Dict
# Aggregate verbal fluency tests and compute patient 
# graph and its features
input_file = "/home/mkhali8/dev/adrc-clustering/data/WordFluencyMultiTest.csv"
patients = read_data_from_file(input_file)
patients = aggregate_fluency_tests(patients)
patients = word_graph(patients)

# Generate feature vector matrix using graph features
n = len(patients)
m = len(patients[0].features) + 1
A = np.zeros(shape=(n,m))
features = patients[0].features.keys() + ["errors"]

for patient in patients:
	A[patient.index] = patient.features.values() + [len(patient.errors)]

# Compute the sum of squared error (SSE)
# and the dunn index (DI) for meausre cluster quality
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

# Run k-means for some k clusters
km = KMeans(n_clusters=4)
clustering = km.fit(A)
centers = clustering.cluster_centers_
labels = list(set(km.labels_))

# Given the matrix A and cluster labels
# Compute distance between centers and patients
# Reorder the matrix, then save the image
reordered_dist_matrix(A, km.labels_)

# Save cluster centers and TF-IDF values for words in each cluster
clust_centroids = cluster_centroids(A, patients, centers, km.labels_)
cluster_word_importance(patients, km.labels_)

# Find representitive patients for each cluster and save
# their verbal fluency graph
for label, centroids in clust_centroids.iteritems():
	for centroid in centroids:
		nx.draw_spectral(centroid.graph)
		plt.savefig('results/cluster_%d_%d.png' % (label, centroid.index))
		plt.clf() 

# Write cluster center to file
with open("results/cluster_centers.csv", "w") as fh:
	w = csv.writer(fh)
	w.writerow(features)

	for center in centers:
		w.writerow(center)
