from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial.distance import pdist, squareform
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter
from nltk import ngrams
from tfidf import *

def dunn_index(X, cluster_labels):
	""" dunn_index()
	Compute dunn index, which is the ratio of the inter cluster
	distance to the intra cluster distance. The lower the ratio
	the more seprarable the clusters are.
	"""
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

def sum_squared_error(X, centers, cluster_labels):
	""" sum_squared_error()
	Compute the sum of sequared error. This is the sum of distances
	between all data points in a cluster and the corresponding cluster
	center.
	"""
	sse = 0
	labels = set(list(cluster_labels))

	for label in labels:
		c = centers[label]
		idx = np.where(cluster_labels == label)
		sse += sum([np.linalg.norm(c-x) for x in X[idx,:]])

	return sse

def reordered_dist_matrix(X, cluster_labels, output):
	""" reordered_dist_matrix()
	Given the feature vector matrix X and the cluster labels, compute 
	the distance between all data points then reorder the distance matrix
	based on the cluster labels, such that data points with same cluster label
	are adjacent.
	"""
	index = np.empty(shape=(0,0))
	labels = list(set(cluster_labels))

	for label in labels:
		idx = np.where(cluster_labels == label)[0]
		index = np.append(index, idx)

	D = squareform(pdist(X, 'euclidean'))
	index = index.astype(int)
	reordered_d = D[index,:][:,index]
	plt.imshow(reordered_d, cmap='Greys_r')
	plt.colorbar()
	plt.savefig(output)
	plt.clf()

def cluster_centroids(X, patients, centers, cluster_labels):
	""" cluster_centroids()
	Given feature vector matrix X, patients, cluster centers and label
	find the patients most representitive of each cluster, the centroid.
	A centroid is the patient closest to the cluster center.
	"""
	centroids = {}
	labels = list(set(cluster_labels))

	for label in cluster_labels:
		centroids[label] = []
		idx = np.where(cluster_labels == label)[0]
		clust_pats = [patients[i] for i in idx]
		center_patient_dist = euclidean_distances(centers[label].reshape(1,-1), X[idx,:])
		idx = np.argsort(center_patient_dist)[0]
		[centroids[label].append(clust_pats[i]) for i in idx[0:5]]

	return centroids

def cluster_words(patients, cluster_labels):
	""" cluster_words()
	Given patients and their corresponding cluster labels compute
	the frequency of each word in that cluster and return a dictionary 
	of WORD:FREQUENCY
	"""
	labels = list(set(cluster_labels))
	clust = []

	for label in labels:
		idx = np.where(cluster_labels == label)[0]
		words = list(itertools.chain(*[set(patients[i].words) for i in idx]))
		c = Counter(words)

		for word, freq in c.iteritems():
			c[word] = round(freq/float(len(idx)), 2)

		clust.append(c)

	return clust

def cluster_ngrams(patients, cluster_labels):
	""" cluster_ngrams()
	Given patients and their corresponding cluster labels compute 
	ngrams (n=2). This function is not used anywhere for now
	"""
	labels = list(set(cluster_labels))
	clust = []

	for label in labels:
		grams2 = []
		idx = np.where(cluster_labels == label)[0]
		for i in idx:
			for gram in ngrams(patients[i].words, 2):
				grams2.append(gram)

		c = Counter(grams2)
		for gram, freq in c.iteritems():
			c[gram] = round(freq/float(len(idx)), 2)

		clust.append(c)

	return clust

def cluster_word_importance(patients, cluster_labels, output):
	""" cluster_word_importance()
	Given patients and their corresponding cluster labels let us compute
	the TF-IDF for each word in each cluster. See tfidf.py for more details.
	The results are saved to CSV file.
	"""
	labels = list(set(cluster_labels))
	documents = []
	results =[]
	clust_size = []

	for label in labels:
		idx = np.where(cluster_labels == label)[0]
		clust_words = list(itertools.chain(*[patients[i].words for i in idx]))
		documents.append(" ".join(clust_words))
		clust_size.append(float(len(idx)))

	features, tfidf, tf = get_tfidf(documents)
	feature = np.array(features)

	for k in labels:
		idx = np.argsort(tfidf[k][::-1][0:30])
		results.append(np.array(features)[idx])
		results.append( (tf[k][idx] / clust_size[k]) * 100 )

	np.savetxt(output, np.stack(results, axis=1), fmt="%s"*len(labels)*2)

def plot_dunn_index(di, n_clusters, output):
	""" plot_dunn_index()
	Given dunn index for n clusters generate a graph
	and save it
	"""
	plt.plot(di, marker='*')
	plt.xlabel("Number of Clusters")
	plt.ylabel("Dunn's Index")
	plt.xticks(range(0,len(n_clusters)), n_clusters)
	plt.savefig(output)
	plt.clf()

def plot_sse(sse, n_clusters, output):
	""" plot_sse()
	Given sum of squared error for n clusters generate a graph
	and save it
	"""
	plt.plot(sse, marker='*')
	plt.xlabel("Number of Clusters")
	plt.ylabel("Sum of Squared Error")
	plt.xticks(range(0,len(n_clusters)), n_clusters)
	plt.savefig(output)
	plt.clf()
