#!/home/mkhali8/anaconda2/envs/adrc-clustering/bin/python -W ignore

import os
import sys
import argparse
import numpy as np
import time
from sklearn.cluster import KMeans
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from libs.features import *
from libs.clustering import *
from config import *

# Define functions
def compute_sse(X): 
	""" compute_sse()
	Compute the sum of squared error for n clusters
	"""
	sse = []
	n_clusters = range(2,15)

	for n in n_clusters:
		km = KMeans(n_clusters=n)
		clustering = km.fit(X)
		sse.append(sum_squared_error(X, clustering.cluster_centers_, km.labels_))

	plot_sse(sse, n_clusters, os.path.join(results_dir, 'sse.png'))

def compute_di(X): 
	""" compute_di()
	Compute dunn index for n clusters
	"""
	di = []
	n_clusters = range(2,15)

	for n in n_clusters:
		km = KMeans(n_clusters=n)
		clustering = km.fit(X)
		di.append(sum_squared_error(X, clustering.cluster_centers_, km.labels_))

	plot_dunn_index(di, n_clusters, os.path.join(results_dir, 'dunn_index.png'))

def save_cluster_centroids(clust_centroids):
	"""save_cluster_centroids()
	Save cluster centroid (patients graphs) to file
	"""
	for label, centroids in clust_centroids.iteritems():
		for centroid in centroids:
			nx.draw_spectral(centroid.graph)
			output = os.path.join(results_dir, 'cluster_%d_%d.png' % (label, centroid.index))
			plt.savefig(output)
			plt.clf() 

def save_cluster_centers(features, centers):
	"""save_cluster_centers()
	Save cluster centers to CSV file
	"""
	with open(os.path.join(results_dir, "cluster_centers.csv"), "w") as fh:
		w = csv.writer(fh)
		w.writerow(features)

		for center in centers:
			w.writerow(center)

if __name__ == "__main__":
	# Parse command line arguments
	default_features = ["node_count", "edge_count", "cycle_count", "error_count", "diameter", "longest_cycle"]

	parser = argparse.ArgumentParser()
	parser.add_argument("-f", "--features", nargs='+', default=default_features, type=str, help="list of features")
	parser.add_argument("-k", "--n_clusters", default=4, type=int, help="number of clusters")
	parser.add_argument("-i", "--input", default=INPUT_FILE, help="Input file (full path)")
	parser.add_argument("-o", "--output", default=OUTPUT_DIR, help="Output directory")
	parser.add_argument("-sse", "--squared_error", help="Compute sum of squared error", action="store_true")
	parser.add_argument("-di", "--dunn_index", help="Compute dunn index", action="store_true")
	args = parser.parse_args()

	# Check if the results dir exists or create one
	print "Creating output directory..."
	print time.strftime("%Y-%m-%d-%H-%M-%D")
	results_dir = os.path.join(args.output, "k"+str(args.n_clusters), time.strftime("%Y-%m-%d-%H-%M-%S"))
	if not os.path.exists(results_dir):
		os.makedirs(results_dir)
	print "Output directory: %s" % results_dir

	# Read patient data from CSV file and convert into Dict
	# Aggregate verbal fluency tests and compute patient 
	# graph and its features
	print "Reading data file..."
	patients = read_data_from_file(args.input)

	print "Generate graph and graph features..."
	patients = aggregate_fluency_tests(patients)
	patients = word_graph(patients)
	patients, features, X = build_feature_matrix(patients, args.features)
	
	if args.squared_error:
		compute_sse(X)
	if args.dunn_index:
		compute_di(X)

	# Run k-means for some k clusters
	print "Run k-means for %d with k=%d..." % (len(patients), args.n_clusters)
	km = KMeans(n_clusters=args.n_clusters)
	clustering = km.fit(X)
	centers = clustering.cluster_centers_
	labels = list(set(km.labels_))

	# Given the matrix A and cluster labels
	# Compute distance between centers and patients
	# Reorder the matrix, then save the image
	print "Cluster summurization..."
	reordered_dist_matrix(X, km.labels_, os.path.join(results_dir, 'reordered_sim_matrix.png'))
	clust_centroids = cluster_centroids(X, patients, centers, km.labels_)
	cluster_word_importance(patients, km.labels_, os.path.join(results_dir, 'tfidf.csv'))
	save_cluster_centroids(clust_centroids)
	save_cluster_centers(features, centers)

	print "Done"
