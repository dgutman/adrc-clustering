import itertools
import math
import numpy as np
from collections import Counter

def get_features_names(documents):
	""" get_features_names()
	Given the clusters (documents) return the set of all 
	unique words appearing in all clusters 
	"""
	features = list(set(itertools.chain(*[document.split() for document in documents])))
	return features

def get_idf(features, documents):
	""" get_idf()
    Compute the inverse document frequency for the set features
    across all documents (clusters)
    """
	idf = []
	k = len(documents)

	for feature in features:
		x = sum([1 for document in documents if feature in document])
		idf.append( math.log( k / float(x) ) )

	return np.array(idf)

def get_tf(features, documents):
	""" get_tf()
	Compute the term frequency for the set features
	across all documents (clusters)
	"""
	tf = []

	for document in documents:
		c = Counter(document.split())
		tf.append([c[feature] for feature in features])
	
	return np.array(tf)

def get_tfidf(documents):
	""" get_tfidf()
	Documents in this context refers to clusters. Imagine each cluster 
	is a document containing the animal names mentioned by the patients
	assigned to this cluster

	First, we get all feature names across all clusters, that means get 
	all the unique words (dog, cat, tiger, etc.) appearing in the clusters

	Second, now we have the features, compute the inverse document frequency (IDF)
	and the term frequency (TF)

	Third, compute the TF-IDF 
	"""
	features = get_features_names(documents)
	idf = get_idf(features, documents)
	tf = get_tf(features, documents)

	tfidf = []
	for k in range(0, len(documents)):
		tfidf.append(idf * tf[k])

	return features, tfidf, tf
