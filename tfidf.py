import itertools
import math
import numpy as np
from collections import Counter

def get_features_names(documents):
	features = list(set(itertools.chain(*[document.split() for document in documents])))
	return features

def get_idf(features, documents):
	idf = []
	k = len(documents)

	for feature in features:
		x = sum([1 for document in documents if feature in document])
		idf.append( math.log( k / float(x) ) )

	return np.array(idf)

def get_tf(features, documents):
	tf = []

	for document in documents:
		c = Counter(document.split())
		tf.append([c[feature] for feature in features])
	
	return np.array(tf)

def get_tfidf(documents):
	features = get_features_names(documents)
	idf = get_idf(features, documents)
	tf = get_tf(features, documents)

	tfidf = []
	for k in range(0, len(documents)):
		tfidf.append(idf * tf[k])

	return features, tfidf, tf
