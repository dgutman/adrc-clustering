from gensim.models import word2vec
#import word2vec
import pickle
from utils import *
import itertools
from sklearn.cluster import KMeans

with open("data/WordFluency.pickle", "r") as fh:
	data = pickle.load(fh)

patients = getPatients(data)
sentences = [patient.words for patient in patients]

words = list(itertools.chain(*sentences))
print [word for word in words if "milk" in word]
print len(words)
model = word2vec.Word2Vec(sentences, size=18180, min_count=0)
X = model.syn0

with open("data/words.csv", "w") as fh:
	for patient in patients:
		for word in patient.words:
			word = re.sub("\s*-\s*p|\s*\(p\)|\s*-\s*i", "", word).lower()
			fh.write(word + "\n")