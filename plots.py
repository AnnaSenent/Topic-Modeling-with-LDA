import gensim
import gensim.corpora as corpora
from gensim import models

import ast
import pandas as pd

import pyLDAvis
import pyLDAvis.gensim_models
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

data = pd.read_csv('data\clean_dataset.csv')

# Convert the string representation of the lists in "content" back into a list type
data.content = data.content.apply(lambda x: ast.literal_eval(x))

# Create a dictionary with the number of times a word appears in the dataset
counts_to_words = corpora.Dictionary(data.content)

# Filtering: remove very rare and very common words
counts_to_words.filter_extremes(no_below=5, no_above=0.5, keep_n=100000)

# Bag-of-words (BOW)
# Convert each document into the Bag-of-words format
bow_corpus = [counts_to_words.doc2bow(text) for text in data.content]

# TF-IDF
tfidf = models.TfidfModel(corpus=bow_corpus, id2word=counts_to_words, normalize=False)

# Transform the corpus
corpus_tfidf = list(map(lambda x: tfidf[x], bow_corpus))

# Load models
bow_model = gensim.models.LdaMulticore.load(r'models\bow_model.model')
tfidf_model = gensim.models.LdaMulticore.load(r'models\tfidf_model.model')

vis1 = pyLDAvis.gensim_models.prepare(bow_model, bow_corpus, counts_to_words)
pyLDAvis.save_html(vis1, r'docs\bow.html')

vis2 = pyLDAvis.gensim_models.prepare(tfidf_model, corpus_tfidf, counts_to_words)
pyLDAvis.save_html(vis2, r'docs\tfidf.html')