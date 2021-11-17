import gensim.corpora as corpora
from gensim import models
from gensim.test.utils import datapath
# from gensim.models import CoherenceModel

import ast
import pandas as pd

data = pd.read_csv('data\clean_dataset.csv')

# Convert the string representations of the lists in content back into a list type
data.content = data.content.apply(lambda x: ast.literal_eval(x))

# Create a dictionary with the number of times a word appears in the dataset
counts_to_words = corpora.Dictionary(data.content)

# Filtering: remove very rare and very common words
counts_to_words.filter_extremes(no_below=5, no_above=0.5, keep_n=100000)

# Bag-of-words (BOW)
# Convert each document into the Bag-of-words format
bow_corpus = [counts_to_words.doc2bow(text) for text in data.content]

# TF-IDF
tfidf = models.TfidfModel(bow_corpus)

# Transform the corpus
corpus_tfidf = map(lambda x: tfidf[x], bow_corpus)

# Building the LDA model
# lda = gensim.models.LdaMulticore(bow_corpus, id2word=dictionary, num_topics=10, passes=50)
#
# temp_file = datapath("model")
# lda_model = lda.save(temp_file)
