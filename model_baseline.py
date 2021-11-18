import gensim
import gensim.corpora as corpora
from gensim import models
from utils import lda_model

import ast
import pandas as pd

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

# Building the LDA model
if __name__ == '__main__':

    # Bow
    bow_model, bow_model_perplexity, bow_model_coherence = lda_model(data.content, bow_corpus, counts_to_words, 20)
    bow_model.save(r'models\bow_model.model')

    print('\nPerplexity (Bow model): ', bow_model_perplexity)
    print('\nCoherence Score: ', bow_model_coherence)

    for idx, topic in bow_model.print_topics(-1):
        print("Topic: {} \nWords: {}".format(topic, idx ))
        print("\n")

    # Tfidf
    tfidf_model, tfidf_model_perplexity, tfidf_model_coherence = lda_model(data.content, corpus_tfidf, counts_to_words, 20)
    tfidf_model.save(r'models\tfidf_model.model')

    print('\nPerplexity (Tfidf model): ', tfidf_model_perplexity)
    print('\nCoherence Score: ', tfidf_model_coherence)

    for idx, topic in tfidf_model.print_topics(-1):
        print("Topic: {} \nWords: {}".format(topic, idx ))
        print("\n")

