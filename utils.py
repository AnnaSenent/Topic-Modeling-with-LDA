import re
import numpy as np
import pandas as pd

import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from gensim import models
from gensim.test.utils import datapath
from gensim.models import CoherenceModel

import spacy
from time import time

stopwords = STOPWORDS.union({'from', 'subject', 're', 'edu', 'use'})

def preprocess(text):

    prep_text = str(text).lower()

    # Remove emails
    prep_text = re.sub('\S*@\S*\s?', '', prep_text)

    # Replace multiple spaces with a single space
    prep_text = re.sub('\s+', ' ', prep_text)

    # Remove single quotes
    prep_text = re.sub("\'", '', prep_text)

    # Tokenize and remove stopwords

    prep_text = [token for token in gensim.utils.simple_preprocess(prep_text, deacc=True) if token not in stopwords]

    return prep_text

def lemmatize(text, pos_tags):

    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

    lemmatized_text = [token.lemma_ for token in nlp(text) if token.pos_ in pos_tags]

    return lemmatized_text

def lda_model(data, corpus, dictionary, topics):

    # Build the model
    model = gensim.models.LdaMulticore(corpus=corpus,
                                       id2word=dictionary,
                                       num_topics=topics,
                                       random_state=100,
                                       workers=3,
                                       chunksize=100,
                                       passes=50,
                                       # alpha='auto',
                                       per_word_topics=True)

    # Compute perplexity
    perplexity = model.log_perplexity(corpus)

    # Compute coherence score
    coherence_model = CoherenceModel(model=model, texts=data, dictionary=dictionary, coherence='c_v')
    coherence_score = coherence_model.get_coherence()

    return model, perplexity, coherence_score