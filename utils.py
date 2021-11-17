import re
import numpy as np
import pandas as pd

import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from gensim.parsing.preprocessing import STOPWORDS

import spacy
from time import time

stopwords = STOPWORDS.union({'from', 'subject', 're', 'edu', 'use'})

def preprocess(text):

    # Remove emails
    prep_text = re.sub('\S*@\S*\s?', '', str(text))

    # Replace multiple spaces with a single space
    prep_text = re.sub('\s+', ' ', prep_text)

    # Remove single quotes
    prep_text = re.sub("\'", '', prep_text)

    # Tokenize and remove stopwords

    prep_text = [token for token in gensim.utils.simple_preprocess(prep_text, deacc=True) if token not in stopwords]

    return prep_text

def lemmatize(text, pos_tags=['NOUN', 'ADJ', 'VERB', 'ADV']):

    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

    lemmatized_text = [token.lemma_ for token in nlp(text) if token.pos_ in pos_tags]

    return lemmatized_text