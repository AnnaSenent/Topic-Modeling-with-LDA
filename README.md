# Topic-Modeling-with-LDA

In this project I will use Latent Dirichlet Allocation (LDA) to extract hidden topics from a corpus.

## Introduction

LDA is a popular algorithm used for Topic Modeling. It builds a topic per document and a set of words per topic, modeled as Dirichlet distributions. Documents are modeled as multinomial distributions of topics and topics are modeled as multinomial distributions of words.

LDA is based on two assumptions:

- Documents are produced from a mixture of topics
- Topics are a mixture of words

These topics generate words based on their probability distribution. LDA will also assume that the every chunk of text it is fed will contain words that are related at some level. We can therefore estimate which topics would have generated a given document and which words would have generated a given topic, since the goal is to obtain the most optimized document-topic distribution and topic-word distribution.

## Libraries Used

- Gensim
- SpaCy
- pyLDAvis

### Dataset

I used the [20 newsgroups](http://qwone.com/~jason/20Newsgroups/) dataset. In the script () I prepare the dataset for later use.

### Preprocessing

Using the functions preprocess and lemmatize from the script (), I cleaned the text and created a new csv file in ()

### Building the LDA models

### Perplexity and coherence scores

The model with the tf-idf vectors achieved a slightly better performance than the one with the Bag-of-words format.

### Visualizing the topics and top words per topic

### Resources
