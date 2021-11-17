from utils import *

data = pd.read_csv('data\clean_dataset.csv')

print(data.target_names.unique())

# Preprocess and tokenize

# Create bigrams and trigrams

# Lemmatize
# data.content.apply(lambda x: ' '.join(lemmatize(x))
