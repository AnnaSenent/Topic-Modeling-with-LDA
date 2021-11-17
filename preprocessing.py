from utils import *

data = pd.read_csv('data\dataset.csv')

print(data.target_names.unique())

t = time()

# Preprocess and tokenize
data.content.map(preprocess)

# Lemmatize
data.content.map(lemmatize)

print('It took: {} mins'.format(round((time() - t) / 60, 2))) # It took: 165.31 mins

data.to_csv('data\clean_dataset.csv', index=False)

data = pd.read_csv('data\clean_dataset.csv')

