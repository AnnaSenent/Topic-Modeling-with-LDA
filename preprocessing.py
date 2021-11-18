from utils import preprocess, lemmatize

data = pd.read_csv('data\dataset.csv')

print(data.target_names.unique())

t = time()

# Preprocess and tokenize
data.content = data.content.apply(lambda x: preprocess(x))

# Lemmatize
data.content = data.content.apply(lambda x: lemmatize(' '.join(x), ['NOUN', 'ADJ', 'VERB', 'ADV']))

print('It took: {} mins'.format(round((time() - t) / 60, 2)))

data.to_csv('data\clean_dataset.csv', index=False)

