import sklearn.datasets
import pandas as pd


dataset = sklearn.datasets.load_files(r'data\20news-bydate-train')

target_names = [dataset.target_names[target_i] for target_i in dataset.target]

dataset = pd.DataFrame({'content': dataset.data, 'target': dataset.target, 'target_names': target_names})

dataset.to_csv('data\dataset.csv', index=False)