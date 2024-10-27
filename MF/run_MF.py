import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from BiasSVD import BiasSVD
from FunkSVD import FunkSVD
from PMF import PMF


df = pd.read_csv('../data/products_reviews.csv', header=0, usecols=['userId', 'productId', 'score'])

np.random.seed(42)
train = df.sample(frac=0.7, random_state=7)
test = df.drop(train.index.tolist())


funk_svd = FunkSVD(lr=0.05, reg=0.005, n_epochs=50, n_factors=20, min_rating=0, max_rating=5)
funk_svd.fit(X=train)
funk_pred = funk_svd.predict(test)
funk_mse = mean_absolute_error(test['score'], funk_pred)
print(f'mse of FunkSVD: {funk_mse:.2f}')


bias_svd = BiasSVD(lr=0.05, reg=0.005, n_epochs=50, n_factors=20, min_rating=0, max_rating=5)
bias_svd.fit(X=train)
bias_pred = bias_svd.predict(test)
bias_mse = mean_absolute_error(test['score'], bias_pred)
print(f'mse of BiasSVD: {bias_mse:.2f}')


pmf_svd = PMF(n_factors=20, lr=0.05, reg=0.1, momentum=0.8, n_epochs=50, min_rating=0, max_rating=5)
pmf_svd.fit(train)
pmf_pred = pmf_svd.predict(test)
pmf_mse = mean_absolute_error(test['score'], pmf_pred)
print(f'mse of PMF: {pmf_mse:.2f}')
