from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from pyspark import SparkContext

# from sklearn.grid_search import GridSearchCV
# Use spark_sklearn’s grid search instead:
from spark_sklearn import GridSearchCV

class timeit():
    from datetime import datetime
    def __enter__(self):
        self.tic = self.datetime.now()
    def __exit__(self, *args, **kwargs):
        print('runtime: {}'.format(self.datetime.now() - self.tic))

sc = SparkContext.getOrCreate()

digits = datasets.load_digits()
X, y = digits.data, digits.target
param_grid = {"max_depth": [3, None],
              "max_features": [1, 3, 10],
              "min_samples_split": [2, 5, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"],
              "n_estimators": [10, 20, 40, 80]}

gs = GridSearchCV(sc=sc, estimator=RandomForestClassifier(), param_grid=param_grid)

with timeit():
    gs.fit(X, y)

print(gs)