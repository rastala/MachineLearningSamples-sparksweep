# Use the Azure Machine Learning data collector to log accuracy
from azureml.logging import get_azureml_logger
logger = get_azureml_logger()

from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from pyspark import SparkContext

# Use spark_sklearn’s grid search:
from spark_sklearn import GridSearchCV
import pandas as pd

class timeit():
    from datetime import datetime
    def __enter__(self):
        self.tic = self.datetime.now()
    def __exit__(self, *args, **kwargs):
        print('runtime: {}'.format(self.datetime.now() - self.tic))

sc = SparkContext.getOrCreate()

digits = datasets.load_digits()
x, y = digits.data, digits.target

# Create hold-out test dataset
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.25)

param_grid = {"max_depth": [3, None],
              "max_features": [1, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"],
              "n_estimators": [10, 20, 40, 80]}

gs = GridSearchCV(sc=sc, estimator=RandomForestClassifier(), cv=4,param_grid=param_grid,refit=True)

with timeit():
    gs.fit(x_train, y_train)

results = pd.DataFrame(gs.cv_results_)
print(results.sort_values(['mean_test_score'],ascending=False)[0:10])

# Validate accuracy of best model against hold-out data
best_model = gs.best_estimator_
test_accuracy = best_model.score(x_test,y_test)
print(test_accuracy)

logger.log('Best model accuracy',test_accuracy)
