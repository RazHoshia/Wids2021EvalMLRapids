# -*- coding: utf-8 -*-
# TODO add documentation, add README, add licence
# TODO add hyperparameter_ranges
import numpy as np
from evalml.automl import AutoMLSearch
from evalml.pipelines import BinaryClassificationPipeline
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from skopt.space import Integer, Real

from estimators.classifier.decision_tree import RapidsRf
from estimators.classifier.k_neighbors import RapidsKNeighborsClassifier
from estimators.classifier.linear_model import RapidsLogisticRegression


class WidsPipeline(BinaryClassificationPipeline):
    component_graph = {
        'Imputer': ['Imputer'],
        'Random KNeighborsClassifier': [RapidsKNeighborsClassifier, 'Imputer'],
        'Linear Regression': [RapidsLogisticRegression, 'Imputer'],
        'Final RF Estimator': [RapidsRf, 'Random KNeighborsClassifier', 'Linear Regression']
    }

    custom_hyperparameters = {
        'Imputer': {
            'numeric_impute_strategy': ['median', 'most_frequent', 'mean']
        },
        'Random KNeighborsClassifier': {
            "n_neighbors": Integer(3, 1000),
        },
        'Linear Regression': {"penalty": ['none'],
                              "tol": Real(1e-4, 1e-3),
                              "C": Real(1.0, 2.0),
                              "fit_intercept": [True, False],
                              "max_iter": Integer(50, 100),
                              "l1_ratio": Real(0.1, 1.0)},
        'Final RF Estimator': {
            "n_estimators": Integer(10, 1000),
            "max_depth": Integer(1, 100),
            "n_bins": Integer(1, 16),
        },
    }


def err_call(exception, traceback, automl, **kwargs):
    raise exception


iris = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(iris.data.astype(np.float64),
                                                    iris.target.astype(np.float64), train_size=0.75, test_size=0.25,
                                                    random_state=42)
# print(WidsPipeline(parameters={}).graph())

automl = AutoMLSearch(X_train=X_train, y_train=y_train,
                      problem_type='binary', objective='auc',
                      allowed_pipelines=[WidsPipeline],
                      error_callback=err_call,
                      max_iterations=3, max_batches=3)

automl.search()

print(automl.best_pipeline.parameters)
