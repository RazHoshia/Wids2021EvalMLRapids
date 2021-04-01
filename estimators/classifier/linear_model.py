# -*- coding: utf-8 -*-
from cuml.linear_model import LogisticRegression as cuLogisticRegression
from evalml.model_family import ModelFamily

from estimators.rapids_base_estimator import RapidsClassifier


class RapidsLogisticRegression(RapidsClassifier):
    name = "Rapids Logistic Regression"
    model_family = ModelFamily.LINEAR_MODEL

    def __init__(self, random_seed=0, penalty='none', tol=1e-4, C=1.0, fit_intercept=True, max_iter=100, l1_ratio=0.5,
                 **kwargs):
        parameters = {"penalty": penalty,
                      "tol": tol,
                      "C": C,
                      "fit_intercept": fit_intercept,
                      "max_iter": max_iter,
                      "l1_ratio": l1_ratio}
        parameters.update(kwargs)
        super().__init__(parameters=parameters,
                         component_obj=cuLogisticRegression(**parameters),
                         random_seed=random_seed)
