# -*- coding: utf-8 -*-
import cudf
from cuml.neighbors import KNeighborsClassifier as cuKNeighborsClassifier
from cuml.svm import SVC as cuSVC
from evalml.model_family import ModelFamily

from estimators.rapids_base_estimator import RapidsClassifier


class RapidsSVC(RapidsClassifier):
    name = "Rapids SVC"
    model_family = ModelFamily.K_NEIGHBORS

    def __init__(self, random_seed=0, C=1.0, kernel='rbf', degree=3, gamma='scale', tol=1e-3, nochange_steps=1000,
                 **kwargs):
        parameters = {"C": C,
                      "kernel": kernel,
                      "degree": degree,
                      "gamma": gamma,
                      "tol": tol,
                      "nochange_steps": nochange_steps}
        parameters.update(kwargs)
        super().__init__(parameters=parameters,
                         component_obj=cuSVC(probability=True, **parameters),
                         random_seed=random_seed
                         )

    def fit(self, X, y=None):
        # see https://github.com/rapidsai/cuml/issues/3090
        X_cudf_train = X.to_dataframe().astype('float32')
        y_cudf_train = cudf.Series(y.to_series())
        self._component_obj.fit(X_cudf_train.to_numpy(), y_cudf_train)
        return self


class RapidsKNeighborsClassifier(RapidsClassifier):
    name = "Rapids KNeighbors Classifier"
    model_family = ModelFamily.K_NEIGHBORS

    def __init__(self, random_seed=0, n_neighbors=5,
                 **kwargs):
        parameters = {"n_neighbors": n_neighbors}
        parameters.update(kwargs)
        super().__init__(parameters=parameters,
                         component_obj=cuKNeighborsClassifier(**parameters),
                         random_seed=random_seed
                         )
