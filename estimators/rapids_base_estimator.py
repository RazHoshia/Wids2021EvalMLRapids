# -*- coding: utf-8 -*-

import cudf
from evalml.pipelines.components.estimators import Estimator
from evalml.utils.woodwork_utils import infer_feature_types
from evalml.problem_types import ProblemTypes


class RapidsEstimator(Estimator):
    """
    Base estimator for rapids. Mostly wraps the regular fit and predict with cudf conversion.
    """
    hyperparameter_ranges = {}
    supported_problem_types = []

    def fit(self, X, y=None):
        X_cudf_train = cudf.DataFrame.from_pandas(X.to_dataframe().astype('float32'))
        y_cudf_train = cudf.Series(y.to_series())
        self._component_obj.fit(X_cudf_train, y_cudf_train)
        return self

    def predict(self, X):
        predictions_pandas = self._component_obj.predict(cudf.DataFrame.from_pandas(X.to_dataframe().astype('float32'))
                                                         ).to_pandas()
        return infer_feature_types(predictions_pandas)

    @property
    def feature_importance(self):
        return infer_feature_types(self._component_obj.feature_importances_.to_pandas())


class RapidsClassifier(RapidsEstimator):
    """
    Adds predict_proba func for classification models.
    Defines the problem types to standard classification problem types (binary and multiclass)
    """
    supported_problem_types = [ProblemTypes.BINARY, ProblemTypes.MULTICLASS]

    def predict_proba(self, X):
        predictions_pandas = self._component_obj.predict_proba(
            cudf.DataFrame.from_pandas(X.to_dataframe().astype('float32'))
        ).to_pandas()
        return infer_feature_types(predictions_pandas)
