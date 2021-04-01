# -*- coding: utf-8 -*-
import cudf
from cuml import PCA as cuPCA
from evalml.pipelines.components.transformers import Transformer
from evalml.utils.woodwork_utils import infer_feature_types


class RapidsConcatPCA(Transformer):
    # TODO search pca on specific fields
    # TODO create a Rapids Transformer base

    name = 'RAPIDS PCA'
    hyperparameter_ranges = {}

    def __init__(self, random_seed=0, n_components=4, **kwargs):
        parameters = {'n_components': n_components}
        parameters.update(kwargs)
        super().__init__(parameters=parameters,
                         component_obj=cuPCA(random_state=random_seed, **parameters),
                         random_seed=random_seed)

    def fit(self, X, y=None):
        X_cudf = cudf.DataFrame.from_pandas(X.to_dataframe().astype('float32'))
        self._component_obj.fit(X_cudf)
        return self

    def transform(self, X, y=None):
        X_return = X.to_dataframe().copy()
        X_embeded = self._component_obj.transform(cudf.from_pandas(X.to_dataframe()).astype('float32'))

        for i in range(len(X_embeded.columns)):
            X_return[f'component_{i}_fe'] = X_embeded[i].to_array()

        return infer_feature_types(X_return)

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X, y)
