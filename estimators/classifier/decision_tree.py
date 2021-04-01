# -*- coding: utf-8 -*-
from cuml.ensemble import RandomForestClassifier as curfc
from evalml.model_family import ModelFamily

from estimators.rapids_base_estimator import RapidsClassifier


class RapidsRf(RapidsClassifier):
    name = "Rapids RF"
    model_family = ModelFamily.DECISION_TREE

    def __init__(self, random_seed=0,
                 n_estimators=100, max_depth=16, n_bins=8, max_leaves=-1, max_features='auto',
                 **kwargs):
        parameters = {"n_estimators": n_estimators,
                      "max_depth": max_depth,
                      "n_bins": n_bins,
                      "max_leaves": max_leaves,
                      "max_features": max_features}
        parameters.update(kwargs)
        super().__init__(parameters=parameters,
                         component_obj=curfc(seed=random_seed, **parameters),
                         random_seed=random_seed)
