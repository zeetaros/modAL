import torch
import numpy as np

from typing import Union

from modAL import ActiveLearner
from modAL.utils.data import modALinput
from modAL.utils.selection import multi_argmax


EPSILON = 1e-32


def max_entropy_pytorch(learner: ActiveLearner, X: modALinput, n_instances: int = 1, n_samples: int = 100,
                        n_subsamples: Union[bool, int] = False):
    if n_subsamples:
        X = np.random.choice(X.shape[0], n_subsamples, replace=False)

    learner.estimator.module_.train(True)
    y_preds = np.vstack([np.expand_dims(learner.predict_proba(X), axis=0) for _ in range(n_samples)])
    expected_p = y_preds.mean(axis=0)
    acquisition = (expected_p * np.log(expected_p + EPSILON)).sum(axis=1)
    query_idx = multi_argmax(acquisition, n_instances=n_instances)
    return query_idx, X[query_idx]
