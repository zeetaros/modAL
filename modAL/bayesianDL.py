import torch
import numpy as np
from skorch import NeuralNet

from typing import Union

from modAL.utils.data import modALinput
from modAL.utils.selection import multi_argmax


EPSILON = 1e-32


def max_entropy_pytorch(learner: NeuralNet, X: modALinput, n_instances: int = 1, n_samples: int = 100,
                        n_subsamples: Union[bool, int] = False):
    if n_subsamples:
        X = np.random.choice(X.shape[0], n_subsamples, replace=False)

    learner.module_.train(True)
    y_preds = torch.cat([learner.predict_proba(X).unsqueeze(dim=0) for _ in range(n_samples)])
    expected_p = torch.mean(y_preds)
    acquisition = torch.sum(expected_p * np.log(expected_p + EPSILON), dim=0)
    query_idx = multi_argmax(acquisition, n_instances=n_instances)
    return query_idx, X[query_idx]
