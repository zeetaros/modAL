import keras.backend as K
from typing import Union
from skorch import NeuralNet
from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor


class KerasException(Exception):
    pass


def initialize_weights(estimator: Union[NeuralNet, KerasRegressor, KerasClassifier]):
    """
    Initializes the weights for the neural network models from PyTorch and Keras.

    Args:
        estimator: the neural network to be initialized.

    Returns:
        estimator: the neural network, with initialized weights.
    """
    if isinstance(estimator, NeuralNet):
        estimator.initialize()
    elif isinstance(estimator, KerasClassifier) or isinstance(estimator, KerasRegressor):
        if hasattr(estimator, 'model'):
            session = K.get_session()
            for layer in estimator.model.layers:
                if hasattr(layer, 'kernel_initializer'):
                    layer.kernel.initializer.run(session=session)
        else:
            raise KerasException('the Keras model has not been fitted yet')
    else:
        raise TypeError('weight reinitialization is only supported for PyTorch and Keras models')

    return estimator
