from zoo.automl.feature.base import BaseEstimator
from sklearn.exceptions import NotFittedError
import numpy as np

def check_is_fitted(estimator, attributes):
    """
    check if the estimator is fitted by attributes
    :param estimator:
    :param attributes: string or a list/tuple of strings
    :return:
    """
    msg = ("This %(name)s instance is not fitted yet. Call 'fit' with "
           "appropriate arguments before using this method.")

    if not isinstance(estimator, BaseEstimator):
        raise TypeError("%s is not an estimator instance." % estimator)

    if not isinstance(attributes, (list, tuple)):
        attributes = [attributes]

    if not all([hasattr(estimator, attr) for attr in attributes]):
        raise NotFittedError(msg % {'name': type(estimator).__name__})


def check_input_array(inputs):
    """
    check if the inputs is array-like and return numpy array inputs
    :param inputs:
    :return:
    """
    if inputs is None:
        raise ValueError("Got None input.")
    if not hasattr(inputs, '__len__') and not hasattr(inputs, 'shape'):
        if hasattr(inputs, '__array__'):
            inputs = np.asarray(inputs)
        else:
            raise TypeError("Expected array-like input, got %s" % type(inputs))
    if len(inputs) == 0:
        raise ValueError("Input is an empty array.")
    return inputs
