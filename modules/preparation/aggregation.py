import numpy as np


def mean(a):
    return np.mean(a)


def aggregate(inputs=[], base_name='aggregation'):

    # parameter preprocessing
    if not isinstance(inputs, list):
        inputs = [inputs]

    # TODO implement aggregation by bookingID
    return