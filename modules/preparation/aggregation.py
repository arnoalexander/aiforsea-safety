import numpy as np


class Aggregation:

    @staticmethod
    def mean(a):
        return np.mean(a)

    @classmethod
    def aggregate(cls, inputs=[]):

        # parameter preprocessing
        if not isinstance(inputs, list):
            inputs = [inputs]

        # TODO implement aggregation by bookingID
        return
