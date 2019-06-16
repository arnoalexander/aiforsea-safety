import numpy as np


class Utility:

    @classmethod
    def make_iterable(cls, obj):
        if obj is None:
            return []
        elif isinstance(obj, (list, tuple, np.ndarray)):
            return obj
        return [obj]
