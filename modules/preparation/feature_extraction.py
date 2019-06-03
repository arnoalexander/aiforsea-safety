import numpy as np

from .. import common


class FeatureExtraction:

    # HELPER METHODS

    @classmethod
    def get_booking_id(cls, dataframe):
        return dataframe[common.Feature.FEAT_booking_id].values[0]

    # MAIN METHODS

    @classmethod
    def switch(cls, dataframe, feature):  # TODO add more extraction by feature
        if feature == common.Feature.FEAT_booking_id:
            return cls.get_booking_id(dataframe)
        else:
            return -1

    @classmethod
    def run(cls, dataframe, features):
        result_list = []
        for feature in features:
            result_list.append(cls.switch(dataframe, feature))
        return np.array(result_list)
