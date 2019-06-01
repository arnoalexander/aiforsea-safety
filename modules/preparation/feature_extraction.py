import numpy as np

from .. import common


class FeatureExtraction:

    # CONSTANT

    EXTRACTED = [common.Feature.FEAT_booking_id,
                 'A',
                 'B']  # TODO add more extracted features

    # HELPER METHODS

    @classmethod
    def get_booking_id(cls, dataframe):
        return dataframe[common.Feature.FEAT_booking_id].values[0]

    # MAIN METHODS

    @classmethod
    def switch(cls, feature, dataframe):  # TODO add more extraction by feature
        if feature == common.Feature.FEAT_booking_id:
            return cls.get_booking_id(dataframe)
        else:
            return -1

    @classmethod
    def run(cls, dataframe):
        result_list = []
        for feat in cls.EXTRACTED:
            result_list.append(cls.switch(feat, dataframe))
        return np.array(result_list)
