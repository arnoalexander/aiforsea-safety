import numpy as np

from .. import common


class FeatureExtraction:

    # HELPER METHODS

    @classmethod
    def get_element(cls, array, at):
        return array[at]

    @classmethod
    def mean(cls, array):
        return np.mean(array)

    # MAIN METHODS

    @classmethod
    def switch(cls, dataframe, feature):  # TODO add more extraction by feature
        if feature == common.Feature.FEAT_booking_id:
            return cls.get_element(dataframe[feature].values, 0)
        elif feature == common.Feature.FEAT_mean_accuracy:
            return cls.mean(dataframe[common.Feature.FEAT_accuracy].values)
        elif feature == common.Feature.FEAT_mean_bearing:
            return cls.mean(dataframe[common.Feature.FEAT_bearing].values)
        elif feature == common.Feature.FEAT_mean_acceleration_x:
            return cls.mean(dataframe[common.Feature.FEAT_acceleration_x].values)
        elif feature == common.Feature.FEAT_mean_acceleration_y:
            return cls.mean(dataframe[common.Feature.FEAT_acceleration_y].values)
        elif feature == common.Feature.FEAT_mean_acceleration_z:
            return cls.mean(dataframe[common.Feature.FEAT_acceleration_z].values)
        elif feature == common.Feature.FEAT_mean_gyro_x:
            return cls.mean(dataframe[common.Feature.FEAT_gyro_x].values)
        elif feature == common.Feature.FEAT_mean_gyro_y:
            return cls.mean(dataframe[common.Feature.FEAT_gyro_y].values)
        elif feature == common.Feature.FEAT_mean_gyro_z:
            return cls.mean(dataframe[common.Feature.FEAT_gyro_z].values)
        elif feature == common.Feature.FEAT_mean_speed:
            return cls.mean(dataframe[common.Feature.FEAT_speed].values)
        else:
            return np.nan

    @classmethod
    def run(cls, dataframe, features):
        result_list = []
        for feature in features:
            result_list.append(cls.switch(dataframe, feature))
        return np.array(result_list)
