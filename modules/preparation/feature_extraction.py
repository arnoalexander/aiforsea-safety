import numpy as np

from ..common import Feature


class FeatureExtraction:

    # HELPER METHODS

    @classmethod
    def get_element(cls, array, at):
        return array[at]

    @classmethod
    def mean(cls, array):
        return np.mean(array)

    # INTERMEDIATE METHODS

    @classmethod
    def expand(cls, dataframe):  # TODO generate new intermediate features in original data
        return dataframe

    @classmethod
    def extract(cls, dataframe):  # TODO add more extraction by feature

        result_dict = dict()

        # primary key
        result_dict[Feature.FEAT_booking_id] = cls.get_element(dataframe[Feature.FEAT_booking_id].values, 0)

        # mean features
        result_dict[Feature.FEAT_mean_accuracy] = cls.mean(dataframe[Feature.FEAT_accuracy].values)
        result_dict[Feature.FEAT_mean_bearing] = cls.mean(dataframe[Feature.FEAT_bearing].values)
        result_dict[Feature.FEAT_mean_acceleration_x] = cls.mean(dataframe[Feature.FEAT_acceleration_x].values)
        result_dict[Feature.FEAT_mean_acceleration_y] = cls.mean(dataframe[Feature.FEAT_acceleration_y].values)
        result_dict[Feature.FEAT_mean_acceleration_z] = cls.mean(dataframe[Feature.FEAT_acceleration_z].values)
        result_dict[Feature.FEAT_mean_gyro_x] = cls.mean(dataframe[Feature.FEAT_gyro_x].values)
        result_dict[Feature.FEAT_mean_gyro_y] = cls.mean(dataframe[Feature.FEAT_gyro_y].values)
        result_dict[Feature.FEAT_mean_gyro_z] = cls.mean(dataframe[Feature.FEAT_gyro_z].values)
        result_dict[Feature.FEAT_mean_speed] = cls.mean(dataframe[Feature.FEAT_speed].values)

        return result_dict

    # MAIN METHOD

    @classmethod
    def run(cls, dataframe):
        extended_dataframe = cls.expand(dataframe)
        return cls.extract(extended_dataframe)
