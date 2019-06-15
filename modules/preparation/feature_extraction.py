import numpy as np

from ..common import Feature, Value


class FeatureExtraction:

    # HELPER METHODS

    @classmethod
    def get_element(cls, array, at):
        return array[at]

    @classmethod
    def mean(cls, array):
        return np.mean(array)

    @classmethod
    def is_speed_missing(cls, speed):
        return speed == -1

    @classmethod
    def is_bearing_missing(cls, bearing, speed):
        return bearing == 0 & (speed == 0 | cls.is_speed_missing(speed))

    @classmethod
    def calc_deltasec(cls, arr_second):
        arr_second_prev = np.insert(arr_second[:-1], 0, np.nan)
        return np.where(arr_second_prev != np.nan, arr_second - arr_second_prev, Value.MISSING_NUM)

    # INTERMEDIATE METHODS

    @classmethod
    def expand(cls, dataframe):

        # preparation
        dataframe = dataframe.reset_index(drop=True)

        # fill expansion
        dataframe[Feature.FEAT_valid_speed] = np.invert(cls.is_speed_missing(dataframe[Feature.FEAT_speed].values)).astype(int)
        dataframe[Feature.FEAT_valid_bearing] = np.invert(cls.is_bearing_missing(dataframe[Feature.FEAT_bearing].values, dataframe[Feature.FEAT_speed].values)).astype(int)
        dataframe[Feature.FEAT_deltasec] = cls.calc_deltasec(dataframe[Feature.FEAT_second].values)

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
