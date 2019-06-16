import numpy as np

from ..common import Feature, Utility, Value


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
        return np.where(np.invert(np.isnan(arr_second_prev)), arr_second - arr_second_prev, Value.MISSING_NUM)

    @classmethod
    def calc_deltasec_valid(cls, arr_second, arr_valid):
        arg_valid_second = np.argwhere(arr_valid == Value.TRUE).flatten()
        arr_valid_second_only = arr_second[arg_valid_second]
        arr_valid_second_only_prev = np.insert(arr_valid_second_only[:-1], 0, np.nan)
        arr_valid_second_prev = np.full(shape=len(arr_second), fill_value=np.nan)
        np.put(arr_valid_second_prev, arg_valid_second, arr_valid_second_only_prev)
        return np.where(np.invert(np.isnan(arr_valid_second_prev)), arr_second - arr_valid_second_prev, Value.MISSING_NUM)

    @classmethod
    def calc_div(cls, arr_numerator, arr_denominator, arr_valid=None, missing_values=(np.nan, Value.MISSING_NUM), default_value=Value.MISSING_NUM):

        # parameter preprocessing
        missing_values = Utility.make_iterable(missing_values)

        # calculation
        if arr_valid is not None:
            return np.where(np.isin(arr_numerator, missing_values) | np.isin(arr_denominator, missing_values) | np.invert(arr_valid.astype(bool)),
                            default_value,
                            arr_numerator / arr_denominator)
        else:
            return np.where(np.isin(arr_numerator, missing_values) | np.isin(arr_denominator, missing_values),
                            default_value,
                            arr_numerator / arr_denominator)

    # INTERMEDIATE METHODS

    @classmethod
    def expand(cls, dataframe):

        # preparation
        dataframe = dataframe.reset_index(drop=True)

        # fill expansion
        dataframe[Feature.FEAT_valid_speed] = np.invert(cls.is_speed_missing(dataframe[Feature.FEAT_speed].values)).astype(int)
        dataframe[Feature.FEAT_valid_bearing] = np.invert(cls.is_bearing_missing(dataframe[Feature.FEAT_bearing].values, dataframe[Feature.FEAT_speed].values)).astype(int)

        dataframe[Feature.FEAT_deltasec] = cls.calc_deltasec(dataframe[Feature.FEAT_second].values)
        dataframe[Feature.FEAT_deltasec_speed] = cls.calc_deltasec_valid(dataframe[Feature.FEAT_second].values, dataframe[Feature.FEAT_valid_speed].values)
        dataframe[Feature.FEAT_deltasec_bearing] = cls.calc_deltasec_valid(dataframe[Feature.FEAT_second].values, dataframe[Feature.FEAT_valid_bearing].values)

        delta_speed_raw = cls.calc_deltasec_valid(dataframe[Feature.FEAT_speed].values, dataframe[Feature.FEAT_valid_speed].values)
        dataframe[Feature.FEAT_delta_speed] = cls.calc_div(delta_speed_raw, dataframe[Feature.FEAT_deltasec_speed].values, dataframe[Feature.FEAT_valid_speed].values)
        delta_bearing_raw = cls.calc_deltasec_valid(dataframe[Feature.FEAT_bearing].values, dataframe[Feature.FEAT_valid_bearing].values)
        dataframe[Feature.FEAT_delta_bearing] = cls.calc_div(delta_bearing_raw, dataframe[Feature.FEAT_deltasec_bearing].values, dataframe[Feature.FEAT_valid_bearing].values)

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
