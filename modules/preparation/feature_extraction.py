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
    def is_raw_missing(cls, num):
        return num == Value.MISSING_NUM

    @classmethod
    def is_bearing_missing(cls, bearing, speed):
        return bearing == 0 & (speed == 0 | cls.is_raw_missing(speed))

    @classmethod
    def calc_diff(cls, arr, arr_valid=None, missing_values=None, default_value=np.nan):

        # parameter preprocessing
        missing_values = Utility.make_iterable(missing_values)

        # calculation
        if arr_valid is not None:
            arg_valid = np.argwhere(arr_valid.astype(bool)).flatten()
            arr_valid_only = arr[arg_valid]
            arr_valid_only_prev = np.insert(arr_valid_only[:-1], 0, np.nan)
            arr_valid_prev = np.full(shape=len(arr), fill_value=np.nan)
            np.put(arr_valid_prev, arg_valid, arr_valid_only_prev)
            return np.where(
                np.isin(arr, missing_values) | np.isin(arr_valid_prev, missing_values) | np.isnan(arr) | np.isnan(arr_valid_prev) | np.invert(arr_valid.astype(bool)),
                default_value,
                arr - arr_valid_prev)
        else:
            arr_prev = np.insert(arr[:-1], 0, np.nan)
            return np.where(
                np.isin(arr, missing_values) | np.isin(arr_prev, missing_values) | np.isnan(arr) | np.isnan(arr_prev),
                default_value,
                arr - arr_prev)

    @classmethod
    def calc_div(cls, arr_numerator, arr_denominator, arr_valid=None, missing_values=None, default_value=np.nan):

        # parameter preprocessing
        missing_values = Utility.make_iterable(missing_values)

        # calculation
        if arr_valid is not None:
            return np.where(
                np.isin(arr_numerator, missing_values) | np.isin(arr_denominator, missing_values) | np.invert(arr_valid.astype(bool)),
                default_value,
                arr_numerator / arr_denominator)
        else:
            return np.where(
                np.isin(arr_numerator, missing_values) | np.isin(arr_denominator, missing_values),
                default_value,
                arr_numerator / arr_denominator)

    @classmethod
    def calc_differential(cls, arr, arr_independent, arr_valid=None, missing_values=None, default_value=np.nan):

        # parameter preprocessing
        missing_values = Utility.make_iterable(missing_values)

        # calculation
        if arr_valid is not None:
            arr_valid_delta = arr_valid.copy()
        else:
            arr_valid_delta = np.full(len(arr), Value.TRUE)
        np.put(arr_valid_delta, np.argmax(arr_valid_delta == Value.TRUE), Value.FALSE)

        delta_raw = cls.calc_diff(
            arr,
            arr_valid,
            missing_values=missing_values,
            default_value=default_value)
        delta_independent = cls.calc_diff(
            arr_independent,
            arr_valid,
            missing_values=missing_values,
            default_value=default_value)
        delta_real = cls.calc_div(
            delta_raw,
            delta_independent,
            arr_valid_delta,
            missing_values=missing_values,
            default_value=default_value)
        return delta_real

    @classmethod
    def calc_scalar(cls, x, y, z):
        return np.sqrt(np.square(x) + np.square(y) + np.square(z))

    # INTERMEDIATE METHODS

    @classmethod
    def expand(cls, dataframe):

        # preparation
        dataframe = dataframe.reset_index(drop=True)

        # fill expansion
        valid_speed = np.invert(cls.is_raw_missing(
            dataframe[Feature.FEAT_speed].values)).astype(int)
        valid_bearing = np.invert(cls.is_bearing_missing(
            dataframe[Feature.FEAT_bearing].values,
            dataframe[Feature.FEAT_speed].values)).astype(int)
        dataframe[Feature.FEAT_valid_speed] = valid_speed
        dataframe[Feature.FEAT_valid_bearing] = valid_bearing

        dataframe[Feature.FEAT_scalar_acceleration] = cls.calc_scalar(
            dataframe[Feature.FEAT_acceleration_x].values,
            dataframe[Feature.FEAT_acceleration_y].values,
            dataframe[Feature.FEAT_acceleration_z].values)
        dataframe[Feature.FEAT_scalar_gyro] = cls.calc_scalar(
            dataframe[Feature.FEAT_gyro_x].values,
            dataframe[Feature.FEAT_gyro_y].values,
            dataframe[Feature.FEAT_gyro_z].values)

        dataframe[Feature.FEAT_deltasec] = cls.calc_diff(
            dataframe[Feature.FEAT_second].values,
            default_value=np.nan)
        dataframe[Feature.FEAT_deltasec_speed] = cls.calc_diff(
            dataframe[Feature.FEAT_second].values,
            valid_speed,
            default_value=np.nan)
        dataframe[Feature.FEAT_deltasec_bearing] = cls.calc_diff(
            dataframe[Feature.FEAT_second].values,
            valid_bearing,
            default_value=np.nan)

        dataframe[Feature.FEAT_delta_speed] = cls.calc_differential(
            dataframe[Feature.FEAT_speed].values,
            dataframe[Feature.FEAT_second].values,
            valid_speed,
            default_value=np.nan)
        dataframe[Feature.FEAT_delta_bearing] = cls.calc_differential(
            dataframe[Feature.FEAT_bearing].values,
            dataframe[Feature.FEAT_second].values,
            valid_bearing,
            default_value=np.nan)
        dataframe[Feature.FEAT_delta_scalar_acceleration] = cls.calc_differential(
            dataframe[Feature.FEAT_scalar_acceleration].values,
            dataframe[Feature.FEAT_second].values,
            default_value=np.nan)
        dataframe[Feature.FEAT_delta_scalar_gyro] = cls.calc_differential(
            dataframe[Feature.FEAT_scalar_gyro].values,
            dataframe[Feature.FEAT_second].values,
            default_value=np.nan)
        dataframe[Feature.FEAT_delta_acceleration_x] = cls.calc_differential(
            dataframe[Feature.FEAT_acceleration_x].values,
            dataframe[Feature.FEAT_second].values,
            default_value=np.nan)
        dataframe[Feature.FEAT_delta_acceleration_y] = cls.calc_differential(
            dataframe[Feature.FEAT_acceleration_y].values,
            dataframe[Feature.FEAT_second].values,
            default_value=np.nan)
        dataframe[Feature.FEAT_delta_acceleration_z] = cls.calc_differential(
            dataframe[Feature.FEAT_acceleration_z].values,
            dataframe[Feature.FEAT_second].values,
            default_value=np.nan)
        dataframe[Feature.FEAT_delta_gyro_x] = cls.calc_differential(
            dataframe[Feature.FEAT_gyro_x].values,
            dataframe[Feature.FEAT_second].values,
            default_value=np.nan)
        dataframe[Feature.FEAT_delta_gyro_y] = cls.calc_differential(
            dataframe[Feature.FEAT_gyro_x].values,
            dataframe[Feature.FEAT_second].values,
            default_value=np.nan)
        dataframe[Feature.FEAT_delta_gyro_z] = cls.calc_differential(
            dataframe[Feature.FEAT_gyro_x].values,
            dataframe[Feature.FEAT_second].values,
            default_value=np.nan)

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
