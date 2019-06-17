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
    def is_num_missing(cls, num):
        return num == Value.MISSING_NUM

    @classmethod
    def is_bearing_missing(cls, bearing, speed):
        return bearing == 0 & (speed == 0 | cls.is_num_missing(speed))

    @classmethod
    def diff_angle_raw(cls, angle_from, angle_to):
        result = (angle_to - angle_from) % Value.ANGLE_FULL
        return np.where(result > Value.ANGLE_FULL / 2, Value.ANGLE_FULL - result, result)

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

        arr_prev = np.insert(arr[:-1], 0, np.nan)
        return np.where(
            np.isin(arr, missing_values) | np.isin(arr_prev, missing_values) | np.isnan(arr) | np.isnan(arr_prev),
            default_value,
            arr - arr_prev)

    @classmethod
    def calc_diff_angle(cls, arr, arr_valid=None, missing_values=None, default_value=np.nan):

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
                np.isin(arr, missing_values) | np.isin(arr_valid_prev, missing_values) | np.isnan(arr) | np.isnan(
                    arr_valid_prev) | np.invert(arr_valid.astype(bool)),
                default_value,
                cls.diff_angle_raw(arr_valid_prev, arr))
        else:
            arr_prev = np.insert(arr[:-1], 0, np.nan)
            return np.where(
                np.isin(arr, missing_values) | np.isin(arr_prev, missing_values) | np.isnan(arr) | np.isnan(arr_prev),
                default_value,
                cls.diff_angle_raw(arr_prev, arr))

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
    def calc_differential_angle(cls, arr, arr_independent, arr_valid=None, missing_values=None, default_value=np.nan):
        # parameter preprocessing
        missing_values = Utility.make_iterable(missing_values)

        # calculation
        if arr_valid is not None:
            arr_valid_delta = arr_valid.copy()
        else:
            arr_valid_delta = np.full(len(arr), Value.TRUE)
        np.put(arr_valid_delta, np.argmax(arr_valid_delta == Value.TRUE), Value.FALSE)

        delta_raw = cls.calc_diff_angle(
            arr,
            arr_valid,
            missing_values=missing_values,
            default_value=default_value)
        delta_independent = cls.calc_diff_angle(
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
        valid_speed = np.invert(cls.is_num_missing(
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
        dataframe[Feature.FEAT_delta_bearing] = cls.calc_differential_angle(
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
            dataframe[Feature.FEAT_gyro_y].values,
            dataframe[Feature.FEAT_second].values,
            default_value=np.nan)
        dataframe[Feature.FEAT_delta_gyro_z] = cls.calc_differential(
            dataframe[Feature.FEAT_gyro_z].values,
            dataframe[Feature.FEAT_second].values,
            default_value=np.nan)

        return dataframe

    @classmethod
    def extract(cls, dataframe):

        result_dict = dict()

        # primary key
        result_dict[Feature.FEAT_booking_id] = cls.get_element(dataframe[Feature.FEAT_booking_id].values, 0)

        # data pre-loading
        data_second = dataframe[Feature.FEAT_second].values
        data_valid_bearing = dataframe[Feature.FEAT_valid_bearing].values
        data_valid_speed = dataframe[Feature.FEAT_valid_speed].values
        data_accuracy = dataframe[Feature.FEAT_accuracy].values
        data_speed = np.flip(np.sort(
            dataframe[dataframe[Feature.FEAT_valid_speed] == Value.TRUE][Feature.FEAT_speed].values))
        data_acceleration_x = np.flip(np.sort(dataframe[Feature.FEAT_acceleration_x].values))
        data_acceleration_y = np.flip(np.sort(dataframe[Feature.FEAT_acceleration_y].values))
        data_acceleration_z = np.flip(np.sort(dataframe[Feature.FEAT_acceleration_z].values))
        data_gyro_x = np.flip(np.sort(dataframe[Feature.FEAT_gyro_x].values))
        data_gyro_y = np.flip(np.sort(dataframe[Feature.FEAT_gyro_y].values))
        data_gyro_z = np.flip(np.sort(dataframe[Feature.FEAT_gyro_z].values))

        data_scalar_acceleration = np.flip(np.sort(dataframe[Feature.FEAT_scalar_acceleration].values))
        data_scalar_gyro = np.flip(np.sort(dataframe[Feature.FEAT_scalar_gyro].values))
        data_delta_scalar_acceleration = np.flip(np.sort(
            dataframe[np.invert(np.isnan(dataframe[Feature.FEAT_delta_scalar_acceleration]))]
            [Feature.FEAT_delta_scalar_acceleration].values))
        data_delta_scalar_gyro = np.flip(np.sort(
            dataframe[np.invert(np.isnan(dataframe[Feature.FEAT_delta_scalar_gyro]))]
            [Feature.FEAT_delta_scalar_gyro].values))
        data_delta_bearing = np.flip(np.sort(
            dataframe[np.invert(np.isnan(dataframe[Feature.FEAT_delta_bearing]))]
            [Feature.FEAT_delta_bearing].values))
        data_delta_speed = np.flip(np.sort(
            dataframe[np.invert(np.isnan(dataframe[Feature.FEAT_delta_speed]))]
            [Feature.FEAT_delta_speed].values))
        data_delta_acceleration_x = np.flip(np.sort(
            dataframe[np.invert(np.isnan(dataframe[Feature.FEAT_delta_acceleration_x]))]
            [Feature.FEAT_delta_acceleration_x].values))
        data_delta_acceleration_y = np.flip(np.sort(
            dataframe[np.invert(np.isnan(dataframe[Feature.FEAT_delta_acceleration_y]))]
            [Feature.FEAT_delta_acceleration_y].values))
        data_delta_acceleration_z = np.flip(np.sort(
            dataframe[np.invert(np.isnan(dataframe[Feature.FEAT_delta_acceleration_z]))]
            [Feature.FEAT_delta_acceleration_z].values))
        data_delta_gyro_x = np.flip(np.sort(
            dataframe[np.invert(np.isnan(dataframe[Feature.FEAT_delta_gyro_x]))]
            [Feature.FEAT_delta_gyro_x].values))
        data_delta_gyro_y = np.flip(np.sort(
            dataframe[np.invert(np.isnan(dataframe[Feature.FEAT_delta_gyro_y]))]
            [Feature.FEAT_delta_gyro_y].values))
        data_delta_gyro_z = np.flip(np.sort(
            dataframe[np.invert(np.isnan(dataframe[Feature.FEAT_delta_gyro_z]))]
            [Feature.FEAT_delta_gyro_z].values))

        # all-data features
        result_dict[Feature.FEAT_percent_active] = len(data_second) / np.max(data_second)
        result_dict[Feature.FEAT_percent_active_bearing] = np.sum(data_valid_bearing) / len(data_valid_bearing)
        result_dict[Feature.FEAT_percent_active_speed] = np.sum(data_valid_speed) / len(data_valid_speed)

        result_dict[Feature.FEAT_mean_accuracy] = np.mean(data_accuracy)
        result_dict[Feature.FEAT_best_accuracy] = np.min(data_accuracy)
        result_dict[Feature.FEAT_worst_accuracy] = np.max(data_accuracy)
        result_dict[Feature.FEAT_stddev_accuracy] = np.std(data_accuracy)

        result_dict[Feature.FEAT_mean_speed] = np.mean(data_speed)
        result_dict[Feature.FEAT_stddev_speed] = np.std(data_speed)
        result_dict[Feature.FEAT_mean_scalar_gyro] = np.mean(data_scalar_gyro)
        result_dict[Feature.FEAT_stddev_scalar_gyro] = np.std(data_scalar_gyro)
        result_dict[Feature.FEAT_mean_scalar_acceleration] = np.mean(data_scalar_acceleration)
        result_dict[Feature.FEAT_stddev_scalar_acceleration] = np.std(data_scalar_acceleration)
        result_dict[Feature.FEAT_mean_acceleration_x] = np.mean(data_acceleration_x)
        result_dict[Feature.FEAT_stddev_acceleration_x] = np.std(data_acceleration_x)
        result_dict[Feature.FEAT_mean_acceleration_y] = np.mean(data_acceleration_y)
        result_dict[Feature.FEAT_stddev_acceleration_y] = np.std(data_acceleration_y)
        result_dict[Feature.FEAT_mean_acceleration_z] = np.mean(data_acceleration_z)
        result_dict[Feature.FEAT_stddev_acceleration_z] = np.std(data_acceleration_z)
        result_dict[Feature.FEAT_mean_gyro_x] = np.mean(data_gyro_x)
        result_dict[Feature.FEAT_stddev_gyro_x] = np.std(data_gyro_x)
        result_dict[Feature.FEAT_mean_gyro_y] = np.mean(data_gyro_y)
        result_dict[Feature.FEAT_stddev_gyro_y] = np.std(data_gyro_y)
        result_dict[Feature.FEAT_mean_gyro_z] = np.mean(data_gyro_z)
        result_dict[Feature.FEAT_stddev_gyro_z] = np.std(data_gyro_z)

        result_dict[Feature.FEAT_mean_delta_bearing] = np.mean(data_delta_bearing)
        result_dict[Feature.FEAT_stddev_delta_bearing] = np.std(data_delta_bearing)
        result_dict[Feature.FEAT_mean_delta_speed] = np.mean(data_delta_speed)
        result_dict[Feature.FEAT_stddev_delta_speed] = np.std(data_delta_speed)
        result_dict[Feature.FEAT_mean_delta_scalar_gyro] = np.mean(data_delta_scalar_gyro)
        result_dict[Feature.FEAT_stddev_delta_scalar_gyro] = np.std(data_delta_scalar_gyro)
        result_dict[Feature.FEAT_mean_delta_scalar_acceleration] = np.mean(data_delta_scalar_acceleration)
        result_dict[Feature.FEAT_stddev_delta_scalar_acceleration] = np.std(data_delta_scalar_acceleration)
        result_dict[Feature.FEAT_mean_delta_gyro_x] = np.mean(data_delta_gyro_x)
        result_dict[Feature.FEAT_stddev_delta_gyro_x] = np.std(data_delta_gyro_x)
        result_dict[Feature.FEAT_mean_delta_gyro_y] = np.mean(data_delta_gyro_y)
        result_dict[Feature.FEAT_stddev_delta_gyro_y] = np.std(data_delta_gyro_y)
        result_dict[Feature.FEAT_mean_delta_gyro_z] = np.mean(data_delta_gyro_z)
        result_dict[Feature.FEAT_stddev_delta_gyro_z] = np.std(data_delta_gyro_z)
        result_dict[Feature.FEAT_mean_delta_acceleration_x] = np.mean(data_delta_acceleration_x)
        result_dict[Feature.FEAT_stddev_delta_acceleration_x] = np.std(data_delta_acceleration_x)
        result_dict[Feature.FEAT_mean_delta_acceleration_y] = np.mean(data_delta_acceleration_y)
        result_dict[Feature.FEAT_stddev_delta_acceleration_y] = np.std(data_delta_acceleration_y)
        result_dict[Feature.FEAT_mean_delta_acceleration_z] = np.mean(data_delta_acceleration_z)
        result_dict[Feature.FEAT_stddev_delta_acceleration_z] = np.std(data_delta_acceleration_z)

        # 20% data feature
        result_dict[Feature.FEAT_mean20_speed] = np.mean(data_speed[:int(len(data_speed)*0.2)])
        result_dict[Feature.FEAT_stddev20_speed] = np.std(data_speed[:int(len(data_speed)*0.2)])
        result_dict[Feature.FEAT_mean20_scalar_gyro] = np.mean(data_scalar_gyro[:int(len(data_scalar_gyro)*0.2)])
        result_dict[Feature.FEAT_stddev20_scalar_gyro] = np.std(data_scalar_gyro[:int(len(data_scalar_gyro)*0.2)])
        result_dict[Feature.FEAT_mean20_scalar_acceleration] = np.mean(data_scalar_acceleration[:int(len(data_scalar_acceleration)*0.2)])
        result_dict[Feature.FEAT_stddev20_scalar_acceleration] = np.std(data_scalar_acceleration[:int(len(data_scalar_acceleration)*0.2)])
        result_dict[Feature.FEAT_mean20_acceleration_x] = np.mean(data_acceleration_x[:int(len(data_acceleration_x)*0.2)])
        result_dict[Feature.FEAT_stddev20_acceleration_x] = np.std(data_acceleration_x[:int(len(data_acceleration_x)*0.2)])
        result_dict[Feature.FEAT_mean20_acceleration_y] = np.mean(data_acceleration_y[:int(len(data_acceleration_y)*0.2)])
        result_dict[Feature.FEAT_stddev20_acceleration_y] = np.std(data_acceleration_y[:int(len(data_acceleration_y)*0.2)])
        result_dict[Feature.FEAT_mean20_acceleration_z] = np.mean(data_acceleration_z[:int(len(data_acceleration_z)*0.2)])
        result_dict[Feature.FEAT_stddev20_acceleration_z] = np.std(data_acceleration_z[:int(len(data_acceleration_z)*0.2)])
        result_dict[Feature.FEAT_mean20_gyro_x] = np.mean(data_gyro_x[:int(len(data_gyro_x)*0.2)])
        result_dict[Feature.FEAT_stddev20_gyro_x] = np.std(data_gyro_x[:int(len(data_gyro_x)*0.2)])
        result_dict[Feature.FEAT_mean20_gyro_y] = np.mean(data_gyro_y[:int(len(data_gyro_y)*0.2)])
        result_dict[Feature.FEAT_stddev20_gyro_y] = np.std(data_gyro_y[:int(len(data_gyro_y)*0.2)])
        result_dict[Feature.FEAT_mean20_gyro_z] = np.mean(data_gyro_z[:int(len(data_gyro_z)*0.2)])
        result_dict[Feature.FEAT_stddev20_gyro_z] = np.std(data_gyro_z[:int(len(data_gyro_z)*0.2)])

        result_dict[Feature.FEAT_mean20_delta_bearing] = np.mean(data_delta_bearing[:int(len(data_delta_bearing)*0.2)])
        result_dict[Feature.FEAT_stddev20_delta_bearing] = np.std(data_delta_bearing[:int(len(data_delta_bearing)*0.2)])
        result_dict[Feature.FEAT_mean20_delta_speed] = np.mean(data_delta_speed[:int(len(data_delta_speed)*0.2)])
        result_dict[Feature.FEAT_stddev20_delta_speed] = np.std(data_delta_speed[:int(len(data_delta_speed)*0.2)])
        result_dict[Feature.FEAT_mean20_delta_scalar_gyro] = np.mean(data_delta_scalar_gyro[:int(len(data_delta_scalar_gyro)*0.2)])
        result_dict[Feature.FEAT_stddev20_delta_scalar_gyro] = np.std(data_delta_scalar_gyro[:int(len(data_delta_scalar_gyro)*0.2)])
        result_dict[Feature.FEAT_mean20_delta_scalar_acceleration] = np.mean(data_delta_scalar_acceleration[:int(len(data_delta_scalar_acceleration)*0.2)])
        result_dict[Feature.FEAT_stddev20_delta_scalar_acceleration] = np.std(data_delta_scalar_acceleration[:int(len(data_delta_scalar_acceleration)*0.2)])
        result_dict[Feature.FEAT_mean20_delta_gyro_x] = np.mean(data_delta_gyro_x[:int(len(data_delta_gyro_x)*0.2)])
        result_dict[Feature.FEAT_stddev20_delta_gyro_x] = np.std(data_delta_gyro_x[:int(len(data_delta_gyro_x)*0.2)])
        result_dict[Feature.FEAT_mean20_delta_gyro_y] = np.mean(data_delta_gyro_y[:int(len(data_delta_gyro_y)*0.2)])
        result_dict[Feature.FEAT_stddev20_delta_gyro_y] = np.std(data_delta_gyro_y[:int(len(data_delta_gyro_y)*0.2)])
        result_dict[Feature.FEAT_mean20_delta_gyro_z] = np.mean(data_delta_gyro_z[:int(len(data_delta_gyro_z)*0.2)])
        result_dict[Feature.FEAT_stddev20_delta_gyro_z] = np.std(data_delta_gyro_z[:int(len(data_delta_gyro_z)*0.2)])
        result_dict[Feature.FEAT_mean20_delta_acceleration_x] = np.mean(data_delta_acceleration_x[:int(len(data_delta_acceleration_x)*0.2)])
        result_dict[Feature.FEAT_stddev20_delta_acceleration_x] = np.std(data_delta_acceleration_x[:int(len(data_delta_acceleration_x)*0.2)])
        result_dict[Feature.FEAT_mean20_delta_acceleration_y] = np.mean(data_delta_acceleration_y[:int(len(data_delta_acceleration_y)*0.2)])
        result_dict[Feature.FEAT_stddev20_delta_acceleration_y] = np.std(data_delta_acceleration_y[:int(len(data_delta_acceleration_y)*0.2)])
        result_dict[Feature.FEAT_mean20_delta_acceleration_z] = np.mean(data_delta_acceleration_z[:int(len(data_delta_acceleration_z)*0.2)])
        result_dict[Feature.FEAT_stddev20_delta_acceleration_z] = np.std(data_delta_acceleration_z[:int(len(data_delta_acceleration_z)*0.2)])

        # 5% data feature
        result_dict[Feature.FEAT_mean5_speed] = np.mean(data_speed[:int(len(data_speed) * 0.05)])
        result_dict[Feature.FEAT_stddev5_speed] = np.std(data_speed[:int(len(data_speed) * 0.05)])
        result_dict[Feature.FEAT_mean5_scalar_gyro] = np.mean(data_scalar_gyro[:int(len(data_scalar_gyro) * 0.05)])
        result_dict[Feature.FEAT_stddev5_scalar_gyro] = np.std(data_scalar_gyro[:int(len(data_scalar_gyro) * 0.05)])
        result_dict[Feature.FEAT_mean5_scalar_acceleration] = np.mean(data_scalar_acceleration[:int(len(data_scalar_acceleration) * 0.05)])
        result_dict[Feature.FEAT_stddev5_scalar_acceleration] = np.std(data_scalar_acceleration[:int(len(data_scalar_acceleration) * 0.05)])
        result_dict[Feature.FEAT_mean5_acceleration_x] = np.mean(data_acceleration_x[:int(len(data_acceleration_x) * 0.05)])
        result_dict[Feature.FEAT_stddev5_acceleration_x] = np.std(data_acceleration_x[:int(len(data_acceleration_x) * 0.05)])
        result_dict[Feature.FEAT_mean5_acceleration_y] = np.mean(data_acceleration_y[:int(len(data_acceleration_y) * 0.05)])
        result_dict[Feature.FEAT_stddev5_acceleration_y] = np.std(data_acceleration_y[:int(len(data_acceleration_y) * 0.05)])
        result_dict[Feature.FEAT_mean5_acceleration_z] = np.mean(data_acceleration_z[:int(len(data_acceleration_z) * 0.05)])
        result_dict[Feature.FEAT_stddev5_acceleration_z] = np.std(data_acceleration_z[:int(len(data_acceleration_z) * 0.05)])
        result_dict[Feature.FEAT_mean5_gyro_x] = np.mean(data_gyro_x[:int(len(data_gyro_x) * 0.05)])
        result_dict[Feature.FEAT_stddev5_gyro_x] = np.std(data_gyro_x[:int(len(data_gyro_x) * 0.05)])
        result_dict[Feature.FEAT_mean5_gyro_y] = np.mean(data_gyro_y[:int(len(data_gyro_y) * 0.05)])
        result_dict[Feature.FEAT_stddev5_gyro_y] = np.std(data_gyro_y[:int(len(data_gyro_y) * 0.05)])
        result_dict[Feature.FEAT_mean5_gyro_z] = np.mean(data_gyro_z[:int(len(data_gyro_z) * 0.05)])
        result_dict[Feature.FEAT_stddev5_gyro_z] = np.std(data_gyro_z[:int(len(data_gyro_z) * 0.05)])

        result_dict[Feature.FEAT_mean5_delta_bearing] = np.mean(data_delta_bearing[:int(len(data_delta_bearing) * 0.05)])
        result_dict[Feature.FEAT_stddev5_delta_bearing] = np.std(data_delta_bearing[:int(len(data_delta_bearing) * 0.05)])
        result_dict[Feature.FEAT_mean5_delta_speed] = np.mean(data_delta_speed[:int(len(data_delta_speed) * 0.05)])
        result_dict[Feature.FEAT_stddev5_delta_speed] = np.std(data_delta_speed[:int(len(data_delta_speed) * 0.05)])
        result_dict[Feature.FEAT_mean5_delta_scalar_gyro] = np.mean(data_delta_scalar_gyro[:int(len(data_delta_scalar_gyro) * 0.05)])
        result_dict[Feature.FEAT_stddev5_delta_scalar_gyro] = np.std(data_delta_scalar_gyro[:int(len(data_delta_scalar_gyro) * 0.05)])
        result_dict[Feature.FEAT_mean5_delta_scalar_acceleration] = np.mean(data_delta_scalar_acceleration[:int(len(data_delta_scalar_acceleration) * 0.05)])
        result_dict[Feature.FEAT_stddev5_delta_scalar_acceleration] = np.std(data_delta_scalar_acceleration[:int(len(data_delta_scalar_acceleration) * 0.05)])
        result_dict[Feature.FEAT_mean5_delta_gyro_x] = np.mean(data_delta_gyro_x[:int(len(data_delta_gyro_x) * 0.05)])
        result_dict[Feature.FEAT_stddev5_delta_gyro_x] = np.std(data_delta_gyro_x[:int(len(data_delta_gyro_x) * 0.05)])
        result_dict[Feature.FEAT_mean5_delta_gyro_y] = np.mean(data_delta_gyro_y[:int(len(data_delta_gyro_y) * 0.05)])
        result_dict[Feature.FEAT_stddev5_delta_gyro_y] = np.std(data_delta_gyro_y[:int(len(data_delta_gyro_y) * 0.05)])
        result_dict[Feature.FEAT_mean5_delta_gyro_z] = np.mean(data_delta_gyro_z[:int(len(data_delta_gyro_z) * 0.05)])
        result_dict[Feature.FEAT_stddev5_delta_gyro_z] = np.std(data_delta_gyro_z[:int(len(data_delta_gyro_z) * 0.05)])
        result_dict[Feature.FEAT_mean5_delta_acceleration_x] = np.mean(data_delta_acceleration_x[:int(len(data_delta_acceleration_x) * 0.05)])
        result_dict[Feature.FEAT_stddev5_delta_acceleration_x] = np.std(data_delta_acceleration_x[:int(len(data_delta_acceleration_x) * 0.05)])
        result_dict[Feature.FEAT_mean5_delta_acceleration_y] = np.mean(data_delta_acceleration_y[:int(len(data_delta_acceleration_y) * 0.05)])
        result_dict[Feature.FEAT_stddev5_delta_acceleration_y] = np.std(data_delta_acceleration_y[:int(len(data_delta_acceleration_y) * 0.05)])
        result_dict[Feature.FEAT_mean5_delta_acceleration_z] = np.mean(data_delta_acceleration_z[:int(len(data_delta_acceleration_z) * 0.05)])
        result_dict[Feature.FEAT_stddev5_delta_acceleration_z] = np.std(data_delta_acceleration_z[:int(len(data_delta_acceleration_z) * 0.05)])

        # 1% data feature
        result_dict[Feature.FEAT_mean1_speed] = np.mean(data_speed[:int(len(data_speed) * 0.01)])
        result_dict[Feature.FEAT_stddev1_speed] = np.std(data_speed[:int(len(data_speed) * 0.01)])
        result_dict[Feature.FEAT_mean1_scalar_gyro] = np.mean(data_scalar_gyro[:int(len(data_scalar_gyro) * 0.01)])
        result_dict[Feature.FEAT_stddev1_scalar_gyro] = np.std(data_scalar_gyro[:int(len(data_scalar_gyro) * 0.01)])
        result_dict[Feature.FEAT_mean1_scalar_acceleration] = np.mean(data_scalar_acceleration[:int(len(data_scalar_acceleration) * 0.01)])
        result_dict[Feature.FEAT_stddev1_scalar_acceleration] = np.std(data_scalar_acceleration[:int(len(data_scalar_acceleration) * 0.01)])
        result_dict[Feature.FEAT_mean1_acceleration_x] = np.mean(data_acceleration_x[:int(len(data_acceleration_x) * 0.01)])
        result_dict[Feature.FEAT_stddev1_acceleration_x] = np.std(data_acceleration_x[:int(len(data_acceleration_x) * 0.01)])
        result_dict[Feature.FEAT_mean1_acceleration_y] = np.mean(data_acceleration_y[:int(len(data_acceleration_y) * 0.01)])
        result_dict[Feature.FEAT_stddev1_acceleration_y] = np.std(data_acceleration_y[:int(len(data_acceleration_y) * 0.01)])
        result_dict[Feature.FEAT_mean1_acceleration_z] = np.mean(data_acceleration_z[:int(len(data_acceleration_z) * 0.01)])
        result_dict[Feature.FEAT_stddev1_acceleration_z] = np.std(data_acceleration_z[:int(len(data_acceleration_z) * 0.01)])
        result_dict[Feature.FEAT_mean1_gyro_x] = np.mean(data_gyro_x[:int(len(data_gyro_x) * 0.01)])
        result_dict[Feature.FEAT_stddev1_gyro_x] = np.std(data_gyro_x[:int(len(data_gyro_x) * 0.01)])
        result_dict[Feature.FEAT_mean1_gyro_y] = np.mean(data_gyro_y[:int(len(data_gyro_y) * 0.01)])
        result_dict[Feature.FEAT_stddev1_gyro_y] = np.std(data_gyro_y[:int(len(data_gyro_y) * 0.01)])
        result_dict[Feature.FEAT_mean1_gyro_z] = np.mean(data_gyro_z[:int(len(data_gyro_z) * 0.01)])
        result_dict[Feature.FEAT_stddev1_gyro_z] = np.std(data_gyro_z[:int(len(data_gyro_z) * 0.01)])

        result_dict[Feature.FEAT_mean1_delta_bearing] = np.mean(data_delta_bearing[:int(len(data_delta_bearing) * 0.01)])
        result_dict[Feature.FEAT_stddev1_delta_bearing] = np.std(data_delta_bearing[:int(len(data_delta_bearing) * 0.01)])
        result_dict[Feature.FEAT_mean1_delta_speed] = np.mean(data_delta_speed[:int(len(data_delta_speed) * 0.01)])
        result_dict[Feature.FEAT_stddev1_delta_speed] = np.std(data_delta_speed[:int(len(data_delta_speed) * 0.01)])
        result_dict[Feature.FEAT_mean1_delta_scalar_gyro] = np.mean(data_delta_scalar_gyro[:int(len(data_delta_scalar_gyro) * 0.01)])
        result_dict[Feature.FEAT_stddev1_delta_scalar_gyro] = np.std(data_delta_scalar_gyro[:int(len(data_delta_scalar_gyro) * 0.01)])
        result_dict[Feature.FEAT_mean1_delta_scalar_acceleration] = np.mean(data_delta_scalar_acceleration[:int(len(data_delta_scalar_acceleration) * 0.01)])
        result_dict[Feature.FEAT_stddev1_delta_scalar_acceleration] = np.std(data_delta_scalar_acceleration[:int(len(data_delta_scalar_acceleration) * 0.01)])
        result_dict[Feature.FEAT_mean1_delta_gyro_x] = np.mean(data_delta_gyro_x[:int(len(data_delta_gyro_x) * 0.01)])
        result_dict[Feature.FEAT_stddev1_delta_gyro_x] = np.std(data_delta_gyro_x[:int(len(data_delta_gyro_x) * 0.01)])
        result_dict[Feature.FEAT_mean1_delta_gyro_y] = np.mean(data_delta_gyro_y[:int(len(data_delta_gyro_y) * 0.01)])
        result_dict[Feature.FEAT_stddev1_delta_gyro_y] = np.std(data_delta_gyro_y[:int(len(data_delta_gyro_y) * 0.01)])
        result_dict[Feature.FEAT_mean1_delta_gyro_z] = np.mean(data_delta_gyro_z[:int(len(data_delta_gyro_z) * 0.01)])
        result_dict[Feature.FEAT_stddev1_delta_gyro_z] = np.std(data_delta_gyro_z[:int(len(data_delta_gyro_z) * 0.01)])
        result_dict[Feature.FEAT_mean1_delta_acceleration_x] = np.mean(data_delta_acceleration_x[:int(len(data_delta_acceleration_x) * 0.01)])
        result_dict[Feature.FEAT_stddev1_delta_acceleration_x] = np.std(data_delta_acceleration_x[:int(len(data_delta_acceleration_x) * 0.01)])
        result_dict[Feature.FEAT_mean1_delta_acceleration_y] = np.mean(data_delta_acceleration_y[:int(len(data_delta_acceleration_y) * 0.01)])
        result_dict[Feature.FEAT_stddev1_delta_acceleration_y] = np.std(data_delta_acceleration_y[:int(len(data_delta_acceleration_y) * 0.01)])
        result_dict[Feature.FEAT_mean1_delta_acceleration_z] = np.mean(data_delta_acceleration_z[:int(len(data_delta_acceleration_z) * 0.01)])
        result_dict[Feature.FEAT_stddev1_delta_acceleration_z] = np.std(data_delta_acceleration_z[:int(len(data_delta_acceleration_z) * 0.01)])

        return result_dict

    # MAIN METHOD

    @classmethod
    def run(cls, dataframe):
        extended_dataframe = cls.expand(dataframe)
        return cls.extract(extended_dataframe)
