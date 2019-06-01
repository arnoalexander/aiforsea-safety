class Feature:

    # 1.1. original
    FEAT_booking_id = 'bookingID'
    FEAT_accuracy = 'Accuracy'
    FEAT_bearing = 'Bearing'
    FEAT_acceleration_x = 'acceleration_x'
    FEAT_acceleration_y = 'acceleration_y'
    FEAT_acceleration_z = 'acceleration_z'
    FEAT_gyro_x = 'gyro_x'
    FEAT_gyro_y = 'gyro_y'
    FEAT_gyro_z = 'gyro_z'
    FEAT_second = 'second'
    FEAT_speed = 'Speed'

    # 1.2. engineered
    PREFIX_mean = 'mean'
    # TODO add engineered/extracted feature

    # 1.3 utility
    UTIL_connect = '_'

    @classmethod
    def feat_at(cls, feat, at):
        return feat + cls.UTIL_connect + str(at)

    @classmethod
    def feat_prefix(cls, feat, prefix):
        return prefix + cls.UTIL_connect + feat
