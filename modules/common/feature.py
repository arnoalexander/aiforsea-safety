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

    TARGET = 'label'

    # 1.2. engineered (expansion)
    FEAT_deltasec = 'deltasec'
    FEAT_deltasec_bearing = 'deltasec_bearing'
    FEAT_deltasec_speed = 'deltasec_speed'

    FEAT_delta_bearing = 'delta_bearing'
    FEAT_delta_speed = 'delta_speed'

    # 1.3. engineered (aggregate)
    FEAT_mean_accuracy = 'mean_accuracy'
    FEAT_mean_bearing = 'mean_bearing'
    FEAT_mean_acceleration_x = 'mean_acceleration_x'
    FEAT_mean_acceleration_y = 'mean_acceleration_y'
    FEAT_mean_acceleration_z = 'mean_acceleration_z'
    FEAT_mean_gyro_x = 'mean_gyro_x'
    FEAT_mean_gyro_y = 'mean_gyro_y'
    FEAT_mean_gyro_z = 'mean_gyro_z'
    FEAT_mean_speed = 'mean_speed'
    # TODO add engineered/extracted feature

    # 1.4. utility
    UTIL_connect = '_'

    @classmethod
    def feat_at(cls, feat, at):
        return feat + cls.UTIL_connect + str(at)
