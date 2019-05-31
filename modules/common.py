# 1. FEATURES

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

# TODO add engineered/extracted feature

# 1.3 utility

FEAT_UTIL_at = '_'

def feat_at(feat, at):
    return feat + FEAT_UTIL_at + str(at)

