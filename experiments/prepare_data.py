import sys
import os

sys.path.append('..')
import definitions
from modules import preparation, common

if __name__ == "__main__":
    divide_by = common.FEAT_booking_id
    sort_by = [common.FEAT_booking_id, common.FEAT_second]
    base_filename = "transformed_train"
    if len(sys.argv) > 1:
        num_partition = int(sys.argv[1])
    else:
        num_partition = 20

    print('Partitioning training data')

    inputs = [os.path.join(definitions.DATA_ORIGIN, file) for file in os.listdir(definitions.DATA_ORIGIN)]
    preparation.run_partition(inputs=inputs, output_dir=definitions.DATA_PART, n=num_partition, base_name=base_filename)
        
    print("Data partition finished")
