import sys
import os

sys.path.append('..')
import definitions
from modules.utilities import Utilities

if __name__ == "__main__":
    divide_by = 'bookingID'
    sort_by = ['bookingID', 'second']
    base_filename = "transformed_train"
    if len(sys.argv) > 1:
        num_partition = int(sys.argv[1])
    else:
        num_partition = 20

    print('Partitioning training data')

    util = Utilities()
    inputs = [os.path.join(definitions.DATA_ORIGIN, file) for file in os.listdir(definitions.DATA_ORIGIN)]
    util.partition(inputs=inputs, output=definitions.DATA_PART, n=num_partition, divide_by=divide_by, sort_by=sort_by, base_name=base_filename)
        
    print("Finished")
