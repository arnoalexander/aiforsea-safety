import sys
import os

sys.path.append('..')
import definitions
from modules import preparation


if __name__ == "__main__":
    filename = "aggregated_train.csv"

    print('Aggregating training data')

    inputs = [os.path.join(definitions.DATA_PART, file) for file in os.listdir(definitions.DATA_PART)]
    df_result = preparation.Aggregation.run(inputs)
    df_result.to_csv(os.path.join(definitions.DATA_AGGREGATED, filename), index=False)
        
    print("Data aggregation finished")
