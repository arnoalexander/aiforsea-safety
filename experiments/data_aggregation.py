import sys
import os

sys.path.append('..')
import definitions
from modules.preparation import Aggregation


if __name__ == "__main__":
    filename = "aggregated_train.csv"

    print('Aggregating training data to', filename)

    inputs = [os.path.join(definitions.DATA_PREP, file) for file in os.listdir(definitions.DATA_PREP)]
    df_result = Aggregation.run(inputs)
    df_result.to_csv(os.path.join(definitions.DATA_AGGREGATED, filename), index=False)
        
    print("Data aggregation finished")
