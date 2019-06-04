import sys
import os
import json

sys.path.append('..')
import definitions
from modules import preparation


if __name__ == "__main__":
    filename = "aggregated_train.csv"

    print('Aggregating training data to', filename)

    inputs = [os.path.join(definitions.DATA_PREP, file) for file in os.listdir(definitions.DATA_PREP)]
    features = json.load(open(definitions.DATA_FEATURES, 'r'))
    df_result = preparation.Aggregation.run(inputs, features)
    df_result.to_csv(os.path.join(definitions.DATA_AGGREGATED, filename), index=False)
        
    print("Data aggregation finished")
