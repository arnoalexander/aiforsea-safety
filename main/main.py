import sys
import os
import pickle
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

sys.path.append('..')
import definitions
from modules.preparation import Preprocessing, Aggregation

if __name__ == "__main__":

    # arg processing
    input_path = sys.argv[1]
    output_path = sys.argv[2]

    # input preprocessing
    if input_path.endswith(".csv"):
        print("Loading file", input_path, "...")
        dataframe = Preprocessing.run(input_path)
    else:
        print("Loading directory content from", input_path, "...")
        inputs = [os.path.join(input_path, file) for file in os.listdir(input_path)]
        dataframe = Preprocessing.run(inputs)
    print("")

    # aggregation
    print("Aggregating data ...")
    dataframe = Aggregation.run(dataframe)
    id = dataframe['bookingID'].values.copy()
    dataframe = dataframe.drop(['bookingID'], axis=1)
    dataframe = dataframe.reindex(sorted(dataframe.columns), axis=1)
    dataframe = dataframe.values
    print("")

    # predicting
    print("Predicting. Writing file to", output_path, "...")
    model = pickle.load(open(definitions.MODEL_FINAL, 'rb'))
    pred = model.predict(dataframe)
    result = pd.DataFrame({
        'bookingID': id,
        'label': pred
    })
    result.to_csv(output_path, index=False)
    print("")

    # finish
    print("DONE")
