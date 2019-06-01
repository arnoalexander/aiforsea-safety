import pandas as pd
import numpy as np
from tqdm import tqdm

from .feature_extraction import FeatureExtraction
from .. import common


class Aggregation:

    @classmethod
    def aggregate(cls, dataframe, by):
        group_df = dataframe.groupby(by)
        list_df = []
        for key, df_part in tqdm(group_df):
            list_df.append(FeatureExtraction.run(df_part))
        list_df = np.array(list_df)
        return pd.DataFrame(data=list_df, columns=FeatureExtraction.EXTRACTED)

    @classmethod
    def run(cls, inputs=[]):

        # parameter preprocessing
        if not isinstance(inputs, list):
            inputs = [inputs]

        # aggregation by id
        df_result = pd.DataFrame()
        for index, input_unit in enumerate(inputs):
            print("Aggregating input (" + str(index+1) + "/" + str(len(inputs)) + ")")
            if isinstance(input_unit, pd.DataFrame):
                df_part = input_unit
            else:
                df_part = pd.read_csv(input_unit)
            df_part = cls.aggregate(df_part, common.Feature.FEAT_booking_id)
            df_result = df_result.append(df_part, ignore_index=True)
        return df_result
