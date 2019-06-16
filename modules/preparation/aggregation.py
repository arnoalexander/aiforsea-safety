import pandas as pd
from tqdm import tqdm

from .feature_extraction import FeatureExtraction
from ..common import Feature


class Aggregation:

    @classmethod
    def aggregate(cls, dataframe, by=Feature.FEAT_booking_id):
        group_df = dataframe.groupby(by)
        list_df = []
        for key, df_part in tqdm(group_df):
            list_df.append(FeatureExtraction.run(df_part))
        return pd.DataFrame(list_df)

    @classmethod
    def run(cls, inputs=None):

        # parameter preprocessing
        if inputs is None:
            inputs = []
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
            df_part = cls.aggregate(df_part, Feature.FEAT_booking_id)
            df_result = df_result.append(df_part, ignore_index=True)

        return df_result
