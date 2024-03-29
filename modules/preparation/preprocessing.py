import os
import numpy as np
import pandas as pd
from tqdm import tqdm

from .. import common


class Preprocessing:

    @classmethod
    def join(cls, inputs=None, filter_by=None, filter_whitelist=None):

        # parameter preprocessing
        inputs = common.Utility.make_iterable(inputs)
        filter_whitelist = common.Utility.make_iterable(inputs)

        df_result = pd.DataFrame()
        for input_unit in tqdm(inputs):
            if isinstance(input_unit, pd.DataFrame):
                df_filtered = input_unit
            else:
                df_filtered = pd.read_csv(input_unit)
            if filter_by:
                df_filtered = df_filtered[df_filtered[filter_by].isin(filter_whitelist)]
            df_result = df_result.append(df_filtered, ignore_index=True)
        return df_result

    @classmethod
    def sort(cls, dataframe, sort_by, ascending=True):
        return dataframe.sort_values(by=sort_by, ascending=ascending)

    @classmethod
    def get_unique_values(cls, inputs, column_name):
        unique_values = []
        for input_unit in tqdm(inputs):
            if isinstance(input_unit, pd.DataFrame):
                df_part = input_unit
            else:
                df_part = pd.read_csv(input_unit)
            unique_values.append(df_part[column_name].unique())
        unique_values = np.array(unique_values)
        return np.unique(unique_values.flatten(), axis=0)

    @classmethod
    def partition(cls, inputs=None, output_dir=None, n=0, filter_by=None, sort_by=None, base_name='partition'):

        # parameter preprocessing
        inputs = common.Utility.make_iterable(inputs)
        if not isinstance(sort_by, list) and sort_by:
            sort_by = [sort_by]

        # one-part dataframe result
        if not output_dir or n < 1:
            df_result = cls.join(inputs)
            if sort_by:
                return cls.sort(df_result, sort_by)
            return df_result

        # one-part no-filter file result
        if not filter_by or n == 1:
            filename = base_name + '_1p_0.csv'
            print("Writing partition result to", filename)
            df_result = cls.join(inputs)
            if sort_by:
                df_result = cls.sort(df_result, sort_by)
            df_result.to_csv(os.path.join(output_dir, filename), index=False)
            return

        # identifying unique id values (divide_by)
        print("Identifying unique IDs")
        unique_ids = cls.get_unique_values(inputs, filter_by)

        # filtered file result
        for i in range(n):
            filename = base_name + '_' + str(n) + 'p_' + str(i) + '.csv'
            print("Writing partition result to", filename, '('+str(i+1)+'/'+str(n)+')')
            filter_whitelist = unique_ids[len(unique_ids) * i // n: len(unique_ids) * (i + 1) // n]
            df_result = cls.join(inputs=inputs, filter_by=filter_by, filter_whitelist=filter_whitelist)
            if sort_by:
                df_result = cls.sort(df_result, sort_by)
            df_result.to_csv(os.path.join(output_dir, filename), index=False)

    @classmethod
    def run(cls, inputs=None, output_dir=None, n=0, base_name='partition'):
        return cls.partition(inputs=inputs,
                             output_dir=output_dir,
                             n=n,
                             filter_by=common.Feature.FEAT_booking_id,
                             sort_by=[common.Feature.FEAT_booking_id, common.Feature.FEAT_second],
                             base_name=base_name)
