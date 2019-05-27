import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append('..')
import definitions


class Utilities:

    def partition(self, inputs=None, output=None, n=0, divide_by=None, sort_by=None, base_name='partition'):

        # parameter preprocessing
        if not isinstance(inputs, list):
            if inputs:
                inputs = [inputs]
            else:
                inputs = []
        if not isinstance(sort_by, list) and sort_by:
            sort_by = [sort_by]

        # helper filter function
        def partition_filter(is_filter=False, divide_by_whitelist=[]):
            df = pd.DataFrame()
            for path in tqdm(inputs):
                df_filtered = pd.read_csv(path)
                if is_filter:
                    df_filtered = df_filtered[df_filtered[divide_by].isin(divide_by_whitelist)]
                df = df.append(df_filtered, ignore_index=True)
            if sort_by:
                df.sort_values(by=sort_by, ascending=True, inplace=True)
            return df

        # dataframe result
        if not output or n < 1:
            return partition_filter()

        # no-filter file result
        if not divide_by or n == 1:
            filename = base_name + '_1p_0.csv'
            print("Writing partition result to", filename)
            df_result = partition_filter()
            df_result.to_csv(os.path.join(output, filename), index=False)
            return

        # identifying unique id values (divide_by)
        print("Identifying unique IDs")
        unique_ids = []
        for path in tqdm(inputs):
            df_part = pd.read_csv(path)
            unique_ids.append(df_part[divide_by].unique())
        unique_ids = np.array(unique_ids)
        unique_ids = np.unique(unique_ids.flatten(), axis=0)

        # filtered file result
        for i in range(n):
            filename = base_name + '_' + str(n) + 'p_' + str(i) + '.csv'
            print("Writing partition result to", filename, '('+str(i+1)+'/'+str(n)+')')
            divide_by_whitelist = unique_ids[len(unique_ids) * i // n: len(unique_ids) * (i + 1) // n]
            df_result = partition_filter(is_filter=True, divide_by_whitelist=divide_by_whitelist)
            df_result.to_csv(os.path.join(output, filename), index=False)


if __name__ == "__main__":
    util = Utilities()
    inputs = [os.path.join(definitions.DATA_ORIGIN, file) for file in os.listdir(definitions.DATA_ORIGIN)]
    util.partition(inputs=inputs, output=definitions.DATA_PART, n=20, divide_by='bookingID', sort_by=['bookingID', 'second'])