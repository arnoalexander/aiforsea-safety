import os
import numpy as np
import pandas as pd
from tqdm import tqdm


def partition(inputs=[], output_dir=None, n=0, divide_by=None, sort_by=None, base_name='partition'):

    # parameter preprocessing
    if not isinstance(inputs, list):
        inputs = [inputs]
    if not isinstance(sort_by, list) and sort_by:
        sort_by = [sort_by]

    # helper filter function
    def _partition_filter(is_filter=False, divide_by_whitelist=[]):
        df = pd.DataFrame()
        for input_unit in tqdm(inputs):
            if isinstance(input_unit, pd.DataFrame):
                df_filtered = input_unit
            else:
                df_filtered = pd.read_csv(input_unit)
            if is_filter:
                df_filtered = df_filtered[df_filtered[divide_by].isin(divide_by_whitelist)]
            df = df.append(df_filtered, ignore_index=True)
        if sort_by:
            df.sort_values(by=sort_by, ascending=True, inplace=True)
        return df

    # dataframe result
    if not output_dir or n < 1:
        return _partition_filter()

    # no-filter file result
    if not divide_by or n == 1:
        filename = base_name + '_1p_0.csv'
        print("Writing partition result to", filename)
        df_result = _partition_filter()
        df_result.to_csv(os.path.join(output_dir, filename), index=False)
        return

    # identifying unique id values (divide_by)
    print("Identifying unique IDs")
    unique_ids = []
    for input_unit in tqdm(inputs):
        if isinstance(input_unit, pd.DataFrame):
            df_part = input_unit
        else:
            df_part = pd.read_csv(input_unit)
        unique_ids.append(df_part[divide_by].unique())
    unique_ids = np.array(unique_ids)
    unique_ids = np.unique(unique_ids.flatten(), axis=0)

    # filtered file result
    for i in range(n):
        filename = base_name + '_' + str(n) + 'p_' + str(i) + '.csv'
        print("Writing partition result to", filename, '('+str(i+1)+'/'+str(n)+')')
        divide_by_whitelist = unique_ids[len(unique_ids) * i // n: len(unique_ids) * (i + 1) // n]
        df_result = _partition_filter(is_filter=True, divide_by_whitelist=divide_by_whitelist)
        df_result.to_csv(os.path.join(output_dir, filename), index=False)


def aggregate(inputs=[], base_name='aggregation'):

    # parameter preprocessing
    if not isinstance(inputs, list):
        inputs = [inputs]

    # TODO implement aggregation by bookingID
    return
