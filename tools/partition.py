import sys, os
import pandas as pd
from tqdm import tqdm

sys.path.append('..')
import definitions

if __name__ == "__main__":
    filter_by = 'bookingID'
    sort_by = 'second'
    base_filename = "transformed_train"
    if (len(sys.argv) > 1):
        num_partition = int(sys.argv[1])
    else:
        num_partition = 15
    
    labels = pd.read_csv(os.path.join(definitions.DATA_LABEL, os.listdir(definitions.DATA_LABEL)[0]))
    
    for i in range(num_partition):
        filename = base_filename + "_" + str(num_partition) + "p_" + str(i) + ".csv"
        print("Partitioning label", str(len(labels)*i//num_partition), "-", str(len(labels)*(i+1)//num_partition - 1), "to", filename)
        
        partition_whitelist = list(labels.iloc[len(labels)*i//num_partition : len(labels)*(i+1)//num_partition][filter_by])
        
        df = pd.DataFrame()
        for file in tqdm(os.listdir(definitions.DATA_ORIGIN)):
            path = os.path.join(definitions.DATA_ORIGIN, file)
            df_filtered = pd.read_csv(path)
            df_filtered = df_filtered[df_filtered[filter_by].isin(partition_whitelist)]
            df = df.append(df_filtered, ignore_index = True)
            
        df.sort_values(by=[filter_by, sort_by], ascending=True, inplace=True)
        df.to_csv(os.path.join(definitions.DATA_PART, filename), index=False)
        
    print("Finished")
