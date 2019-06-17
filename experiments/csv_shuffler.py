import sys
import pandas as pd
import numpy as np

if __name__ == "__main__":
    input_path = sys.argv[1]
    output_path = sys.argv[2]

    data = pd.read_csv(input_path)
    data = data.reindex(np.random.permutation(data.index))
    data.to_csv(output_path, index=False)
