<link href="misc/style.css" rel="stylesheet"></link>

# AI for SEA - Safety Challenge

![AI for SEA Safety Challenge](misc/Safety.png#center)

> Grab has been proactively pushing to make transportation in SEA safer. 
As part of the effort, we want to identify dangerous drivings in a timely manner.

This model will predict dangerous trips based on telematics data.

## How to Make Prediction?

Run following commands.

```sh
cd main
python <input_path> <output_path>
```

`<input_path>` 
is path to a test file in csv format (single file) 
**OR** 
path to a directory containing test files in csv format (may contains multiple file).  

`<output_path>`
is path for output csv file containing prediction result.  

Example:

```sh
cd main
python test.csv result.csv
```

## Behind the Scene

Are you interested about how the model was made? 
If yes, please do these steps to re-run the experiment process.  

1. Prepare the data directory

   Inside this project, create folders with following hierarchy.
   
   ```
    .
    └── data
        ├── origin
        ├── label
        ├── prep
        └── aggregated
   ```
   
   Fill `data/origin` with train files in csv format and `data/label` with label file in csv format.
   The data can be downloaded [here](https://s3-ap-southeast-1.amazonaws.com/grab-aiforsea-dataset/safety.zip).
   Let the remaining directory empty.
   
2. Generate intermediate data

   The data will undergo 2 processes : preprocessing and aggregation.
   In preprocessing, data will be sorted by booking ID to ease the aggregation process, partitioned into several parts to accommodate machine with low RAM capacity, and then stored in `data/prep`.
   Then, in aggregation, rows with similar booking ID will be aggregated into one new row and stored in `data/aggregated`.
   
   To do the processes mentioned above, run these commands.  
   
   ```sh
   cd experiments
   python data_preprocessing.py
   python data_aggregation.py
   ```
   
3. Run the jupyter notebook

   `experiments/Experiment.ipynb` is the notebook containing the experiments
