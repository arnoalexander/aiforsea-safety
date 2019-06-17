# Experiment Documentation

## Feature Engineering

In this experiment, there are 2 big step of feature engineering

1. Expansion
   
   These new feature groups are generated into original data.
   
   * *Validity*. There are some implicit missing values, e.g. speed = -1 while it is impossible for speed to be less than zero.
   Validity features will mark a row with 0 if it has missing value, otherwise 1.
   
   * *Differential*. Dangerous driving are involving some sudden movement, e.g. sudden acceleration, sudden braking, sudden turning, etc.
   These behaviour would be detected in sensor as high change in sensor reading over short period of time.
   To represent them, features about change of reading per time duration are added.
   
2. Aggregation

   Initially, there are many row of data for single booking ID.
   To make it ready for training process, rows with similar booking ID are aggregated into one row by extracting these feature groups.
   
   * *Mean* : Mean of top 100%, 20%, 5%, and 1% value from expanded data.
   It will represent data from all perspective, from general trip statistics to specific extreme event in a trip.
   
   * *Standard Deviation* : Standard deviation of top 100%, 20%, 5%, and 1% value from expanded data.
   It will represent data from all perspective, from general trip statistics to specific extreme event in a trip.

## Model Construction and Evaluation

The model used is gradient boosting (LightGBM) and evaluated using Stratified K-Fold.
