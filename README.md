# Anomaly Detection Utility
We were trying to perform Anomaly Detection on incoming API Traffic. It used an LSTM-Autoencoder architecture.  
Here is parts of the preprocessing, modelling and scoring utility code that I wrote for the project.

The code has functionality including but not limited to -
* Importing custom CSV files with nuanced datetime formats.
* Resampling missing timestamps by various methods e.g. setting missing values to mean of last n timestamps.
* Custom `train_test_split` alternative that lets you easily split by whole months of data instead of a decimal %.

## Project Structure
`src/SupervisedModel.py` contains code for the final preprocessing and model training.
`src/Utility.py` contains code for various dataframe operations that were relevant to the analysis and data cleaning part of the project.
`src/exceptions` contain some custom Python Exception classes.
`src/tests` contain some manual testcases.


### Instructions for reading raw CSV into Top-3-format CSV  
  
  
#### Load up time series utility library
`from Utility import TimeSeriesDf`  
  
#### Use following to load up a single CSV file
```python
tsdf = TimeSeriesDf(
        csvpath,
        granularity_ts='%Y-%m-%d %H:%M',
        timestamp_column_name='client_received_start_timestamp',
        delimiter='|
    )  
```  
#### OR
#### Use following to load up all CSVs in a folder (one additional parameter)
```python
tsdf = TimeSeriesDf(
        csvfolder,
        granularity_ts='%Y-%m-%d %H:%M',
        timestamp_column_name='client_received_start_timestamp',
        delimiter='|',
        folder='yes'
    )
```  
  
##### You can also set the custom granularity that is needed using the granularity_ts parameter above.

#### Series of commands to scale, impute zeroes to mean, and convert dataframe into Top-3-format
```python
tsdf.scale('request_size')
tsdf.set_missing_minutes_to_zero()
tsdf.impute_zero_to_mean('request_size')
tsdf.dataframe_max_transformer("apiproxy")
```

#### Can use following to save the converted CSV to a data/object store
`tsdf.dataFrame.to_csv(csvpath)`