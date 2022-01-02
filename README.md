# Anomaly-Detection-Utility

## Instructions for reading raw CSV into Top-3-format CSV  
  
  
#### Load up time series utility library
`from tsutility import TimeSeriesDf`  
  
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