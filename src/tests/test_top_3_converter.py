from tsutility import TimeSeriesDf  

csvpath = ""

tsdf = TimeSeriesDf(
        csvpath,
        granularity_ts='%Y-%m-%d %H:%M',
        timestamp_column_name='client_received_start_timestamp',
        delimiter='|'
    )

# OR use following line for folder
# tsdf = TimeSeriesDf(csvfolder, granularity_ts='%Y-%m-%d %H:%M', timestamp_column_name='client_received_start_timestamp', delimiter='|', folder='yes')

tsdf.scale('request_size')
tsdf.set_missing_minutes_to_zero()
tsdf.impute_zero_to_mean('request_size')
tsdf.dataframe_max_transformer("apiproxy")

tsdf.dataFrame.to_csv(csvpath)