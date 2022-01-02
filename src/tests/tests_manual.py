from tsutility import TimeSeriesDf

csvpath = '../../../../../../Data/API-HUB-DATA/21-01-smaller.csv'
csvfolder = '../../../../../../Data/API-HUB-DATA/'

tsdf = TimeSeriesDf(csvpath)
tsdf.convert_timestamp_column('%Y-%m-%d %H:%M') # tscol not set during init, both params are required now
tsdf.convert_timestamp_column('%Y-%m-%d %H:%M', 'client_received_start_timestamp')
tsdf.dataFrame
tsdf.dataFrame['client_received_start_timestamp']
tsdf.convert_timestamp_column('client_received_start_timestamp', '%Y-%m-%d %H:%M')
tsdf = TimeSeriesDf(csvpath, 'yes', 'col') # tscol without gran_ts error
tsdf = TimeSeriesDf(csvpath, 'duh', 'col') # folder parameter error
tsdf = TimeSeriesDf('yes', 'col') # should be non existing folder error but give folder parameter error so okay
tsdf = TimeSeriesDf(csvpath)
tsdf = TimeSeriesDf(csvpath, granularity_ts='%Y-%m-%d %H:%M', timestamp_column_name='client_received_start_timestamp', delimiter='|')
tsdf.create_sliding_windows('request_size')
tsdf.X_slides
tsdf.y_slides
tsdf.scale('request_size')
tsdf = TimeSeriesDf(csvfolder, granularity_ts='%Y-%m-%d %H:%M', timestamp_column_name='client_received_start_timestamp', delimiter='|', folder='yes')
tsdf.set_missing_minutes_to_zero()
tsdf.impute_zero_to_mean('request_size')
tsdf.dataframe_max_transformer("apiproxy")
