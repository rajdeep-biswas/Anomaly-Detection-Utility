from tsutility import TimeSeriesDf
from ad import SupervisedModel

csvinpath = '../../../../../../Data/API-HUB-DATA/21-01-smaller.csv'
csvoutpath = '../../../../../../Data/API-HUB-DATA/21-01-maxed.csv'
labelspath = '../../../../../../Data/LABELS1.csv'
saved_model_path = '../../../../../../Models/trialmodel'

tsdf = TimeSeriesDf(csvinpath, granularity_ts='%Y-%m-%d %H:%M', timestamp_column_name='client_received_start_timestamp', delimiter='|')
df = tsdf.dataFrame
df = df.groupby(by=['client_received_start_timestamp', 'apiproxy'], axis=0).count()
tsdf.dataframe_max_transformer("apiproxy")
tsdf.dataFrame.to_csv(csvoutpath)

adsm = SupervisedModel()
adsm.import_data(csvoutpath, csvoutpath, labelspath)
adsm.normalize_features()
adsm.init_month_starts()
adsm.split_by_month_index()
adsm.create_and_train_model(epochs=10)
adsm.plot_loss()
adsm.plot_accuracy()
adsm.model.save(saved_model_path)
