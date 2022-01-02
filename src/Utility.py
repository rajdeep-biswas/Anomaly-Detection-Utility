import pandas as pd
import numpy as np

from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler

import os
import datetime

from exceptions.tsexceptions import CSVPathError, IncorrectParameterError, TimestampColumnNotSetError

class TimeSeriesDf():

    dataFrame = pd.DataFrame()
    timestamp_column_name = None
    X_slides = []
    y_slides = []

    scaler_dict = {
        'standard': StandardScaler,
        'minmax': MinMaxScaler,
        'maxabs': MaxAbsScaler
    }

    scaler = None

    def __init__(self,
                csv_path = None,
                folder = 'no',
                granularity_ts = None,
                timestamp_column_name = None,
                delimiter = ','):

        '''
        Constructor for TimeSeriesDf class. Loads up a single CSV file or a folder containing CSV files and automatically appends them based alphanumeric filename.
        Arguments -
        csv_path: The source CSV file path.
        folder: 'yes' can be specified if you choose to read an entire folder containing CSV files (named alphanumerically in terms of timestamps). 'no' for a single CSV file.
        granularity_ts: The format of the timestamp present in the CSV has to be presented as a string here. For example: '%Y-%m-%d %H:%M'. It can be made more granular by removing smaller parts of the timestamp such as seconds or milliseconds.
        timestamp_column_name: The header / title of the column that contains the timestamp in the CSV file(s).
        delimiter: Defaults to commma (,) in case of true CSV files, variations may include pipe (|), etc.
        '''

        if not csv_path:
            raise CSVPathError("No csv_path provided.") 

        if timestamp_column_name and not granularity_ts:
            raise IncorrectParameterError("If timestamp_column_name is provided, then granularity_ts needs to be provided with desired granularity and in accordance with format that is contained in CSV.")

        if folder == 'yes':

            print("Reading CSV files from folder...")

            all_files = sorted(os.listdir(csv_path))
            csv_dict = {}
            csv_files = []

            for a_file in all_files:
                if a_file[-3:] == 'csv':
                    csv_files.append(a_file)

            for csv_file in tqdm(csv_files):
                if csv_file[-3:] == 'csv':
                    csv_dict[csv_file] = pd.read_csv(os.path.join(csv_path, csv_file), delimiter=delimiter)

            self.dataFrame = csv_dict[csv_files[0]].append([csv_dict[csv_file] for csv_file in csv_files[1:]])

        elif folder == 'no':

            print("Reading CSV file.")

            self.dataFrame = pd.read_csv(csv_path, delimiter=delimiter)

        else:
            raise IncorrectParameterError("Provided parameter value for folder should be 'yes' or 'no' (default is 'no').")

        print("Done reading CSV file(s).")

        if timestamp_column_name:
            self.timestamp_column_name = timestamp_column_name
            self.convert_timestamp_column(granularity_ts, timestamp_column_name)

    
    def convert_timestamp_column(self, granularity_ts, timestamp_column_name = None):

        '''
        Method that converts the timestamp string in the DataFrame into pandas datetime instance.
        Arguments -
        granularity_ts: The format of the timestamp present in the CSV has to be presented as a string here. For example: '%Y-%m-%d %H:%M'. It can be made more granular by removing smaller parts of the timestamp such as seconds or milliseconds.
        timestamp_column_name: The header / title of the column that contains the timestamp in the CSV file(s).
        '''

        print("Converting timestamp column.")

        if not self.timestamp_column_name \
            and not timestamp_column_name:

            raise IncorrectParameterError("timestamp_column_name has not been set during initialization. Please call convert_timestamp_column() with both granularity_ts and timestamp_column_name parameters.")
        
        if not granularity_ts:
            raise IncorrectParameterError("granularity_ts is a required parameter.")

        self.dataFrame[timestamp_column_name] = pd.to_datetime(pd.to_datetime(self.dataFrame[timestamp_column_name]).dt.strftime(granularity_ts))

        if not self.timestamp_column_name:
            self.timestamp_column_name = timestamp_column_name


    def check_missing_timestamps(self, min_leniency = 1):

        '''
        Method that shows you durations of timestamps that are missing in the dataframe.
        Arguments -
        min_leniency: Lower limit from which continuous timestamps are to be considered "missing".
        '''

        if min_leniency < 0 \
            or min_leniency > 59:

            raise IncorrectParameterError("min_leniency has to be within the range of 0 minutes and 59 minutes.")
            return

        if min_leniency < 10:
            min_leniency = '0' + str(min_leniency)

        else:
            min_leniency = str(min_leniency)

        if not self.timestamp_column_name:
            raise TimestampColumnNotSetError("timestamp_column_name has to be set before you can use this feature.")
            return
        
        print("Timestamp X has missing data for Y minutes -")
        nothingFoundFlag = True
            
        for i in range(1, len(self.dataFrame)):
                
            if str(
                self.dataFrame[self.timestamp_column_name][i] - \
                self.dataFrame[self.timestamp_column_name][i - 1])[7:] \
                > "00:" + min_leniency + ":00":

                print(
                    self.dataFrame[self.timestamp_column_name][i - 1],
                    "for",
                    str(self.dataFrame[self.timestamp_column_name][i] - self.dataFrame[self.timestamp_column_name][i - 1])[7:]
                )

                nothingFoundFlag = False

        if nothingFoundFlag:
            print("No consecutive timestamps had a gap greater than 1 minute.")


    def sliding_windows(self, data, seq_length = 15):

        '''
        Method that converts linear temporal data into sliding windows.
        Arguments -
        data: Linear list of one feature at a time that serves as input.
        seq_length: The sequence length that is used as the sliding window size.
        '''

        print("Generating sliding windows.")

        x = []
        y = []

        for i in range(len(data) - seq_length):
            _x = data[i : i + seq_length]
            _y = data[i + seq_length]
            x.append(_x)
            y.append(_y)

        return np.array(x),np.array(y)

    
    def create_sliding_windows(self, data_column_name = None, window_size = 15):

        '''
        Method that uses sliding_windows() method to generate and store sliding window training data as data members.
        Arguments -
        data_column_name: The header / title of the column that contains the timestamp in the dataframe.
        window_size: The sequence size that is used as the sliding window length.
        '''

        if not data_column_name:
            raise IncorrectParameterError("data_column_name needs to be passed in to create sliding windows.")
            return

        self.X_slides, self.y_slides = self.sliding_windows(self.dataFrame[data_column_name], seq_length = window_size)


    def train_test_split(self, start_month, train_month_count):
        # TODO once month start is done
        # This can be refactored into SupervisedModel class's split_by_month_index method.
        return


    def scale(self, columns = None, scaler_type = 'minmax'):

        '''
        Method that uses a scaler to normalize the three callcount features.
        Argument -
        columns: Columns of the dataframe over which the scaler should be fit and transformed.
        scaler: Default usage is Sklearn's MinMaxScaler builtin. Can be replaced with any other Sklearn scaler of user choice; can be specified using strings as follows.
            'minmax', 'standard' or 'maxabs' (defaults to 'minmax').
        '''

        print("Scaling dataset.")

        if not columns:
            raise IncorrectParameterError("columns were not provided.")
            return

        if type(columns) is str:
            columns = [columns]

        # TODO assert in a better way that column is a 1D list of strings and not any higher.
        if type(columns[0]) is not str:
            raise IncorrectParameterError("columns parameter has to be a 1D list of strings.")
            return

        if scaler_type not in ['minmax',
                          'standard',
                          'maxabs']:
            
            raise IncorrectParameterError("scaler_type can be 'minmax', 'standard' or 'maxabs' (default: 'minmax').")
            return

        Scaler = self.scaler_dict[scaler_type]
        scaler = Scaler()
        scaler.fit(self.dataFrame[columns])
        self.dataFrame[columns] = scaler.transform(self.dataFrame[columns])

        self.scaler = scaler
        return scaler


    # TODO leaving out the part for calculating mae, anomaly truth values, etc within the dataframe, for now.


    def set_missing_minutes_to_zero(self):

        '''
        Method that imputes the missing timestamps by setting every missing minute to a 0 (zero) callcount.
        '''

        print("Imputing missing timestamps to zero.")

        if not self.timestamp_column_name:
            raise TimestampColumnNotSetError("timestamp_column_name has to be set before you can use this feature.")
            return

        self.dataFrame = self.dataFrame.set_index(self.timestamp_column_name)
        first_ts = self.dataFrame.index[0]
        last_ts = self.dataFrame.index[-1]
        incr_timestamp = self.dataFrame.iloc[0].name

        for i in tqdm(range(int((last_ts - first_ts).total_seconds() // 60))):
            try:
                self.dataFrame.loc[incr_timestamp]
            except:
                self.dataFrame.loc[incr_timestamp] = 0
            incr_timestamp += datetime.timedelta(minutes = 1)

        self.dataFrame = self.dataFrame.reset_index()


    def impute_zero_to_mean(self, data_column_name = None, mean_of_last = 15):

        '''
        Method that imputes timestamps with 0 (zero) callcounts to the mean of last n values. (Default n: 15)
        Arguments -
        data_column_name: The header / title of the column that contains the timestamp in the dataframe.
        mean_of_last: 'n' as explained in this stringdoc.
        '''

        print("Imputing zero valued timestamps to mean of last " + str(mean_of_last) + " minutes.")

        if mean_of_last < 1:
            raise TimestampColumnNotSetError("Last n values cannot be less than 1.")

        if not data_column_name:
            raise TimestampColumnNotSetError("data_column_name is a required parameter.")

        for i in tqdm(range(mean_of_last, len(self.dataFrame))):
            if self.dataFrame[self.timestamp_column_name].values[i] == 0:
                self.dataFrame.at[i, data_column_name] = round(sum([calls for calls in self.dataFrame[data_column_name].values[i-mean_of_last:i]]) / mean_of_last)

    new_df = pd.DataFrame(columns = ['total', 'max1', 'max2'])


    def flatten_and_sort(self, array):

        '''
        Method used to flatten a 2D univariate array into a single 1D array.
        (Quick and dirty replacement for numpy's reshape).
        Argument -
        array: To pass in the 2D list as input.
        '''

        flattened_array = []
        for item in array:
            flattened_array.append(item[0])
        flattened_array.sort()
        return flattened_array


    def dataframe_max_transformer(self, apiproxy_column_name='apiproxy'):

        '''
        Converts a regular DataFrame into the Total, Max1, Max2 Callcounts as derived feature columns.
        Note: This replaces the default member DataFrame with the trivariate one; removing the additional (and now-meaningless) columns.
        Argument -
        apiproxy_column_name: Column header / title that contains the API Proxy names in the dataframe.
        '''

        print("Converting dataframe into three-feature columns. This can take a while, below is a progress bar!")
        
        df = self.dataFrame
        df = df.groupby(by=[self.timestamp_column_name, apiproxy_column_name], axis=0).count()
        
        new_df = pd.DataFrame(columns = ['total', 'max1', 'max2'])

        for index in tqdm(df.index.values.tolist()):
            callcounts = self.flatten_and_sort(df.loc[index[0]].values)
            new_df.loc[index[0], 'total'] = sum(callcounts)
            new_df.loc[index[0], 'max1'] = callcounts[-1]
            new_df.loc[index[0], 'max2'] = callcounts[-2] if len(callcounts) > 1 else 0

        self.dataFrame =  pd.DataFrame.from_dict({
            "timestamps": [str(val) for val in new_df.index],
            "count_total": new_df.total.values.tolist(),
            "count_max1": new_df.max1.values.tolist(),
            "count_max2": new_df.max2.values.tolist()
        })