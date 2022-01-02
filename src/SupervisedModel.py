import os
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import tqdm
import pickle
import joblib
import keras
from tqdm import tqdm
from functools import reduce
from matplotlib.pyplot import figure
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from keras.models import Sequential, Model, load_model
from keras.layers import LSTM, Input, Dropout, Dense, RepeatVector, TimeDistributed, ReLU
from tensorflow import data



class SupervisedModel():


    anomaly_removed_df = None
    anomaly_included_df = None
    labels_df = None
    final_columns = []
    anomaly_days = []
    month_splits = None
    seq_length = None
    X_train = []
    y_train = []
    X_test = []
    y_val = []
    y_test = []
    model = None
    model_history = None



    def __init__(self,
                final_columns = ['index', 'count_total', 'count_max1', 'count_max2'],
                seq_length = 15
        ):

        '''
        Constructor for SupervisedModel class.
        Arguments -
        final_columns: Named columns that are expected in the transformed dataframe that contains multivarite derived features.
        seq_length: The sequence length that is used as the sliding window size.
        '''

        self.final_columns = final_columns
        self.seq_length = seq_length


    def import_data(self,
            normalized_data_path,
            unnormalized_data_path,
            labels_path):

        '''
        Method to load up CSV files using pandas builtin and store as dataframes into data members.
        Arguments -
        normalized_data_path: CSV file that contains the three-feature columns with anomalies having been normalized.
        unnormalized_data_path: CSV file that contains the three-feature columns including anomalies.
        labels_path: CSV file that contains labels (only used for scoring).
        '''

        print("Reading CSV files...")

        self.anomaly_removed_df = pd.read_csv(normalized_data_path)
        self.anomaly_included_df = pd.read_csv(unnormalized_data_path)
        self.labels_df = pd.read_csv(labels_path)

        print("Done reading.")


    def resample_data(self,
            column_names = ['Unnamed: 0.1', 'Unnamed: 0', 'y'],
            resample_time = '1min'):

        '''
        Method to resample dataframe timestamps across the three member dataframes.
        Arguments -
        column_names: Names of columns containing the timestamps in the dataframes anomaly_removed_df, anomaly_included_df and labels_df, respectively.
        resample_time: Resolution of duration to which the timestamps should be resampled to (defaults to 1 minute).
        '''

        print("Resampling timestamps across three CSV files.")

        self.anomaly_removed_df = self.anomaly_removed_df.set_index(column_names[0])
        self.anomaly_removed_df.index = pd.to_datetime(self.anomaly_removed_df.index, format='%d/%m/%y %H:%M')
        self.anomaly_removed_df = self.anomaly_removed_df.resample(resample_time).sum()

        self.anomaly_included_df = self.anomaly_included_df.set_index(column_names[1])
        self.anomaly_included_df.index = pd.to_datetime(self.anomaly_included_df.index, format='%d/%m/%y %H:%M')
        self.anomaly_included_df = self.anomaly_included_df.resample(resample_time).sum()

        self.anomaly_days = self.labels_df[self.labels_df[column_names[2]] == 1].Timestamp
        self.anomaly_days = pd.to_datetime(pd.to_datetime(self.anomaly_days, format='%d/%m/%y %H:%M').dt.strftime('%d/%m/%y'))
        self.anomaly_days = list(set(self.anomaly_days))

        print("Done resampling.")


    def normalize_features(self, scaler=None):

        '''
        Method that uses a scaler to normalize the three callcount features.
        Argument -
        scaler: Default usage is Sklearn's MinMaxScaler builtin. Can be replaced with any other Sklearn scaler of user choice.
        '''

        print("Normalizing features.")

        if scaler:
            scaler = joblib.load("../Models/scaler.save")

        else:
            scaler = MinMaxScaler(feature_range=(0, 1))
        
        self.anomaly_removed_df[self.final_columns[1:]] = scaler.fit_transform(self.anomaly_removed_df[self.final_columns[1:]])
        self.anomaly_included_df[self.final_columns[1:]] = scaler.transform(self.anomaly_included_df[self.final_columns[1:]])


    def init_month_starts(self, column_names=['Unnamed: 0']):

        '''
        Method that calculates a DataFrame that can help with determining indices for exact validation splitting for exact months.
        Argument -
        column_names: Takes in the column name that contains the timestamp in the source dataframe.
        '''

        self.anomaly_removed_df[[column_names[0]]] = pd.to_datetime(self.anomaly_removed_df[column_names[0]])
        temp_df = self.anomaly_removed_df.set_index(column_names[0])
        self.month_splits = temp_df.resample('MS').asfreq()


    def show_month_starts(self):

        '''
        Method that simply displays the dataframe for determining month starts (and ends).
        '''

        print(self.month_splits)


    def sliding_windows(self, data, seq_length):

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


    def split_by_month_index(self, start_month = 0, end_month = -1):

        '''
        # TODO
        This method, as of yet, DOES NOT split up the data into different training and test sets.
        Expected behaviour (achieved via improving code) would be to split up data into said sets.
        Arguments -
        start_month: Starting index of the month as can be determined using show_month_starts().
        end_month: Ending index of the month as can be determined using show_month_starts().
        '''

        print("Splitting into training and test sets (*not really, yet).")

        if end_month == -1:
            end_month = len(self.anomaly_removed_df)
        
        X_callcount, y_callcount = self.sliding_windows(self.anomaly_removed_df[self.final_columns[1]].values, self.seq_length)
        X_max1, y_max1 = self.sliding_windows(self.anomaly_removed_df[self.final_columns[2]].values, self.seq_length)
        X_max2, y_max2 = self.sliding_windows(self.anomaly_removed_df[self.final_columns[3]].values, self.seq_length)

        month_start = start_month
        month_split = end_month

        X_train_callcount_1 = X_callcount[month_start:month_split]
        y_train_callcount_1 = y_callcount[month_start:month_split]
        X_train_max1_1 = X_max1[month_start:month_split]
        y_train_max1_1 = y_max1[month_start:month_split]
        X_train_max2_1 = X_max2[month_start:month_split]
        y_train_max2_1 = y_max1[month_start:month_split]


        for i in range(len(X_train_callcount_1)):

            self.X_train.append([
                            X_train_callcount_1[i],
                            X_train_max1_1[i],
                            X_train_max2_1[i]
                        ])


        for i in range(len(y_train_callcount_1)):

            self.y_train.append([
                            y_train_callcount_1[i],
                            y_train_max1_1[i],
                            y_train_max2_1[i]
                        ])

        self.X_train = np.array(self.X_train)
        self.y_train = np.array(self.y_train)

        self.X_train = self.X_train.reshape((self.X_train.shape[0], self.X_train.shape[1], self.X_train.shape[2]))
        self.y_train = self.y_train.reshape((self.y_train.shape[0], self.y_train.shape[1], 1))


        X_callcount, y_callcount = self.sliding_windows(self.anomaly_included_df[self.final_columns[1]].values, self.seq_length)
        X_max1, y_max1 = self.sliding_windows(self.anomaly_included_df[self.final_columns[2]].values, self.seq_length)
        X_max2, y_max2 = self.sliding_windows(self.anomaly_included_df[self.final_columns[3]].values, self.seq_length)


        X_test_callcount_1 = X_callcount[month_start:month_split]
        y_test_callcount_1 = y_callcount[month_start:month_split]
        X_test_max1_1 = X_max1[month_start:month_split]
        y_test_max1_1 = y_max1[month_start:month_split]
        X_test_max2_1 = X_max2[month_start:month_split]
        y_test_max2_1 = X_max2[month_start:month_split]


        for i in range(len(X_test_callcount_1)):

            self.X_test.append([
                            X_test_callcount_1[i],
                            X_test_max1_1[i],
                            X_test_max2_1[i]
                        ])


        for i in range(len(y_test_callcount_1)):

            self.y_val.append([
                            y_test_callcount_1[i],
                            y_test_max1_1[i],
                            y_test_max2_1[i]
                        ])

        self.X_test = np.array(self.X_test)
        self.y_test = np.array(self.y_val)

        self.X_test = self.X_test.reshape((self.X_test.shape[0], self.X_test.shape[1], self.X_test.shape[2]))
        self.y_test = self.y_test.reshape((self.y_test.shape[0], self.y_test.shape[1], 1))


    def create_and_train_model(self, 
                    units = 64,
                    dropout_rate = 0.2,
                    epochs = 50,
                    batch_size = 72,
                    validation_split = 0.3
        ):

        '''
        Method that initializes, compiles and trains the model.
        Arguments -
        kwargs from the keras library for various parts of the training pipeline, namely, Sequential() constructor, and compile() and fit() methods.
        Defaults are as given below -
        units = 64, dropout_rate = 0.2, epochs = 50, batch_size = 72, validation_split = 0.3.
        '''

        print("Initializing model. Training should commence soon.")

        self.model = Sequential([
            LSTM(units=units, input_shape=(self.X_train.shape[1], self.X_train.shape[2])),
            Dropout(rate=dropout_rate),
            RepeatVector(self.X_train.shape[1]),
            LSTM(units=units, return_sequences=True),
            Dropout(rate=dropout_rate),
            TimeDistributed(Dense(self.X_train.shape[2]))
        ])

        self.model.compile(
            optimizer=keras.optimizers.Adam(),
            loss='mae',
            metrics=['accuracy']
        )

        self.model.model_history = self.model.fit(
            self.X_train,
            self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            shuffle=False
        )


    def plot_loss(self):

        '''
        Method that plots the training and validation losses recorded over the model training epochs.
        '''

        plt.plot(self.model.model_history.history['loss'], label='Training loss')
        plt.plot(self.model.model_history.history['val_loss'], label='Validation loss')
        plt.legend()
        plt.show()


    def plot_accuracy(self):

        '''
        Method that plots the training and validation accuracies recorded over the model training epochs.
        '''

        plt.plot(self.model.model_history.history['accuracy'], label='Training accuracy')
        plt.plot(self.model.model_history.history['val_accuracy'], label='Validation accuracy')
        plt.legend()
        plt.show()