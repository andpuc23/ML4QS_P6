import pandas as pd
import numpy as np
from datetime import timedelta
import matplotlib.pyplot as plt
from pathlib import Path
import os

# Get a list of all files and directories in the specified directory
directory_path = "our_code/data"
files = os.listdir(directory_path)

for file in files:
    DATASET_PATH = f'our_code/data/{file}/merged.csv'
    RESULT_PATH = Path('./intermediate_datafiles/')
    RESULT_FNAME = f'summary_{file}.csv'

    # Ensure the result directory exists
    RESULT_PATH.mkdir(exist_ok=True, parents=True)

    # Target granularity in milliseconds
    GRANULARITY = 100

    # Read the dataset
    df = pd.read_csv(DATASET_PATH)

    # Convert time columns to datetime
    time_columns = ['Time (s)_acc', 'Time (s)_gyro', 'Time (s)_mag', 'Time (s)_lin_acc', 'Time (s)_prox']

    for col in time_columns:
        df[col] = pd.to_datetime(df[col], unit='s', origin='unix')

    # Create a new dataframe with the desired granularity
    start_time = 0
    end_time = df[time_columns].max().max()
    timestamps = pd.date_range(start_time, end_time, freq=f'{GRANULARITY}ms')
    data_table = pd.DataFrame(index=timestamps)

    # Function to add numerical data
    def add_numerical_data(df, time_col, value_cols, prefix):
        for col in value_cols:
            data_table[f'{prefix}{col}'] = np.nan
        for i in range(len(data_table.index)):
            relevant_rows = df[(df[time_col] >= data_table.index[i]) & (df[time_col] < (data_table.index[i] + timedelta(milliseconds=GRANULARITY)))]
            for col in value_cols:
                if len(relevant_rows) > 0:
                    data_table.loc[data_table.index[i], f'{prefix}{col}'] = relevant_rows[col].mean()
                else:
                    data_table.loc[data_table.index[i], f'{prefix}{col}'] = np.nan

    # Add accelerometer data
    add_numerical_data(df, 'Time (s)_acc', ['Acceleration x (m/s^2)', 'Acceleration y (m/s^2)', 'Acceleration z (m/s^2)'], 'acc_')

    # Add gyroscope data
    add_numerical_data(df, 'Time (s)_gyro', ['Gyroscope x (rad/s)', 'Gyroscope y (rad/s)', 'Gyroscope z (rad/s)'], 'gyr_')

    # Add magnetometer data
    add_numerical_data(df, 'Time (s)_mag', ['Magnetic field x (µT)', 'Magnetic field y (µT)', 'Magnetic field z (µT)'], 'mag_')

    # Add linear acceleration data
    add_numerical_data(df, 'Time (s)_lin_acc', ['Linear Acceleration x (m/s^2)', 'Linear Acceleration y (m/s^2)', 'Linear Acceleration z (m/s^2)'], 'lin_acc_')

    # Save the resulting dataframe
    data_table.to_csv(RESULT_PATH / RESULT_FNAME)

print('The code has run through successfully!')
