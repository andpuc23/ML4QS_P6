import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import os
import glob

def low_pass_filter(data, cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

def process_and_save_csv_files(folder_path):
    cutoff_frequency = 0.1  # Adjust this value based on your data
    sampling_rate = 1  # Assuming 1 Hz sampling rate, adjust if different
    window_size = 100

    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv'):
            file_path = os.path.join(folder_path, file_name)
            df = pd.read_csv(file_path)

            # Calculate the magnitude of acceleration, gyroscope, linear acceleration, and magnetic field data
            df['acc_magnitude'] = np.sqrt(df['acc_Acceleration x (m/s^2)']**2 + df['acc_Acceleration y (m/s^2)']**2 + df['acc_Acceleration z (m/s^2)']**2)
            df['gyr_magnitude'] = np.sqrt(df['gyr_Gyroscope x (rad/s)']**2 + df['gyr_Gyroscope y (rad/s)']**2 + df['gyr_Gyroscope z (rad/s)']**2)
            df['lin_acc_magnitude'] = np.sqrt(df['lin_acc_Linear Acceleration x (m/s^2)']**2 + df['lin_acc_Linear Acceleration y (m/s^2)']**2 + df['lin_acc_Linear Acceleration z (m/s^2)']**2)
            df['mag_magnitude'] = np.sqrt(df['mag_Magnetic field x (µT)']**2 + df['mag_Magnetic field y (µT)']**2 + df['mag_Magnetic field z (µT)']**2)

            # Apply the low-pass filter to the magnitude data
            df['filtered_acc_magnitude'] = low_pass_filter(df['acc_magnitude'], cutoff_frequency, sampling_rate)
            df['filtered_mag_magnitude'] = low_pass_filter(df['mag_magnitude'], cutoff_frequency, sampling_rate)

            # Calculate the rolling average for all numeric columns
            numeric_df = df.select_dtypes(include=['float64', 'int64'])
            rolling_df = numeric_df.rolling(window=window_size, min_periods=1).mean()

            # Add the rolling average columns back to the original DataFrame
            for col in rolling_df.columns:
                df[f'average_{col}'] = rolling_df[col]

            # Calculate the rolling average for the magnitude of acceleration, gyroscope, linear acceleration, and magnetic field
            df['average_acc_magnitude'] = df['acc_magnitude'].rolling(window=window_size, min_periods=1).mean()
            df['average_gyr_magnitude'] = df['gyr_magnitude'].rolling(window=window_size, min_periods=1).mean()
            df['average_lin_acc_magnitude'] = df['lin_acc_magnitude'].rolling(window=window_size, min_periods=1).mean()
            df['average_mag_magnitude'] = df['mag_magnitude'].rolling(window=window_size, min_periods=1).mean()

            # Check if the word "cycling" is in the file name and create the "activity" column
            if 'cycling' in file_name.lower():
                df['activity'] = 'cycling'
            elif 'running' in file_name.lower():
                df['activity'] = 'running'
            elif 'walking' in file_name.lower():
                df['activity'] = 'walking'

            # Drop magnetometer columns before saving the DataFrame
            df_to_save = df.drop(columns=[col for col in df.columns if 'mag_' in col])

            # Define the output folder
            output_folder = 'preprocessed'

            # Create the folder if it doesn't exist
            os.makedirs(output_folder, exist_ok=True)

            # Save the resulting DataFrame to a CSV file
            output_file_name = os.path.join(output_folder, f'output_{os.path.basename(file_name)}')
            df_to_save.to_csv(output_file_name, index=False)
            print(f"DataFrame saved to {output_file_name}")

folder_path = 'intermediate_datafiles'
process_and_save_csv_files(folder_path)



# List all CSV files in the directory
csv_files = glob.glob('preprocessed/*.csv')

# Concatenate all CSV files
df_list = [pd.read_csv(file) for file in csv_files]
combined_df = pd.concat(df_list, ignore_index=True)

# Drop the rows with missing values
combined_df = combined_df.dropna()

# Save the combined dataframe to a new CSV file
combined_df.to_csv('combined_data.csv', index=False)

print(f"Combined all the data to combined_data.csv")
