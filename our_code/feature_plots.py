import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# Load the CSV file into a DataFrame
file_name = 'intermediate_datafiles/summary_cycling 2024-06-07 12-40-37.csv'
df = pd.read_csv(file_name)

# Calculate the magnitude of acceleration, gyroscope, linear acceleration, and magnetic field data
df['acc_magnitude'] = np.sqrt(df['acc_Acceleration x (m/s^2)']**2 + df['acc_Acceleration y (m/s^2)']**2 + df['acc_Acceleration z (m/s^2)']**2)
df['gyr_magnitude'] = np.sqrt(df['gyr_Gyroscope x (rad/s)']**2 + df['gyr_Gyroscope y (rad/s)']**2 + df['gyr_Gyroscope z (rad/s)']**2)
df['lin_acc_magnitude'] = np.sqrt(df['lin_acc_Linear Acceleration x (m/s^2)']**2 + df['lin_acc_Linear Acceleration y (m/s^2)']**2 + df['lin_acc_Linear Acceleration z (m/s^2)']**2)
df['mag_magnitude'] = np.sqrt(df['mag_Magnetic field x (µT)']**2 + df['mag_Magnetic field y (µT)']**2 + df['mag_Magnetic field z (µT)']**2)

# Define a function to apply a low-pass filter
def low_pass_filter(data, cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

# Parameters for the low-pass filter
cutoff_frequency = 0.1 # Adjust this value based on your data
sampling_rate = 1 # Assuming 1 Hz sampling rate, adjust if different

# Apply the low-pass filter to the magnitude data
df['filtered_acc_magnitude'] = low_pass_filter(df['acc_magnitude'], cutoff_frequency, sampling_rate)
df['filtered_mag_magnitude'] = low_pass_filter(df['mag_magnitude'], cutoff_frequency, sampling_rate)

# Set the window size for the rolling average
window_size = 100

# Exclude non-numeric columns (e.g., time columns)
numeric_df = df.select_dtypes(include=['float64', 'int64'])

# Calculate the rolling average for all numeric columns
rolling_df = numeric_df.rolling(window=window_size, min_periods=1).mean()

# Add the rolling average columns back to the original DataFrame
for col in rolling_df.columns:
    df[f'average_{col}'] = rolling_df[col]

# Calculate the rolling average for the magnitude of acceleration, gyroscope, linear acceleration, and magnetic field
df['average_acc_magnitude'] = df['acc_magnitude'].rolling(window=window_size, min_periods=1).mean()
df['average_gyr_magnitude'] = df['gyr_magnitude'].rolling(window=window_size, min_periods=1).mean()
df['average_lin_acc_magnitude'] = df['lin_acc_magnitude'].rolling(window=window_size, min_periods=1).mean()
df['average_mag_magnitude'] = df['mag_magnitude'].rolling(window=window_size, min_periods=1).mean()

# Plot the original and filtered acceleration magnitude along with the rolling average
plt.figure(figsize=(14, 14))

# Plot the acceleration magnitude and its rolling average
plt.subplot(3, 1, 1)
plt.plot(df['acc_magnitude'], label='Original Acceleration Magnitude')
plt.plot(df['filtered_acc_magnitude'], label='Filtered Acceleration Magnitude', linewidth=2)
plt.plot(df['average_acc_magnitude'], label='Rolling Average Acceleration Magnitude', color='red', linewidth=2)
plt.title('Acceleration Magnitude')
plt.xlabel('Time')
plt.ylabel('Magnitude (m/s^2)')
plt.legend()

# Plot the gyroscope magnitude and its rolling average
plt.subplot(3, 1, 2)
plt.plot(df['gyr_magnitude'], label='Original Gyroscope Magnitude')
plt.plot(df['average_gyr_magnitude'], label='Rolling Average Gyroscope Magnitude', color='red', linewidth=2)
plt.title('Gyroscope Magnitude')
plt.xlabel('Time')
plt.ylabel('Magnitude (rad/s)')
plt.legend()

# Plot the linear acceleration magnitude and its rolling average
plt.subplot(3, 1, 3)
plt.plot(df['lin_acc_magnitude'], label='Original Linear Acceleration Magnitude')
plt.plot(df['average_lin_acc_magnitude'], label='Rolling Average Linear Acceleration Magnitude', color='red', linewidth=2)
plt.title('Linear Acceleration Magnitude')
plt.xlabel('Time')
plt.ylabel('Magnitude (m/s^2)')
plt.legend()

plt.tight_layout()
plt.show()

# Plot the magnetic field magnitude and its rolling average
plt.figure(figsize=(10, 6)) # Optional: Set the figure size for better visualization
plt.plot(df['mag_magnitude'], label='Original Magnetic Field Magnitude')
plt.plot(df['filtered_mag_magnitude'], label='Filtered Magnetic Field Magnitude', linewidth=2)
plt.plot(df['average_mag_magnitude'], label='Rolling Average Magnetic Field Magnitude', color='red', linewidth=2)
plt.title('Magnetic Field Magnitude')
plt.xlabel('Time')
plt.ylabel('Magnitude (µT)')
plt.legend()
plt.autoscale(enable=True, axis='y', tight=True)
plt.tight_layout()
plt.show()
