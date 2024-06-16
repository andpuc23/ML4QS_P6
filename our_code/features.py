import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# Load the CSV file into a DataFrame
file_name = 'intermediate_datafiles/summary_cycling 2024-06-07 12-40-37.csv'
df = pd.read_csv(file_name)

# Calculate the magnitude of acceleration and gyroscope data
df['acc_magnitude'] = np.sqrt(df['acc_Acceleration x (m/s^2)']**2 + df['acc_Acceleration y (m/s^2)']**2 + df['acc_Acceleration z (m/s^2)']**2)
df['gyr_magnitude'] = np.sqrt(df['gyr_Gyroscope x (rad/s)']**2 + df['gyr_Gyroscope y (rad/s)']**2 + df['gyr_Gyroscope z (rad/s)']**2)

# Define a function to apply a low-pass filter
def low_pass_filter(data, cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

# Parameters for the low-pass filter
cutoff_frequency = 0.1  # Adjust this value based on your data
sampling_rate = 1  # Assuming 1 Hz sampling rate, adjust if different

# Apply the low-pass filter to the magnitude data
df['filtered_acc_magnitude'] = low_pass_filter(df['acc_magnitude'], cutoff_frequency, sampling_rate)

# Plot the original and filtered acceleration magnitude
plt.figure(figsize=(14, 7))
plt.subplot(2, 1, 1)
plt.plot(df['acc_magnitude'], label='Original Acceleration Magnitude')
plt.plot(df['filtered_acc_magnitude'], label='Filtered Acceleration Magnitude', linewidth=2)
plt.title('Acceleration Magnitude')
plt.xlabel('Time')
plt.ylabel('Magnitude (m/s^2)')
plt.legend()

# Plot the original and filtered gyroscope magnitude
plt.subplot(2, 1, 2)
plt.plot(df['gyr_magnitude'], label='Original Gyroscope Magnitude')
plt.title('Gyroscope Magnitude')
plt.xlabel('Time')
plt.ylabel('Magnitude (rad/s)')
plt.legend()

plt.tight_layout()
plt.show()