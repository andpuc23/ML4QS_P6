import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt

# Load the CSV file
file_name = 'intermediate_datafiles/summary_cycling 2024-06-07 12-40-37.csv'
df = pd.read_csv(file_name)

# Calculate gyroscope magnitude
df['gyr_magnitude'] = np.sqrt(df['gyr_Gyroscope x (rad/s)']**2 + df['gyr_Gyroscope y (rad/s)']**2 + df['gyr_Gyroscope z (rad/s)']**2)

# Define the low-pass filter
def low_pass_filter(data, cutoff_frequency, sampling_rate, order=5):
    nyquist = 0.5 * sampling_rate
    normal_cutoff = cutoff_frequency / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

# Parameters for the low-pass filter
cutoff_frequency = 0.1  # Adjust as needed
sampling_rate = 1  # Adjust based on your data's sampling rate

# Apply the low-pass filter to the gyroscope magnitude
df['filtered_gyr_magnitude'] = low_pass_filter(df['gyr_magnitude'], cutoff_frequency, sampling_rate)

# Plot the original and filtered gyroscope magnitude
plt.figure(figsize=(10, 6))
plt.plot(df['gyr_magnitude'], label='Original Gyroscope Magnitude')
plt.plot(df['filtered_gyr_magnitude'], label='Filtered Gyroscope Magnitude', color='orange')
plt.xlabel('Time')
plt.ylabel('Magnitude (rad/s)')
plt.title('Gyroscope Magnitude')
plt.legend()
plt.show()

# Save the DataFrame to a new CSV file
df.to_csv('filtered_gyroscope_data.csv', index=False)