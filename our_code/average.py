import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
file_name = 'intermediate_datafiles/summary_cycling 2024-06-07 12-40-37.csv'
df = pd.read_csv(file_name)

# Calculate the magnitude of acceleration and gyroscope data
df['acc_magnitude'] = np.sqrt(df['acc_Acceleration x (m/s^2)']**2 + df['acc_Acceleration y (m/s^2)']**2 + df['acc_Acceleration z (m/s^2)']**2)
df['gyr_magnitude'] = np.sqrt(df['gyr_Gyroscope x (rad/s)']**2 + df['gyr_Gyroscope y (rad/s)']**2 + df['gyr_Gyroscope z (rad/s)']**2)

# Set the window size for the rolling average
window_size = 10

# Exclude non-numeric columns (e.g., time columns)
numeric_df = df.select_dtypes(include=['float64', 'int64'])

# Calculate the rolling average for all numeric columns
rolling_df = numeric_df.rolling(window=window_size, min_periods=1).mean()

# Add the rolling average columns back to the original DataFrame
for col in rolling_df.columns:
    df[f'average_{col}'] = rolling_df[col]

# Calculate the rolling average for the magnitude of acceleration and gyroscope
df['average_acc_magnitude'] = df['acc_magnitude'].rolling(window=window_size, min_periods=1).mean()
df['average_gyr_magnitude'] = df['gyr_magnitude'].rolling(window=window_size, min_periods=1).mean()

# Check if the word "cycling" is in the file name and create the "activity" column
if 'cycling' in file_name.lower():
    df['activity'] = 'cycling'

# Display the resulting DataFrame
print(df)

# Save the resulting DataFrame to a CSV file
output_file_name = 'output_summary_cycling.csv'
df.to_csv(output_file_name, index=False)

print(f"DataFrame saved to {output_file_name}")
print(df.columns.tolist())

# Plot the rolling average magnitude for acceleration and gyroscope
plt.figure(figsize=(14, 7))

# Plot the rolling average for acceleration magnitude
plt.subplot(2, 1, 1)
plt.plot(df['average_acc_magnitude'], label='Rolling Average Acceleration Magnitude', color='blue', linewidth=2)
plt.title('Rolling Average Acceleration Magnitude')
plt.xlabel('Time')
plt.ylabel('Magnitude (m/s^2)')
plt.legend()

# Plot the rolling average for gyroscope magnitude
plt.subplot(2, 1, 2)
plt.plot(df['average_gyr_magnitude'], label='Rolling Average Gyroscope Magnitude', color='red', linewidth=2)
plt.title('Rolling Average Gyroscope Magnitude')
plt.xlabel('Time')
plt.ylabel('Magnitude (rad/s)')
plt.legend()

plt.tight_layout()
plt.show()