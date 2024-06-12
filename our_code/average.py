import pandas as pd

# Load the dataset
# Assuming the dataset is in a CSV file named 'data.csv'
file_name = 'intermediate_datafiles/summary_cycling 2024-06-07 12-40-37.csv'
df = pd.read_csv(file_name)

# Set the window size for the rolling average
window_size = 10

# Exclude non-numeric columns (e.g., time columns)
numeric_df = df.select_dtypes(include=['float64', 'int64'])

# Calculate the rolling average for all numeric columns
rolling_df = numeric_df.rolling(window=window_size, min_periods=1).mean()

# Add the rolling average columns back to the original DataFrame
for col in rolling_df.columns:
    df[f'average_{col}'] = rolling_df[col]

# Check if the word "cycling" is in the file name and create the "activity" column
if 'cycling' in file_name.lower():
    df['activity'] = 'cycling'

# Display the resulting DataFrame
print(df)

# Save the resulting DataFrame to a CSV file
df.to_csv('output_summary_cycling.csv', index=False)

print(df.columns.tolist())
