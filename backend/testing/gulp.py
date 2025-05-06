import pandas as pd
import numpy as np

# Read the CSV and drop missing values
df = pd.read_csv(r'/mnt/raw_data/biomarkers.csv').dropna()

# Print column names to verify
print(df.columns)

# Function to convert values to 0, 1, or 2 based on quartiles
def quartile_label(series):
    q1 = series.quantile(0.33333)  # Lower quartile (25th percentile)
    q3 = series.quantile(0.66666)  # Upper quartile (75th percentile)
    return series.apply(lambda x: 0 if x <= q1 else 2 if x >= q3 else 1)

# Create a new DataFrame to store the converted values
df_quartiles = df.copy()

# Loop through each column and apply the quartile conversion
for column in df_quartiles.columns:
    # Ensure the column is numeric; skip if not
    if df_quartiles[column].dtype in ['float64', 'int64']:
        df_quartiles[column] = quartile_label(df_quartiles[column])
    else:
        print(f"Skipping non-numeric column: {column}")

# Print the first few rows to verify
output_path = r'/mnt/raw_data/biomarkers_thirds.csv'
df_quartiles.to_csv(output_path, index=False)
print(df_quartiles.head())