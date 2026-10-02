#.parquet spot checker 

import pandas as pd
# Load the Parquet file
df34 = pd.read_parquet(r"..\data\corrected_df34_full.parquet")
# Filter the row with the specific coordinates
filtered_row = df34[(df34['x'] == 267) & (df34['y'] == 117) & (df34['z'] == 256)]
pd.set_option('display.max_columns', None) 
pd.set_option('display.max_colwidth', None) 
pd.set_option('display.width', None)
# Display the filtered row
print(filtered_row)