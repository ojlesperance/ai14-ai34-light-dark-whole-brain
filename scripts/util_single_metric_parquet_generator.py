#Generates variance only (or othr single-metric) parquet files for conversion into .nii.gz maps, Ai14

import pandas as pd

# Path to your original file
input_path = r"..\data\corrected_df14_withStdOverMean.parquet"

# Load the full Parquet file
df = pd.read_parquet(input_path)

# ---------------------------------------------------------
# Keep ONLY the desired columns
# ---------------------------------------------------------
keep_cols = ["x", "y", "z", "Ai14+Dark_std_over_mean"]
df_reduced = df[keep_cols]

# ---------------------------------------------------------
# Save reduced Parquet
# ---------------------------------------------------------
output_path = r"..\output\corrected_df14_DarkStdOverMean.parquet"
df_reduced.to_parquet(output_path)

print("Finished creating reduced Dark_std_over_mean Parquet file.")

