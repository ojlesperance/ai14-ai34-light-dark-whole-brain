#Corrects region names on original parquet file, adds subregion delineation. Then calculates mean/var, Dark z - Light z, and Dark var - Light var

import pandas as pd

# ---------------------------------------------------------
# Load original voxel file (region_id == lowered_ID)
# ---------------------------------------------------------
df = pd.read_parquet(
    r"..\data\Ai14_voxel_info.parquet"
)

# ---------------------------------------------------------
# Load atlas CSV containing lowered_ID → region mappings
# ---------------------------------------------------------
atlas = pd.read_csv(
    r"..\data\CCFv3-2020_info_oliver_voxel_info.csv"
)

# Keep only the columns we need
atlas_subset = atlas[['lowered_ID', 'collapsed_region', 'abbreviation', 'region']]

# ---------------------------------------------------------
# Merge voxel lowered_ID (stored as region_id) with atlas lowered_ID
# ---------------------------------------------------------
df = df.merge(
    atlas_subset,
    left_on='region_id',     # voxel lowered_ID
    right_on='lowered_ID',   # atlas lowered_ID
    how='left'
)

# ---------------------------------------------------------
# Rename atlas columns for clarity
# ---------------------------------------------------------
df = df.rename(columns={
    'collapsed_region': 'region_collapsed',
    'abbreviation': 'region_abbrev',
    'region': 'region_coarse'
})

# ---------------------------------------------------------
# Remove only the atlas duplicate lowered_ID column
# Keep region_y (your legacy custom region set)
# Remove region_x (incorrect original region column)
# ---------------------------------------------------------
cols_to_drop = []
if 'lowered_ID' in df.columns:
    cols_to_drop.append('lowered_ID')
if 'region_x' in df.columns:
    cols_to_drop.append('region_x')

df = df.drop(columns=cols_to_drop)

# ---------------------------------------------------------
# Compute Light/Dark mean/var ratios
# ---------------------------------------------------------
df["Ai14+Light_mean_over_var"] = df["Ai14+Light_mean"] / df["Ai14+Light_var"]
df["Ai14+Dark_mean_over_var"]  = df["Ai14+Dark_mean"]  / df["Ai14+Dark_var"]

# ---------------------------------------------------------
# Compute Dark minus Light differences
# ---------------------------------------------------------
df["Ai14+Dark_Mean_Minus_Light_Mean"] = df["Ai14+Dark_mean"] - df["Ai14+Light_mean"]
df["Ai14+Dark_Var_Minus_Light_var"]   = df["Ai14+Dark_var"]  - df["Ai14+Light_var"]

# ---------------------------------------------------------
# Save final corrected + enriched Parquet
# ---------------------------------------------------------
output_path = r"..\data\corrected_df14_full.parquet"
df.to_parquet(output_path)

print("Region labels added, region_x removed, region_y preserved, and all derived statistics computed.")

