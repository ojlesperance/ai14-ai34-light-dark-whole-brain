# Generates Ai14 cFos- raw cell count and z-scored count binned by regionY for Supplemental Fig. 2

import pandas as pd
import numpy as np
import nibabel as nib
import os

# ==========================================
# 1. FILE PATHS & SETUP
# ==========================================
base_dir = r"..\data"

# Your pre-existing parquet containing the spatial map (x, y, z, region_y)
parquet_path = os.path.join(base_dir, "corrected_df14_full.parquet")

# Output files
out_parquet = os.path.join(base_dir, "Ai14_Individual_Animals_Voxel_Data.parquet")
out_excel = os.path.join(base_dir, "Ai14_Individual_Animals_RegionY_Summary.xlsx")

# The six individual NIfTI files (ensure extension matches exactly what is in your folder, e.g., .nii or .nii.gz)
sample_files = {
    "Dark_01": "Ai14-Dark_sample01_cell_centroids_min6_s100_LRavg.nii",
    "Dark_02": "Ai14-Dark_sample02_cell_centroids_min6_s100_LRavg.nii",
    "Dark_03": "Ai14-Dark_sample03_cell_centroids_min6_s100_LRavg.nii",
    "Light_09": "Ai14-Light_sample09_cell_centroids_min6_s100_LRavg.nii",
    "Light_10": "Ai14-Light_sample10_cell_centroids_min6_s100_LRavg.nii",
    "Light_11": "Ai14-Light_sample11_cell_centroids_min6_s100_LRavg.nii"
}

# ==========================================
# 2. LOAD ATLAS COORDINATES & FILTER
# ==========================================
print("Loading base spatial atlas mapping from Parquet...")
df = pd.read_parquet(parquet_path)

# Drop any unassigned background voxels
df = df.dropna(subset=['region_y']).copy()

# Isolate the right hemisphere (x = 190 to 379) to prevent LR double counting of averaged voxels
df = df[(df['x'] >= 190) & (df['x'] <= 379)]
df['region_y'] = df['region_y'].replace('PTL', 'PTLp')

# Ensure coordinates are integers for array indexing
df['x'] = df['x'].astype(int)
df['y'] = df['y'].astype(int)
df['z'] = df['z'].astype(int)

# ==========================================
# 3. EXTRACT NIFTI DATA & Z-SCORE PER ANIMAL
# ==========================================
print("Extracting raw voxel data and computing intra-animal Z-scores...")
raw_cols = []
z_cols = []

for sample_name, filename in sample_files.items():
    filepath = os.path.join(base_dir, filename)
    
    # Try .nii.gz if .nii is not found
    if not os.path.exists(filepath) and os.path.exists(filepath + ".gz"):
        filepath += ".gz"
        
    if os.path.exists(filepath):
        print(f"  Processing {sample_name}...")
        img_data = nib.load(filepath).get_fdata()
        
        # Extract data using (z, y, x) indexing as defined by your parquet_to_nii script!
        raw_counts = img_data[df['z'], df['y'], df['x']]
        df[sample_name] = raw_counts
        raw_cols.append(sample_name)
        
        # Calculate mean and standard deviation for THIS specific animal's volume
        animal_mean = np.mean(raw_counts)
        animal_std = np.std(raw_counts, ddof=1)
        
        # Z-score the counts: (voxel - animal_mean) / animal_std
        z_col_name = f"{sample_name}_Z"
        df[z_col_name] = (raw_counts - animal_mean) / (animal_std + 1e-8)
        z_cols.append(z_col_name)
    else:
        print(f"  ERROR: {filename} not found in {base_dir}")

# Save the comprehensive voxel-level data back to a new Parquet file
df.to_parquet(out_parquet)
print(f"\nSaved comprehensive voxel mapping to: {out_parquet}")

# ==========================================
# 4. AGGREGATE BY REGION_Y
# ==========================================
print("Aggregating sums and averages by RegionY...")

# Define aggregation dictionary: Sum for raw counts, Mean for Z-scores
agg_dict = {}
for col in raw_cols:
    agg_dict[col] = 'sum'    # Total cells per region
for col in z_cols:
    agg_dict[col] = 'mean'   # Average Z-scored activation per region

agg_df = df.groupby('region_y').agg(agg_dict).reset_index()

# Rename columns for clarity in Excel
rename_dict = {'region_y': 'Region'}
for col in raw_cols:
    rename_dict[col] = f"{col}_Total_Cells"
for col in z_cols:
    rename_dict[col] = f"{col}_Avg_Z_Score"

agg_df = agg_df.rename(columns=rename_dict)
agg_df = agg_df.round(3)

# Sort by Dark_01 raw counts just to have a logical descending list
if 'Dark_01_Total_Cells' in agg_df.columns:
    agg_df = agg_df.sort_values(by='Dark_01_Total_Cells', ascending=False)

# ==========================================
# 5. EXPORT TO EXCEL
# ==========================================
agg_df.to_excel(out_excel, sheet_name='Animal_Summary', index=False)
print(f"Saved aggregated regional data to: {out_excel}")
print("Process completed successfully.")