# makes scrambled maps of individual mice, RH only (Strict PM Mask), independent random permutations

import os
import numpy as np
import pandas as pd
import nibabel as nib

# ==========================================
# 1. FILE PATHS & SETUP
# ==========================================
data_dir = r"..\data\Individual cFos+ mouse maps"
ref_file = r"..\data\corrected_df34_full.parquet"

# ==========================================
# 2. DEFINE STRICT ANATOMICAL MASK GROUPS
# ==========================================
def get_base_group(region_y):
    if region_y in ['PL', 'ILA', 'AI', 'PERI', 'ECT', 'ORB', 'VISC', 'VIS', 'TEa', 'MO', 'SS', 'ACA', 'RSP', 'AUD', 'PTLp', 'PTL', 'FRP', 'GU', 'ISO']: return 'Isocortex'
    if region_y in ['ENT', 'PAR', 'HATA', 'SUB', 'ProS', 'PRE', 'POST', 'DG', 'CA3', 'CA2', 'CA1', 'IG', 'HPF', 'RHP', 'HIP', 'FC']: return 'HPF'
    if region_y in ['EP', 'PA', 'CLA', 'BMA', 'BLA', 'LA', 'CTXsp']: return 'CTXsp'
    if region_y in ['SI', 'MS', 'TRS', 'GP', 'BST', 'PAL', 'FS', 'OT', 'ACB', 'CP', 'MEA', 'AAA', 'IA', 'SH', 'SF', 'LS', 'STR']: return 'CNU'
    if region_y in ['VPM', 'VPL', 'VM', 'VAL', 'SPA', 'SPF', 'LGd', 'MH', 'PT', 'PVT', 'CM', 'CL', 'PIL', 'IMD', 'SMT', 'MD', 'RT', 'AV', 'LD', 'LP', 'POL', 'PO', 'IGL', 'LGv', 'TH', 'PM', 'PH', 'VMH', 'MPN', 'TM', 'SUM', 'MM', 'TU', 'LHA', 'RCH', 'PVi', 'DMH', 'ADP', 'MEPO', 'SBPV', 'PV', 'PS', 'HY']: return 'IB'
    if region_y in ['SNc', 'DR', 'IPDL', 'IPL', 'PPN', 'PBG', 'IC', 'SC', 'SAG', 'INC', 'Su', 'ND', 'PAG', 'VTN', 'VTA', 'CUN', 'PN', 'MT', 'SNr', 'MRN', 'NOT', 'RPF', 'APN', 'PPT', 'MB']: return 'MB'
    if region_y in ['AON', 'TT', 'DP', 'PIR', 'NLOT', 'COA', 'PAA', 'TR', 'OLF']: return 'OLF'
    return 'Exclude'

# ==========================================
# 3. FIND ALL INDIVIDUAL MOUSE MAPS
# ==========================================
nifti_files = []
for root, dirs, files in os.walk(data_dir):
    for file in files:
        if file.endswith(".nii.gz") and "_scrambled" not in file:
            nifti_files.append(os.path.join(root, file))

if not nifti_files:
    print(f"No .nii.gz files found in {data_dir}. Exiting.")
    exit()

print(f"Found {len(nifti_files)} original mouse maps to scramble.")

# ==========================================
# 4. BUILD STRICT PITMAN-MORGAN MASK
# ==========================================
print("Loading reference Parquet to define strict Area of Interest bounds...")
ref_df = pd.read_parquet(ref_file, columns=['x', 'y', 'z', 'region_y', 'region_abbrev'])

# Harmonize PTL -> PTLp
ref_df['region_y'] = ref_df['region_y'].replace('PTL', 'PTLp')
ref_df['region_abbrev'] = ref_df['region_abbrev'].replace('PTL', 'PTLp')
ref_df['Base_Group'] = ref_df['region_y'].apply(get_base_group)

# Apply exact Pitman-Morgan exclusions
valid_df = ref_df[ref_df['Base_Group'] != 'Exclude'].copy()
valid_df = valid_df[~valid_df['region_abbrev'].astype(str).str.startswith(('MOB', 'AOB'))]
valid_df = valid_df[valid_df['z'] >= 61] 
valid_df = valid_df[(valid_df['x'] >= 190) & (valid_df['x'] <= 379)]

# Grab the shape of the first NIfTI file to size our 3D mask
sample_img = nib.load(nifti_files[0])
brain_shape = sample_img.shape

# Map valid Parquet coordinates back into the 3D boolean mask
strict_hemi_mask = np.zeros(brain_shape, dtype=bool)
strict_hemi_mask[valid_df['z'].values, valid_df['y'].values, valid_df['x'].values] = True

num_valid_voxels = np.sum(strict_hemi_mask)
print(f"Strict Area of Interest defined. Found {num_valid_voxels} valid voxels to scramble.")

# ==========================================
# 5. INDEPENDENT SCRAMBLE & SAVE ITERATION
# ==========================================
# Setting seed outside the loop ensures the sequence of permutations is reproducible
np.random.seed(42) 

for file_path in nifti_files:
    print(f"Processing: {os.path.basename(file_path)}...")
    
    img = nib.load(file_path)
    data = img.get_fdata()
    
    # Safety Check
    if data.shape != brain_shape:
        print(f"  --> ERROR: Shape mismatch for {os.path.basename(file_path)}. "
              f"Expected {brain_shape}, got {data.shape}. Skipping.")
        continue
    
    # 1. Extract ONLY the voxels inside the strict Pitman-Morgan mask
    valid_voxels = data[strict_hemi_mask]
    
    # 2. Generate an INDEPENDENT random permutation for this specific mouse
    independent_permutation = np.random.permutation(num_valid_voxels)
    scrambled_voxels = valid_voxels[independent_permutation]
    
    # 3. Create a copy of the original array (preserves excluded regions entirely untouched)
    scrambled_data = data.copy()
    
    # 4. Inject the universally shuffled values back into the strict mask coordinates
    scrambled_data[strict_hemi_mask] = scrambled_voxels
    
    # 5. Save new NIfTI
    scrambled_img = nib.Nifti1Image(scrambled_data, img.affine, img.header)
    
    dir_name = os.path.dirname(file_path)
    base_name = os.path.basename(file_path)
    name_without_ext = base_name.replace('.nii.gz', '')
    
    new_file_name = f"{name_without_ext}_scrambled.nii.gz"
    new_file_path = os.path.join(dir_name, new_file_name)
    
    nib.save(scrambled_img, new_file_path)
    print(f"  --> Saved: {new_file_name}")

print("\nSuccess! All maps have been spatially scrambled independently within the defined Area of Interest.")