#makes scrambled maps of individual mice, RH only, same permutation applied to all

import os
import numpy as np
import nibabel as nib

# ==========================================
# 1. FILE PATHS & SETUP
# ==========================================
data_dir = r"..\data\Individual cFos+ mouse maps"
atlas_path = r"..\data\atlas_CCFv3_2020_30um.nii.gz"

print("Loading Atlas to define right hemisphere boundaries...")
atlas_img = nib.load(atlas_path)
atlas_data = atlas_img.get_fdata()

# Create a 3D grid of the spatial coordinates. 
# In a standard (X, Y, Z) NIfTI array, Z is the 3rd dimension (index 2).
z_coords = np.indices(atlas_data.shape)[2]

# Create the combined mask: 
# Must be a valid atlas region AND located in the right hemisphere (Z > 189)
right_hemi_mask = (atlas_data > 0) & (z_coords > 189)
num_valid_voxels = np.sum(right_hemi_mask)

print(f"Atlas loaded. Found {num_valid_voxels} valid right hemisphere voxels.")

# ==========================================
# 2. GENERATE UNIVERSAL PERMUTATION
# ==========================================
# Create ONE permutation index to be applied identically to every mouse map
# Setting a seed ensures you can recreate this exact scramble later if needed
np.random.seed(42) 
universal_permutation = np.random.permutation(num_valid_voxels)
print("Universal spatial permutation array generated.")

# ==========================================
# 3. FIND ALL INDIVIDUAL MOUSE MAPS
# ==========================================
# Use os.walk to go through the main directory and all subdirectories
nifti_files = []
for root, dirs, files in os.walk(data_dir):
    for file in files:
        if file.endswith(".nii.gz") and "_scrambled" not in file:
            nifti_files.append(os.path.join(root, file))

print(f"Found {len(nifti_files)} original mouse maps to scramble.")

# ==========================================
# 4. SCRAMBLE & SAVE ITERATION
# ==========================================
for file_path in nifti_files:
    print(f"Processing: {os.path.basename(file_path)}...")
    
    # Load the individual mouse map
    img = nib.load(file_path)
    data = img.get_fdata()
    
    # Safety Check: Ensure the mouse map has the exact same dimensions as the atlas
    if data.shape != atlas_data.shape:
        print(f"  --> ERROR: Shape mismatch for {os.path.basename(file_path)}. "
              f"Expected {atlas_data.shape}, got {data.shape}. Skipping.")
        continue
    
    # 1. Extract ONLY the valid right hemisphere voxels
    valid_voxels = data[right_hemi_mask]
    
    # 2. Apply the UNIVERSAL permutation to those voxels
    scrambled_voxels = valid_voxels[universal_permutation]
    
    # 3. Create a copy of the original array to preserve the left hemisphere and background
    scrambled_data = data.copy()
    
    # 4. Inject the universally shuffled values back into the right hemisphere coordinates
    scrambled_data[right_hemi_mask] = scrambled_voxels
    
    # 5. Create a new NIfTI image using the scrambled data and the original affine/header
    scrambled_img = nib.Nifti1Image(scrambled_data, img.affine, img.header)
    
    # Construct the new filename
    dir_name = os.path.dirname(file_path)
    base_name = os.path.basename(file_path)
    name_without_ext = base_name.replace('.nii.gz', '')
    
    new_file_name = f"{name_without_ext}_scrambled.nii.gz"
    new_file_path = os.path.join(dir_name, new_file_name)
    
    # Save the output
    nib.save(scrambled_img, new_file_path)
    print(f"  --> Saved: {new_file_name}")

print("\nSuccess! All maps have been spatially scrambled strictly within the right hemisphere bounds using an identical universal permutation.")