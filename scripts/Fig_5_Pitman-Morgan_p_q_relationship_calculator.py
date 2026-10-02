# Pitman-Morgan p/q relationship calculator

import numpy as np
import pandas as pd
import nibabel as nib
import scipy.stats as stats
import statsmodels.stats.multitest as mt
import os
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. FILE PATHS & SETUP
# ==========================================
data_dir = r"..\data\Individual cFos+ mouse maps"

# --- Ai34 Files ---
ref_file_34 = r"..\data\corrected_df34_full.parquet"
orig_files_34 = [
    "Ai34+Light_sample28_Ai34_ochann_CCF_space_v2_s100_LRavg_sub_all_neg_z.nii.gz",
    "Ai34+Light_sample29_Ai34_ochann_CCF_space_v2_s100_LRavg_sub_all_neg_z.nii.gz",
    "Ai34+Light_sample30_Ai34_ochann_CCF_space_v2_s100_LRavg_sub_all_neg_z.nii.gz",
    "Ai34+Light_sample31_Ai34_ochann_CCF_space_v2_s100_LRavg_sub_all_neg_z.nii.gz",
    "Ai34+Light_sample32_Ai34_ochann_CCF_space_v2_s100_LRavg_sub_all_neg_z.nii.gz"
]
scram_files_34 = [
    "Ai34+Light_sample28_Ai34_ochann_CCF_space_v2_s100_LRavg_sub_all_neg_z_scrambled.nii.gz",
    "Ai34+Light_sample29_Ai34_ochann_CCF_space_v2_s100_LRavg_sub_all_neg_z_scrambled.nii.gz",
    "Ai34+Light_sample30_Ai34_ochann_CCF_space_v2_s100_LRavg_sub_all_neg_z_scrambled.nii.gz",
    "Ai34+Light_sample31_Ai34_ochann_CCF_space_v2_s100_LRavg_sub_all_neg_z_scrambled.nii.gz",
    "Ai34+Light_sample32_Ai34_ochann_CCF_space_v2_s100_LRavg_sub_all_neg_z_scrambled.nii.gz"
]

# --- Ai14 Files ---
ref_file_14 = r"..\data\corrected_df14_full.parquet"
orig_files_14 = [
    "Ai14+Light_sample12_cell_centroids_min6_s100_LRavg_sub_neg_z.nii.gz",
    "Ai14+Light_sample13_cell_centroids_min6_s100_LRavg_sub_neg_z.nii.gz",
    "Ai14+Light_sample14_cell_centroids_min6_s100_LRavg_sub_neg_z.nii.gz",
    "Ai14+Light_sample15_cell_centroids_min6_s100_LRavg_sub_neg_z.nii.gz",
    "Ai14+Light_sample16_cell_centroids_min6_s100_LRavg_sub_neg_z.nii.gz"
]
scram_files_14 = [
    "Ai14+Light_sample12_cell_centroids_min6_s100_LRavg_sub_neg_z_scrambled.nii.gz",
    "Ai14+Light_sample13_cell_centroids_min6_s100_LRavg_sub_neg_z_scrambled.nii.gz",
    "Ai14+Light_sample14_cell_centroids_min6_s100_LRavg_sub_neg_z_scrambled.nii.gz",
    "Ai14+Light_sample15_cell_centroids_min6_s100_LRavg_sub_neg_z_scrambled.nii.gz",
    "Ai14+Light_sample16_cell_centroids_min6_s100_LRavg_sub_neg_z_scrambled.nii.gz"
]

# ==========================================
# 2. SHARED HELPER FUNCTIONS
# ==========================================
def get_base_group(region_y):
    # FIXED: Replaced region_parent with region_y
    if region_y in ['PL', 'ILA', 'AI', 'PERI', 'ECT', 'ORB', 'VISC', 'VIS', 'TEa', 'MO', 'SS', 'ACA', 'RSP', 'AUD', 'PTLp', 'FRP', 'GU', 'ISO']: return 'Isocortex'
    if region_y in ['ENT', 'PAR', 'HATA', 'SUB', 'ProS', 'PRE', 'POST', 'DG', 'CA3', 'CA2', 'CA1', 'IG', 'HPF', 'RHP', 'HIP', 'FC']: return 'HPF'
    if region_y in ['EP', 'PA', 'CLA', 'BMA', 'BLA', 'LA', 'CTXsp']: return 'CTXsp'
    if region_y in ['SI', 'MS', 'TRS', 'GP', 'BST', 'PAL', 'FS', 'OT', 'ACB', 'CP', 'MEA', 'AAA', 'IA', 'SH', 'SF', 'LS', 'STR']: return 'CNU'
    if region_y in ['VPM', 'VPL', 'VM', 'VAL', 'SPA', 'SPF', 'LGd', 'MH', 'PT', 'PVT', 'CM', 'CL', 'PIL', 'IMD', 'SMT', 'MD', 'RT', 'AV', 'LD', 'LP', 'POL', 'PO', 'IGL', 'LGv', 'TH', 'PM', 'PH', 'VMH', 'MPN', 'TM', 'SUM', 'MM', 'TU', 'LHA', 'RCH', 'PVi', 'DMH', 'ADP', 'MEPO', 'SBPV', 'PV', 'PS', 'HY']: return 'IB'
    if region_y in ['SNc', 'DR', 'IPDL', 'IPL', 'PPN', 'PBG', 'IC', 'SC', 'SAG', 'INC', 'Su', 'ND', 'PAG', 'VTN', 'VTA', 'CUN', 'PN', 'MT', 'SNr', 'MRN', 'NOT', 'RPF', 'APN', 'PPT', 'MB']: return 'MB'
    if region_y in ['AON', 'TT', 'DP', 'PIR', 'NLOT', 'COA', 'PAA', 'TR', 'OLF']: return 'OLF'
    return 'Exclude'

def load_maps(file_list, directory):
    arrays = []
    for f in file_list:
        path = os.path.join(directory, f)
        img = nib.load(path)
        arrays.append(np.asanyarray(img.dataobj).squeeze())
    return np.stack(arrays, axis=-1)

# ==========================================
# 3. MAIN PITMAN-MORGAN ENGINE
# ==========================================
def run_diagnostic(orig_files, scram_files, ref_file, cohort_name):
    print(f"\n[{cohort_name}] Loading 3D Brain Volumes...")
    orig_data = load_maps(orig_files, data_dir)
    scram_data = load_maps(scram_files, data_dir)
    brain_shape = orig_data.shape[:-1]

    print(f"[{cohort_name}] Building anatomical mask...")
    ref_df = pd.read_parquet(ref_file, columns=['x', 'y', 'z', 'region_y', 'region_abbrev'])
    ref_df['region_y'] = ref_df['region_y'].replace('PTL', 'PTLp')
    ref_df['Base_Group'] = ref_df['region_y'].apply(get_base_group)

    valid_df = ref_df[ref_df['Base_Group'] != 'Exclude'].copy()
    valid_df = valid_df[~valid_df['region_abbrev'].astype(str).str.startswith(('MOB', 'AOB'))]
    valid_df = valid_df[valid_df['z'] >= 61] 
    valid_df = valid_df[(valid_df['x'] >= 190) & (valid_df['x'] <= 379)]

    anat_mask = np.zeros(brain_shape, dtype=bool)
    anat_mask[valid_df['z'].values, valid_df['y'].values, valid_df['x'].values] = True

    var_mask = (np.var(orig_data, axis=-1) > 0) & (np.var(scram_data, axis=-1) > 0)
    brain_mask = anat_mask & var_mask
    brain_indices = np.where(brain_mask)

    X = orig_data[brain_indices]
    Y = scram_data[brain_indices]

    print(f"[{cohort_name}] Executing Two-Tailed Pitman-Morgan Test on {X.shape[0]} valid voxels...")
    n = X.shape[1]

    S = X + Y
    D = X - Y

    S_mean = S.mean(axis=1, keepdims=True)
    D_mean = D.mean(axis=1, keepdims=True)
    S_diff = S - S_mean
    D_diff = D - D_mean

    cov = (S_diff * D_diff).sum(axis=1)
    var_S = (S_diff**2).sum(axis=1)
    var_D = (D_diff**2).sum(axis=1)

    denom = np.sqrt(var_S * var_D)
    r = np.divide(cov, denom, out=np.zeros_like(cov), where=denom != 0)

    df_stat = n - 2
    t_stat = r * np.sqrt(df_stat / (1 - r**2 + 1e-12))
    p_vals = 2 * stats.t.sf(np.abs(t_stat), df=df_stat)

    print(f"[{cohort_name}] Applying FDR Correction...")
    _, q_vals, _, _ = mt.multipletests(p_vals, alpha=0.20, method='fdr_bh')

    print("\n" + "="*50)
    print(f"   EXACT P-VALUE THRESHOLDS FOR FDR Q-VALUES ({cohort_name})")
    print("="*50)
    for q in [0.05, 0.10, 0.15, 0.20]:
        mask = q_vals < q
        if mask.any():
            max_p = p_vals[mask].max()
            print(f"For q < {q:.2f}, the maximum allowable p-value is: {max_p:.8f}")
        else:
            print(f"For q < {q:.2f}, no voxels survive FDR correction.")
    print("="*50 + "\n")

    # Clean up memory before the next cohort runs
    del orig_data, scram_data, X, Y
    import gc
    gc.collect()

# ==========================================
# 4. EXECUTE FOR BOTH COHORTS
# ==========================================
run_diagnostic(orig_files_34, scram_files_34, ref_file_34, "Ai34")
run_diagnostic(orig_files_14, scram_files_14, ref_file_14, "Ai14")