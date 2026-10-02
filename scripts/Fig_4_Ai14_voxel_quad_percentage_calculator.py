# Calculates % of Q1-4 voxels in each anatomical area, Ai14 (Memory Optimized)

import pandas as pd
import numpy as np
import os

# ==========================================
# 1. FILE PATHS & SETUP
# ==========================================
base_dir = r"..\data"
out_dir = r"..\output"

parquet_path = os.path.join(base_dir, "corrected_df14_full.parquet")
out_excel = os.path.join(out_dir, "Ai14_Voxel_Quadrant_Percentages.xlsx")

# ==========================================
# 2. MEMORY-OPTIMIZED DATA LOADING
# ==========================================
print("Loading base spatial mapping from Parquet (Memory Optimized)...")

# ONLY load the 7 columns necessary for the quadrant math to prevent RAM crashes
columns_to_load = [
    'x', 'region_y', 'region_abbrev', 
    'Ai14+Light_var', 'Ai14+Dark_var', 
    'Ai14+Light_mean', 'Ai14+Dark_mean'
]

df = pd.read_parquet(parquet_path, columns=columns_to_load)

# Isolate the right hemisphere (x = 190 to 379) to prevent LR double counting
df = df[(df['x'] >= 190) & (df['x'] <= 379)].copy()
df = df.dropna(subset=['region_y', 'region_abbrev'])
df['region_y'] = df['region_y'].replace('PTL', 'PTLp')
df['region_abbrev'] = df['region_abbrev'].replace('PTL', 'PTLp')

# ==========================================
# 3. CALCULATE VOXEL-WISE METRICS
# ==========================================
print("Calculating Voxel-wise Cohen's d and Log2SD...")

# Ensure no division by zero or log of zero
eps = 1e-8 

var_l = df['Ai14+Light_var']
var_d = df['Ai14+Dark_var']
mean_l = df['Ai14+Light_mean']
mean_d = df['Ai14+Dark_mean']

# Pooled standard deviation
pooled_std = np.sqrt((var_l + var_d) / 2 + eps)

# Light Cohen's d: Positive = Light Mean > Dark Mean
df['Cohens_d'] = (mean_l - mean_d) / pooled_std

# Light/Dark Log2SD: Positive = Light Var > Dark Var (Negative = Dark Var > Light Var)
df['Log2SD'] = np.log2(np.sqrt(var_l + eps) / np.sqrt(var_d + eps))

# ==========================================
# 4. ASSIGN QUADRANTS
# ==========================================
print("Assigning Phase-Space Quadrants...")
conditions = [
    (df['Cohens_d'] > 0) & (df['Log2SD'] > 0),   # Q1: +d, +var
    (df['Cohens_d'] < 0) & (df['Log2SD'] > 0),   # Q2: -d, +var
    (df['Cohens_d'] < 0) & (df['Log2SD'] < 0),   # Q3: -d, -var
    (df['Cohens_d'] > 0) & (df['Log2SD'] < 0)    # Q4: +d, -var
]
choices = ['Q1', 'Q2', 'Q3', 'Q4']

df['Quadrant'] = np.select(conditions, choices, default='Boundary/Zero')

# Filter out the extremely rare boundary/zero cases where values perfectly tie
df = df[df['Quadrant'] != 'Boundary/Zero'].copy()

# ==========================================
# 5. ASSIGN ANATOMICAL GROUPS
# ==========================================
print("Grouping by Anatomy...")

def get_base_group(region_y):
    if region_y in ['PL', 'ILA', 'AI', 'PERI', 'ECT', 'ORB', 'VISC', 'VIS', 'TEa', 'MO', 'SS', 'ACA', 'RSP', 'AUD', 'PTLp', 'PTL', 'FRP', 'GU', 'ISO']: return 'Isocortex'
    if region_y in ['ENT', 'PAR', 'HATA', 'SUB', 'ProS', 'PRE', 'POST', 'DG', 'CA3', 'CA2', 'CA1', 'IG', 'HPF', 'RHP', 'HIP', 'FC']: return 'HPF'
    if region_y in ['EP', 'PA', 'CLA', 'BMA', 'BLA', 'LA', 'CTXsp']: return 'CTXsp'
    if region_y in ['SI', 'MS', 'TRS', 'GP', 'BST', 'PAL', 'FS', 'OT', 'ACB', 'CP', 'MEA', 'AAA', 'IA', 'SH', 'SF', 'LS', 'STR']: return 'CNU'
    if region_y in ['VPM', 'VPL', 'VM', 'VAL', 'SPA', 'SPF', 'LGd', 'MH', 'PT', 'PVT', 'CM', 'CL', 'PIL', 'IMD', 'SMT', 'MD', 'RT', 'AV', 'LD', 'LP', 'POL', 'PO', 'IGL', 'LGv', 'TH', 'PM', 'PH', 'VMH', 'MPN', 'TM', 'SUM', 'MM', 'TU', 'LHA', 'RCH', 'PVi', 'DMH', 'ADP', 'MEPO', 'SBPV', 'PV', 'PS', 'HY']: return 'IB'
    if region_y in ['SNc', 'DR', 'IPDL', 'IPL', 'PPN', 'PBG', 'IC', 'SC', 'SAG', 'INC', 'Su', 'ND', 'PAG', 'VTN', 'VTA', 'CUN', 'PN', 'MT', 'SNr', 'MRN', 'NOT', 'RPF', 'APN', 'PPT', 'MB']: return 'MB'
    if region_y in ['AON', 'TT', 'DP', 'PIR', 'NLOT', 'COA', 'PAA', 'TR', 'OLF']: return 'OLF'
    return 'Exclude'

def is_visual_area(abbrev):
    abbrev = str(abbrev)
    if abbrev.startswith('VIS') and not abbrev.startswith('VISC'): return True
    if abbrev.startswith('LGd'): return True  # LGv explicitly removed
    if abbrev == 'SC' or abbrev.startswith(('SCm', 'SCs', 'SCig', 'SCop', 'SCsg', 'SCzo', 'SCdw', 'SCiw', 'SCdg')): return True
    return False

df['Anatomy'] = df['region_y'].apply(get_base_group)
df['is_VS'] = df['region_abbrev'].apply(is_visual_area) | df['region_y'].apply(is_visual_area)

df = df[df['Anatomy'] != 'Exclude']

# Create a master list of subsets to analyze
subsets = {
    'Whole Brain': df,
    'Isocortex': df[df['Anatomy'] == 'Isocortex'],
    'HPF (Hippocampal Formation)': df[df['Anatomy'] == 'HPF'],
    'CTXsp (Cortical Subplate)': df[df['Anatomy'] == 'CTXsp'],
    'CNU (Cerebral Nuclei)': df[df['Anatomy'] == 'CNU'],
    'IB (Interbrain)': df[df['Anatomy'] == 'IB'],
    'MB (Midbrain)': df[df['Anatomy'] == 'MB'],
    'Visual System': df[df['is_VS']] 
}

# ==========================================
# 6. CALCULATE PERCENTAGES & EXPORT
# ==========================================
print("Calculating Percentages...")
results = []

for group_name, subset_df in subsets.items():
    total_voxels = len(subset_df)
    if total_voxels == 0: continue
    
    # Count voxels in each quadrant
    counts = subset_df['Quadrant'].value_counts()
    
    # Calculate percentage (Count / Total * 100)
    q1_pct = (counts.get('Q1', 0) / total_voxels) * 100
    q2_pct = (counts.get('Q2', 0) / total_voxels) * 100
    q3_pct = (counts.get('Q3', 0) / total_voxels) * 100
    q4_pct = (counts.get('Q4', 0) / total_voxels) * 100
    
    results.append({
        'Anatomical Group': group_name,
        'Q1 (%) [+ Mean, + Var]': round(q1_pct, 2),
        'Q2 (%) [- Mean, + Var]': round(q2_pct, 2),
        'Q3 (%) [- Mean, - Var]': round(q3_pct, 2),
        'Q4 (%) [+ Mean, - Var]': round(q4_pct, 2),
        'Total Voxels': total_voxels
    })

final_df = pd.DataFrame(results)

# Export to formatted Excel
excel_writer = pd.ExcelWriter(out_excel, engine='xlsxwriter')
final_df.to_excel(excel_writer, sheet_name='Quadrant_Percentages', index=False)
worksheet = excel_writer.sheets['Quadrant_Percentages']

# Formatting
workbook = excel_writer.book
header_format = workbook.add_format({'bold': True, 'bg_color': '#f2f2f2', 'border': 1, 'align': 'center', 'valign': 'vcenter', 'text_wrap': True})
data_format = workbook.add_format({'align': 'center', 'valign': 'vcenter', 'border': 1})

worksheet.set_column(0, 0, 26, data_format)
worksheet.set_column(1, 4, 18, data_format)
worksheet.set_column(5, 5, 15, data_format)

for col_num, value in enumerate(final_df.columns.values):
    worksheet.write(0, col_num, value, header_format)

excel_writer.close()
print(f"Success! Excel saved to {out_excel}")