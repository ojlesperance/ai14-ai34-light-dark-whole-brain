# Ai14 Pitman-Morgan Engine and Excel Table Generator

import numpy as np
import pandas as pd
import nibabel as nib
import scipy.stats as stats
import statsmodels.stats.multitest as mt
from scipy.ndimage import label, generate_binary_structure
import os
import pyarrow.parquet as pq

# ==========================================
# 1. FILE PATHS & SETUP
# ==========================================
data_dir = r"C:\Users\ojles\Documents\Ai34_Project\Individual cFos+ mouse maps (BS) for Dan"
out_dir = r"C:\Users\ojles\Documents\Ai34_Project"

ref_file = r"C:\Users\ojles\Documents\Ai34_Project\corrected_df14_full.parquet"
csv_info_file = r"C:\Users\ojles\Documents\Ai34_Project\CCFv3-2020_info_oliver_voxel_info.csv" 
out_excel_master = r"C:\Users\ojles\Documents\Ai34_Project\Ai14_Scram_Master_Tables_TwoWay.xlsx"

orig_files = [
    "Ai14+Light_sample12_cell_centroids_min6_s100_LRavg_sub_neg_z.nii.gz",
    "Ai14+Light_sample13_cell_centroids_min6_s100_LRavg_sub_neg_z.nii.gz",
    "Ai14+Light_sample14_cell_centroids_min6_s100_LRavg_sub_neg_z.nii.gz",
    "Ai14+Light_sample15_cell_centroids_min6_s100_LRavg_sub_neg_z.nii.gz",
    "Ai14+Light_sample16_cell_centroids_min6_s100_LRavg_sub_neg_z.nii.gz"
]

scram_files = [
    "Ai14+Light_sample12_cell_centroids_min6_s100_LRavg_sub_neg_z_scrambled.nii.gz",
    "Ai14+Light_sample13_cell_centroids_min6_s100_LRavg_sub_neg_z_scrambled.nii.gz",
    "Ai14+Light_sample14_cell_centroids_min6_s100_LRavg_sub_neg_z_scrambled.nii.gz",
    "Ai14+Light_sample15_cell_centroids_min6_s100_LRavg_sub_neg_z_scrambled.nii.gz",
    "Ai14+Light_sample16_cell_centroids_min6_s100_LRavg_sub_neg_z_scrambled.nii.gz"
]

q_thresholds = [0.05, 0.10, 0.15, 0.20]
min_cluster_size = 100

# ==========================================
# 2. DEFINE ANATOMICAL GROUPS
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

def is_visual_area(abbrev):
    abbrev = str(abbrev)
    if abbrev.startswith('VIS') and not abbrev.startswith('VISC'): return True
    if abbrev.startswith('LGd') or abbrev.startswith('LGv') or abbrev == 'IGL': return True
    if abbrev == 'SC' or abbrev.startswith(('SCm', 'SCs', 'SCig', 'SCop', 'SCsg', 'SCzo', 'SCdw', 'SCiw', 'SCdg')): return True
    return False

# ==========================================
# 3. LOAD DATA & BUILD STRICT MASK
# ==========================================
print("Loading 3D Brain Volumes...")
def load_maps(file_list):
    arrays = []
    for f in file_list:
        path = os.path.join(data_dir, f)
        img = nib.load(path)
        arrays.append(np.asanyarray(img.dataobj).squeeze())
    return np.stack(arrays, axis=-1)

orig_data = load_maps(orig_files)
scram_data = load_maps(scram_files)
brain_shape = orig_data.shape[:-1]

print("Building strict anatomical mask from reference dataset...")
ref_df = pd.read_parquet(ref_file, columns=['x', 'y', 'z', 'region_y', 'region_abbrev'])

# EXTREMELY CRITICAL: Harmonize PTL -> PTLp BEFORE applying get_base_group to ensure PTLp gets masked!
ref_df['region_y'] = ref_df['region_y'].replace('PTL', 'PTLp')
ref_df['region_abbrev'] = ref_df['region_abbrev'].replace('PTL', 'PTLp')
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

# ==========================================
# 4. VECTORIZED PITMAN-MORGAN TEST
# ==========================================
print(f"Executing Two-Tailed Pitman-Morgan Test on {X.shape[0]} valid voxels...")
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

print("Calculating raw voxel-wise Log2SD Ratios...")
std_orig = np.std(X, axis=1)
std_scram = np.std(Y, axis=1)
log2_sd_ratio = np.log2(std_scram / std_orig)

results_df = pd.DataFrame({
    'x': brain_indices[2], 
    'y': brain_indices[1], 
    'z': brain_indices[0], 
    'log2_sd_ratio': log2_sd_ratio
})

# ==========================================
# 5. FDR CORRECTION & 3D CLUSTERING
# ==========================================
print("Applying FDR Correction...")
_, q_vals, _, _ = mt.multipletests(p_vals, alpha=0.20, method='fdr_bh')

q_map_3d = np.ones(brain_shape)
q_map_3d[brain_indices] = q_vals

r_map_3d = np.zeros(brain_shape)
r_map_3d[brain_indices] = r

struct = generate_binary_structure(3, 3)
cluster_masks_memory = {}

def cluster_and_save(sig_mask, q_col, out_name):
    labeled_array, num_features = label(sig_mask, structure=struct)
    final_cluster_map = np.zeros_like(labeled_array, dtype=np.uint8)
    
    if num_features > 0:
        cluster_sizes = np.bincount(labeled_array.ravel())
        valid_clusters = np.where(cluster_sizes >= min_cluster_size)[0]
        valid_clusters = valid_clusters[valid_clusters > 0]
        
        current_id = 1
        for cid in valid_clusters:
            final_cluster_map[labeled_array == cid] = current_id
            current_id += 1
            
    if final_cluster_map.max() > 0:
        z, y, x = np.nonzero(final_cluster_map)
        df_out = pd.DataFrame({'x': x, 'y': y, 'z': z, 'cluster_id': final_cluster_map[z, y, x]})
        
        df_out.to_parquet(out_name, index=False, engine="pyarrow")
        print(f"  -> Saved {len(df_out)} voxels to {os.path.basename(out_name)}")
        
        df_out[q_col + '_mask'] = True
        return df_out[['x', 'y', 'z', q_col + '_mask']]
    else:
        print(f"  -> No valid clusters for {os.path.basename(out_name)}")
        return pd.DataFrame(columns=['x', 'y', 'z', q_col + '_mask'])

for q_thresh in q_thresholds:
    q_str = str(q_thresh).replace('.', '')
    
    # S_q: Scrambled > Original (r < 0)
    sig_mask_scram = (q_map_3d < q_thresh) & (r_map_3d < 0) & brain_mask
    mask_df_s = cluster_and_save(sig_mask_scram, f'S_q{q_str}', os.path.join(out_dir, f"Ai14_ScramgtOG_q{q_str}.parquet"))
    cluster_masks_memory[f'S_q{q_str}'] = mask_df_s
    
    # O_q: Original > Scrambled (r > 0)
    sig_mask_orig = (q_map_3d < q_thresh) & (r_map_3d > 0) & brain_mask
    mask_df_o = cluster_and_save(sig_mask_orig, f'O_q{q_str}', os.path.join(out_dir, f"Ai14_OGgtScram_q{q_str}.parquet"))
    cluster_masks_memory[f'O_q{q_str}'] = mask_df_o

# ==========================================
# 6. TABLE GENERATOR: MERGING DATA
# ==========================================
print("\nPreparing Excel Export...")
df = valid_df.copy()
df['is_VS'] = df['region_abbrev'].apply(is_visual_area) | df['region_y'].apply(is_visual_area)

df = df.merge(results_df, on=['x', 'y', 'z'], how='left')

q_name_mapping = {
    '005': '05',
    '01': '10',
    '015': '15',
    '02': '20'
}

for raw_q_col, mask_df in cluster_masks_memory.items():
    prefix, suffix = raw_q_col.split('q')
    clean_q_col = f"{prefix}q{q_name_mapping[suffix]}"
    
    if not mask_df.empty:
        df = df.merge(mask_df, on=['x', 'y', 'z'], how='left')
        df[clean_q_col] = df[raw_q_col + '_mask'].fillna(False)
        df.drop(columns=[raw_q_col + '_mask'], inplace=True)
    else:
        df[clean_q_col] = False

df_condensed = pd.concat([df.copy().assign(Group_Label=df['Base_Group']), df[df['is_VS']].copy().assign(Group_Label='VS')])
df_iso = df[df['Base_Group'] == 'Isocortex'].copy()
df_iso['Layer'] = 'Layer ' + df_iso['region_abbrev'].str.extract(r'(1|2/3|4|5|6a|6b|6)$')[0].astype(str)
df_iso = df_iso.dropna(subset=['Layer'])

# ==========================================
# 7. MULTI-CUTOFF STATS AGGREGATION
# ==========================================
def calc_stats(data, group_col):
    data = data.copy()
    grouped = data.groupby(group_col).agg(
        Total_Voxels=('x', 'size'),
        Avg_Log2SD=('log2_sd_ratio', 'mean'),
        S_q05_sig=('S_q05', 'sum'), S_q10_sig=('S_q10', 'sum'), S_q15_sig=('S_q15', 'sum'), S_q20_sig=('S_q20', 'sum'),
        O_q05_sig=('O_q05', 'sum'), O_q10_sig=('O_q10', 'sum'), O_q15_sig=('O_q15', 'sum'), O_q20_sig=('O_q20', 'sum')
    ).reset_index()
    
    pct = lambda x: (x / grouped['Total_Voxels']) * 100
    cleaned = pd.DataFrame({
        'Region': grouped[group_col],
        "Avg. Log2SD (Scram/Orig)": grouped['Avg_Log2SD'],
        '% Sig Scram > Orig (q < 0.05)': pct(grouped['S_q05_sig']), 
        '% Sig Scram > Orig (q < 0.10)': pct(grouped['S_q10_sig']),
        '% Sig Scram > Orig (q < 0.15)': pct(grouped['S_q15_sig']), 
        '% Sig Scram > Orig (q < 0.20)': pct(grouped['S_q20_sig']),
        '% Sig Orig > Scram (q < 0.05)': pct(grouped['O_q05_sig']), 
        '% Sig Orig > Scram (q < 0.10)': pct(grouped['O_q10_sig']),
        '% Sig Orig > Scram (q < 0.15)': pct(grouped['O_q15_sig']), 
        '% Sig Orig > Scram (q < 0.20)': pct(grouped['O_q20_sig'])
    })
    return cleaned.sort_values(by="Avg. Log2SD (Scram/Orig)", ascending=False).round(3)

table1_df = calc_stats(df, 'region_y')                               
table2_df = calc_stats(df[df['Base_Group'] == 'Isocortex'], 'region_abbrev') 
table3_df = calc_stats(df[df['region_abbrev'] != 'OLF'], 'region_abbrev') 
table4_df = calc_stats(df_condensed, 'Group_Label')
table5_df = calc_stats(df_iso, 'Layer')

# ==========================================
# 8. PROFESSIONAL EXCEL EXPORT
# ==========================================
ccf_df = pd.read_csv(csv_info_file)
def rgb_to_hex(r, g, b):
    try: return f"#{int(r):02X}{int(g):02X}{int(b):02X}"
    except: return "#FFFFFF" 
ccf_df['hex_color'] = ccf_df.apply(lambda row: rgb_to_hex(row['R'], row['G'], row['B']), axis=1)
exact_color_map = dict(zip(ccf_df['abbreviation'], ccf_df['hex_color']))

allen_base_colors = {
    'Isocortex': '#08858C', 'HPF': '#7ED04B', 'CTXsp': '#8DD3D7', 
    'CNU': '#98D6F9', 'IB': '#FF7080', 'MB': '#FF64FF', 'VS': '#6A0DAD', 'OLF': '#92C592'
}

POS_COLOR = '#A6A6A6' 
NEG_COLOR = '#595959'

excel_writer = pd.ExcelWriter(out_excel_master, engine='xlsxwriter')
workbook = excel_writer.book
bounds_format = workbook.add_format({'font_color': '#000000', 'align': 'right', 'valign': 'vcenter'})

def write_formatted_outputs(df_table, sheet_name, table_type='standard'):
    df_table.to_excel(excel_writer, sheet_name=sheet_name, index=False, header=False, startrow=2)
    worksheet = excel_writer.sheets[sheet_name]
    
    header_format = workbook.add_format({'bold': True, 'bg_color': '#f2f2f2', 'font_color': '#000000', 'border': 1, 'align': 'center', 'valign': 'vcenter', 'text_wrap': True})
    bold_only = workbook.add_format({'bold': True, 'font_color': '#000000'})
    subscript_format = workbook.add_format({'bold': True, 'font_color': '#000000', 'font_script': 2}) 
    
    worksheet.merge_range(0, 0, 1, 0, "Region", header_format)
    worksheet.merge_range(0, 1, 1, 1, "", header_format)
    worksheet.write_rich_string(0, 1, bold_only, "Avg. Log", subscript_format, "2", bold_only, " St. Dev. Ratio (Scram/Orig)", header_format)
    
    worksheet.merge_range(0, 2, 0, 5, "% Significant Scram > Orig", header_format)
    worksheet.merge_range(0, 6, 0, 9, "% Significant Orig > Scram", header_format)
    
    q_labels = ["q < 0.05", "q < 0.10", "q < 0.15", "q < 0.20"]
    for i, label in enumerate(q_labels):
        worksheet.write(1, 2 + i, label, header_format)
        worksheet.write(1, 6 + i, label, header_format)

    for row_idx, (_, row) in enumerate(df_table.iterrows()):
        row_num = row_idx + 2 
        region = row['Region']
        
        if table_type == 'condensed': base_color = allen_base_colors.get(region, '#D3D3D3'); is_vs = (region == 'VS')
        elif table_type == 'layer': base_color = allen_base_colors['Isocortex']; is_vs = False
        else: base_color = exact_color_map.get(region, '#D3D3D3'); is_vs = is_visual_area(region)

        format_region = workbook.add_format({'bg_color': base_color, 'font_color': '#FFFFFF', 'bold': True, 'border': 1, 'align': 'center', 'valign': 'vcenter'})
        format_data = workbook.add_format({'bg_color': base_color, 'font_color': '#FFFFFF', 'bold': True, 'border': 1, 'align': 'right', 'valign': 'vcenter'})
        
        if is_vs:
            format_region.set_pattern(6); format_region.set_fg_color('#6A0DAD')
            format_data.set_pattern(6); format_data.set_fg_color('#6A0DAD')

        for col_num in range(len(df_table.columns)):
            val = row.iloc[col_num]
            cell_fmt = format_region if col_num == 0 else format_data
            if isinstance(val, (int, float)): worksheet.write_number(row_num, col_num, val, cell_fmt)
            else: worksheet.write(row_num, col_num, val, cell_fmt)
            
    end_row = len(df_table) + 2
    for col_idx in range(2, 10):
        worksheet.write_number(end_row, col_idx, 100, bounds_format)

    worksheet.set_column(0, 0, 18.33)    
    worksheet.set_column(1, 1, 18.33)   
    worksheet.set_column(2, 9, 8.00)    
    
    worksheet.conditional_format(2, 1, 499, 1, {'type': 'data_bar', 'bar_solid': True, 'bar_color': POS_COLOR, 'bar_negative_color': NEG_COLOR, 'bar_no_border': True})
    for col_idx in range(2, 6):
        worksheet.conditional_format(2, col_idx, 499, col_idx, {'type': 'data_bar', 'bar_solid': True, 'bar_color': POS_COLOR, 'min_type': 'num', 'min_value': 0, 'max_type': 'num', 'max_value': 100, 'bar_no_border': True})
    for col_idx in range(6, 10):
        worksheet.conditional_format(2, col_idx, 499, col_idx, {'type': 'data_bar', 'bar_solid': True, 'bar_color': NEG_COLOR, 'min_type': 'num', 'min_value': 0, 'max_type': 'num', 'max_value': 100, 'bar_no_border': True})

write_formatted_outputs(table4_df, 'Condensed_Table', 'condensed')
write_formatted_outputs(table3_df, 'All_Subregions_Table', 'standard')
write_formatted_outputs(table2_df, 'Isocortex_Table', 'standard')
write_formatted_outputs(table5_df, 'Layers_Table', 'layer')
write_formatted_outputs(table1_df, 'RegionY_Table', 'standard')

excel_writer.close()
print(f"Success! Ai14 Pipeline Complete. Master Excel saved to: {out_excel_master}")