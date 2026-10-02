# Cohen's d master table generator: Ai34 @ multiple q-value cutoffs from .nii.gz -> .parquet map conversions

import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import os

# ==========================================
# 1. FILE PATHS & SETUP
# ==========================================
input_file = r"..\data\corrected_df34_full.parquet"
csv_info_file = r"..\data\CCFv3-2020_info_oliver_voxel_info.csv" 

out_table1 = r"..\output\Ai34_Table1_RegionY_Allen.html"
out_table2 = r"..\output\Ai34_Table2_Isocortex_Allen.html"
out_table3 = r"..\output\Ai34_Table3_All_Subregions_Allen.html"
out_table4 = r"..\output\Ai34_Table4_Condensed_Groups.html"
out_table5 = r"..\output\Ai34_Table5_Cortical_Layers.html"
out_excel_master = r"..\output\Ai34_Master_Tables_Formatted.xlsx"
out_csv1 = r"..\output\Ai34_Mouse_Zscores_Condensed.csv"
out_csv2 = r"..\output\Ai34_Mouse_Zscores_Layers.csv"

# ==========================================
# 2. SIGNIFICANCE CLUSTER MASKS (Q-VALUE CUTOFFS)
# ==========================================
CLUSTER_FILES = {
    'L_q05': r"..\output\Ai34_LgtD_q05.parquet",
    'L_q10': r"..\output\Ai34_LgtD_q10.parquet",
    'L_q15': r"..\output\Ai34_LgtD_q15.parquet",
    'L_q20': r"..\output\Ai34_LgtD_q02.parquet",
    
    'D_q05': r"..\output\Ai34_DgtL_q05.parquet",
    'D_q10': r"..\output\Ai34_DgtL_q10.parquet",
    'D_q15': r"..\output\Ai34_DgtL_q15.parquet",
    'D_q20': r"..\output\Ai34_DgtL_q02.parquet"
}

# ==========================================
# 3. DYNAMIC COLUMN LOADING & COHEN'S D
# ==========================================
print("Loading Parquet data...")
all_cols = pq.ParquetFile(input_file).schema.names
mouse_cols = [c for c in all_cols if ('+Light_' in c or '+Dark_' in c) and 'mean' not in c.lower() and 'var' not in c.lower()]

columns_to_load = [
    'x', 'y', 'z', 'region_y', 'region_abbrev', 'Ai34+Dark_mean', 'Ai34+Light_mean', 
    'Ai34+Dark_var', 'Ai34+Light_var', 'L_gt_D_1-p', 'D_gt_L_1-p'
] + mouse_cols

df = pd.read_parquet(input_file, columns=columns_to_load)

# Strictly isolate the right hemisphere (x = 190 to 379)
df = df[(df['x'] >= 190) & (df['x'] <= 379)]
df = df.dropna(subset=['L_gt_D_1-p', 'D_gt_L_1-p'])

pooled_var = (df['Ai34+Dark_var'] + df['Ai34+Light_var']) / 2
pooled_std = np.sqrt(pooled_var + 1e-8) 
df['Cohens_d_Light'] = (df['Ai34+Light_mean'] - df['Ai34+Dark_mean']) / pooled_std

df['region_y'] = df['region_y'].replace('PTL', 'PTLp')
df['region_abbrev'] = df['region_abbrev'].replace('PTL', 'PTLp')

# ==========================================
# 4. APPLYING SPATIAL CLUSTER MASKS
# ==========================================
print("Applying significance cluster masks from parquet files...")
for q_col, filepath in CLUSTER_FILES.items():
    if os.path.exists(filepath):
        cluster_col = 'L_gt_D_clusters' if q_col.startswith('L') else 'D_gt_L_clusters'
        mask_df = pd.read_parquet(filepath, columns=['x', 'y', 'z', cluster_col])
        mask_df[q_col + '_mask'] = mask_df[cluster_col] > 0
        df = df.merge(mask_df[['x', 'y', 'z', q_col + '_mask']], on=['x', 'y', 'z'], how='left')
        df[q_col] = df[q_col + '_mask'].fillna(False)
        df.drop(columns=[q_col + '_mask'], inplace=True)
        print(f"  -> Successfully mapped {q_col} from {os.path.basename(filepath)}")
    else:
        df[q_col] = False

# ==========================================
# 5. STRICT CCF MAPPING & COLORS
# ==========================================
ccf_df = pd.read_csv(csv_info_file)
def rgb_to_hex(r, g, b):
    try: return f"#{int(r):02X}{int(g):02X}{int(b):02X}"
    except: return "#FFFFFF" 
ccf_df['hex_color'] = ccf_df.apply(lambda row: rgb_to_hex(row['R'], row['G'], row['B']), axis=1)
exact_color_map = dict(zip(ccf_df['abbreviation'], ccf_df['hex_color']))

allen_base_colors = {
    'Isocortex': '#08858C', 'HPF': '#7ED04B', 'CTXsp': '#8DD3D7', 
    'CNU': '#98D6F9', 'IB': '#FF7080', 'MB': '#FF64FF', 'VS': '#6A0DAD',
    'OLF': '#92C592'
}

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

df['Base_Group'] = df['region_y'].apply(get_base_group)
df['is_VS'] = df['region_abbrev'].apply(is_visual_area) | df['region_y'].apply(is_visual_area)
df = df[~df['region_abbrev'].astype(str).str.startswith(('MOB', 'AOB'))]
df = df[df['Base_Group'] != 'Exclude'].copy()

# ==========================================
# 6. EXPANDED MULTI-CUTOFF STATS ENGINE
# ==========================================
def calc_stats(data, group_col):
    data = data.copy()
    grouped = data.groupby(group_col).agg(
        Avg_Cohens_d=('Cohens_d_Light', 'mean'),
        Total_Voxels=('Cohens_d_Light', 'size'),
        L_q05_sig=('L_q05', 'sum'), L_q10_sig=('L_q10', 'sum'), L_q15_sig=('L_q15', 'sum'), L_q20_sig=('L_q20', 'sum'),
        D_q05_sig=('D_q05', 'sum'), D_q10_sig=('D_q10', 'sum'), D_q15_sig=('D_q15', 'sum'), D_q20_sig=('D_q20', 'sum')
    ).reset_index()
    
    pct = lambda x: (x / grouped['Total_Voxels']) * 100
    cleaned = pd.DataFrame({
        'Region': grouped[group_col],
        "Average Cohen's d": grouped['Avg_Cohens_d'],
        '% Sig L>D (q < 0.05)': pct(grouped['L_q05_sig']), '% Sig L>D (q < 0.10)': pct(grouped['L_q10_sig']),
        '% Sig L>D (q < 0.15)': pct(grouped['L_q15_sig']), '% Sig L>D (q < 0.20)': pct(grouped['L_q20_sig']),
        '% Sig D>L (q < 0.05)': pct(grouped['D_q05_sig']), '% Sig D>L (q < 0.10)': pct(grouped['D_q10_sig']),
        '% Sig D>L (q < 0.15)': pct(grouped['D_q15_sig']), '% Sig D>L (q < 0.20)': pct(grouped['D_q20_sig'])
    })
    return cleaned.sort_values(by="Average Cohen's d", ascending=False).round(3)

print("Calculating formatted stats...")
table1_df = calc_stats(df, 'region_y')                               
table2_df = calc_stats(df[df['Base_Group'] == 'Isocortex'], 'region_abbrev') 
table3_df = calc_stats(df[df['region_abbrev'] != 'OLF'], 'region_abbrev') 

df_condensed = pd.concat([df.copy().assign(Group_Label=df['Base_Group']), df[df['is_VS']].copy().assign(Group_Label='VS')])
table4_df = calc_stats(df_condensed, 'Group_Label')

df_iso = df[df['Base_Group'] == 'Isocortex'].copy()
df_iso['Layer'] = 'Layer ' + df_iso['region_abbrev'].str.extract(r'(1|2/3|4|5|6a|6b|6)$')[0].astype(str)
df_iso = df_iso.dropna(subset=['Layer'])
table5_df = calc_stats(df_iso, 'Layer')

# ==========================================
# 7. PROFESSIONAL HTML AND EXCEL ENGINE
# ==========================================
print("Exporting Professional HTML and Excel files...")

POS_COLOR = '#A6A6A6' 
NEG_COLOR = '#595959' 

pub_styles = [
    {'selector': 'table', 'props': 'border-collapse: collapse; font-family: Arial, sans-serif; font-size: 14px; margin: 25px 0; border: 2px solid black;'},
    {'selector': 'th', 'props': 'background-color: #f2f2f2; color: black; font-weight: bold; padding: 12px; border: 1px solid black; text-align: center; font-size: 15px;'},
    {'selector': 'td', 'props': 'padding: 10px; border: 1px solid #ddd; text-align: center;'}
]

fmt_dict = {"Average Cohen's d": "{:.3f}"}
for q in ['0.05', '0.10', '0.15', '0.20']:
    fmt_dict[f'% Sig L>D (q < {q})'] = "{:.3f}"
    fmt_dict[f'% Sig D>L (q < {q})'] = "{:.3f}"

excel_writer = pd.ExcelWriter(out_excel_master, engine='xlsxwriter')
workbook = excel_writer.book
bounds_format = workbook.add_format({'font_color': '#000000', 'align': 'center', 'valign': 'vcenter'})

def write_formatted_outputs(df_table, html_out, sheet_name, table_type='standard'):
    def html_color(row):
        region = row['Region']
        if table_type == 'condensed': base_color = allen_base_colors.get(region, '#D3D3D3'); is_vs = (region == 'VS')
        elif table_type == 'layer': base_color = allen_base_colors['Isocortex']; is_vs = False
        else: base_color = exact_color_map.get(region, '#D3D3D3'); is_vs = is_visual_area(region)
            
        r, g, b = tuple(int(base_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
        lum = (0.299*r + 0.587*g + 0.114*b)/255
        txt = f"color: {'black' if lum > 0.5 else 'white'}; font-weight: bold; text-shadow: 1px 1px 1px {'white' if lum > 0.5 else 'black'};"
        bg = f"background: repeating-linear-gradient(90deg, {base_color}, {base_color} 6px, #6A0DAD 6px, #6A0DAD 12px);" if is_vs else f"background-color: {base_color};"
        return [f'{bg} {txt}'] * len(row)

    styled_df = df_table.style.hide(axis='index').set_table_styles(pub_styles).apply(html_color, axis=1).format(fmt_dict)
    styled_df.to_html(html_out)

    df_table.to_excel(excel_writer, sheet_name=sheet_name, index=False, header=False, startrow=2)
    worksheet = excel_writer.sheets[sheet_name]
    
    header_format = workbook.add_format({'bold': True, 'bg_color': '#f2f2f2', 'font_color': '#000000', 'border': 1, 'align': 'center', 'valign': 'vcenter', 'text_wrap': True})
    bold_only = workbook.add_format({'bold': True, 'font_color': '#000000'})
    bold_italic = workbook.add_format({'bold': True, 'italic': True, 'font_color': '#000000'})
    
    worksheet.merge_range(0, 0, 1, 0, "Region", header_format)
    worksheet.merge_range(0, 1, 1, 1, "", header_format)
    worksheet.write_rich_string(0, 1, bold_only, "Average Cohen's ", bold_italic, "d", header_format)
    
    worksheet.merge_range(0, 2, 0, 5, "% Significant Light > Dark", header_format)
    worksheet.merge_range(0, 6, 0, 9, "% Significant Dark > Light", header_format)
    
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

        if is_vs:
            cell_format = workbook.add_format({'bg_color': base_color, 'fg_color': '#6A0DAD', 'pattern': 6, 'font_color': '#FFFFFF', 'bold': True, 'border': 1, 'align': 'center', 'valign': 'vcenter'})
        else:
            cell_format = workbook.add_format({'bg_color': base_color, 'font_color': '#FFFFFF', 'bold': True, 'border': 1, 'align': 'center', 'valign': 'vcenter'})

        for col_num in range(len(df_table.columns)):
            val = row.iloc[col_num]
            if isinstance(val, (int, float)): worksheet.write_number(row_num, col_num, val, cell_format)
            else: worksheet.write(row_num, col_num, val, cell_format)
            
    end_row = len(df_table) + 2
    for col_idx in range(2, 10):
        worksheet.write_number(end_row, col_idx, 100, bounds_format)

    worksheet.set_column(0, 0, 8.33)    
    worksheet.set_column(1, 1, 18.33)   
    worksheet.set_column(2, 9, 8.00)    
    
    worksheet.conditional_format(2, 1, 499, 1, {
        'type': 'data_bar', 'bar_solid': True, 
        'bar_color': POS_COLOR, 'bar_negative_color': NEG_COLOR, 'bar_no_border': True
    })
    for col_idx in range(2, 6):
        worksheet.conditional_format(2, col_idx, 499, col_idx, {
            'type': 'data_bar', 'bar_solid': True, 'bar_color': POS_COLOR,
            'min_type': 'num', 'min_value': 0, 'max_type': 'num', 'max_value': 100, 'bar_no_border': True
        })
    for col_idx in range(6, 10):
        worksheet.conditional_format(2, col_idx, 499, col_idx, {
            'type': 'data_bar', 'bar_solid': True, 'bar_color': NEG_COLOR,
            'min_type': 'num', 'min_value': 0, 'max_type': 'num', 'max_value': 100, 'bar_no_border': True
        })

write_formatted_outputs(table4_df, out_table4, 'Condensed_Table', 'condensed')
write_formatted_outputs(table3_df, out_table3, 'All_Subregions_Table', 'standard')
write_formatted_outputs(table2_df, out_table2, 'Isocortex_Table', 'standard')
write_formatted_outputs(table5_df, out_table5, 'Layers_Table', 'layer')
write_formatted_outputs(table1_df, out_table1, 'RegionY_Table', 'standard')

excel_writer.close()

mouse_cond_export = df_condensed.groupby('Group_Label')[mouse_cols].mean().round(3)
mouse_cond_export.index.name = 'Region'
mouse_cond_export.to_csv(out_csv1)

mouse_layer_export = df_iso.groupby('Layer')[mouse_cols].mean().round(3)
mouse_layer_export.index.name = 'Region'
mouse_layer_export.to_csv(out_csv2)

print(f"Success! Master formatted Excel saved to: {out_excel_master}")