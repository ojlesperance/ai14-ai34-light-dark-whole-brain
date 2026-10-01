#Quad plot figure generator from .csv, Ai34

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
from scipy.stats import spearmanr
import re

# ==========================================
# 1. FILE PATHS & SETUP
# ==========================================
data_csv = r"C:\Users\ojles\Documents\Ai34_Project\Ai34_Quad_All_Subregions.csv"
ccf_csv = r"C:\Users\ojles\Documents\Ai34_Project\CCFv3-2020_info_oliver_voxel_info.csv"
output_dir = r"C:\Users\ojles\Documents\Ai34_Project\\" 
reporter = "Ai34" 

df = pd.read_csv(data_csv)
ccf_df = pd.read_csv(ccf_csv)

abbrev_to_parent = dict(zip(ccf_df['abbreviation'], ccf_df['region']))
df['Region'] = df['Region'].replace('PTL', 'PTLp')
abbrev_to_parent['PTLp'] = 'PTLp'

# ==========================================
# 2. ALLEN ATLAS MAPPING LOGIC 
# ==========================================
allen_colors = {
    'Isocortex': '#66C2A5', 'HPF': '#7ED04B', 'CTXsp': '#8DD3D7', 
    'CNU': '#98D6F9', 'IB': '#FF7080', 'MB': '#FF64FF', 'VS': '#B39DDB'         
}

def get_base_group(region_parent):
    if region_parent in ['PL', 'ILA', 'AI', 'PERI', 'ECT', 'ORB', 'VISC', 'VIS', 'TEa', 'MO', 'SS', 'ACA', 'RSP', 'AUD', 'PTLp', 'PTL', 'FRP', 'GU', 'ISO']: return 'Isocortex'
    if region_parent in ['ENT', 'PAR', 'HATA', 'SUB', 'ProS', 'PRE', 'POST', 'DG', 'CA3', 'CA2', 'CA1', 'IG', 'HPF', 'RHP', 'HIP', 'FC']: return 'HPF'
    if region_parent in ['EP', 'PA', 'CLA', 'BMA', 'BLA', 'LA', 'CTXsp']: return 'CTXsp'
    if region_parent in ['SI', 'MS', 'TRS', 'GP', 'BST', 'PAL', 'FS', 'OT', 'ACB', 'CP', 'MEA', 'AAA', 'IA', 'SH', 'SF', 'LS', 'STR']: return 'CNU'
    if region_parent in ['VPM', 'VPL', 'VM', 'VAL', 'SPA', 'SPF', 'LGd', 'MH', 'PT', 'PVT', 'CM', 'CL', 'PIL', 'IMD', 'SMT', 'MD', 'RT', 'AV', 'LD', 'LP', 'POL', 'PO', 'IGL', 'LGv', 'TH', 'PM', 'PH', 'VMH', 'MPN', 'TM', 'SUM', 'MM', 'TU', 'LHA', 'RCH', 'PVi', 'DMH', 'ADP', 'MEPO', 'SBPV', 'PV', 'PS', 'HY']: return 'IB'
    if region_parent in ['SNc', 'DR', 'IPDL', 'IPL', 'PPN', 'PBG', 'IC', 'SC', 'SAG', 'INC', 'Su', 'ND', 'PAG', 'VTN', 'VTA', 'CUN', 'PN', 'MT', 'SNr', 'MRN', 'NOT', 'RPF', 'APN', 'PPT', 'MB']: return 'MB'
    return 'Exclude'

def is_visual_area(abbrev, parent):
    abbrev, parent = str(abbrev), str(parent)
    if abbrev.startswith('VIS') and not abbrev.startswith('VISC'): return True
    if parent.startswith('VIS') and not parent.startswith('VISC'): return True
    if abbrev.startswith('LGd') or abbrev.startswith('LGv') or abbrev == 'IGL': return True
    if abbrev == 'SC' or abbrev.startswith(('SCm', 'SCs', 'SCig', 'SCop', 'SCsg', 'SCzo', 'SCdw', 'SCiw', 'SCdg')): return True
    return False

df['Parent'] = df['Region'].map(abbrev_to_parent).fillna(df['Region'])
df['Base_Group'] = df['Parent'].apply(get_base_group)
df['is_VS'] = df.apply(lambda row: is_visual_area(row['Region'], row['Parent']), axis=1)

df.loc[df['is_VS'] == True, 'Base_Group'] = 'VS'
df = df[~df['Region'].astype(str).str.startswith(('MOB', 'AOB'))]
df = df[df['Base_Group'] != 'Exclude'].copy()
df['Plot_Color'] = df['Base_Group'].map(allen_colors)

def format_pval(p):
    return "< 0.001" if p < 0.001 else f"= {p:.3f}"

def format_label(region):
    if '-' in region: return region.replace('-', '\n')
    match = re.search(r'(1|2/3|4|5|6a|6b|6)$', region)
    if match:
        layer = match.group(1)
        base = region[:-len(layer)]
        if base: return f"{base}\n{layer}"
    return region 

# ==========================================
# 3. EXACT PROPORTIONAL SIZING (Grid Target)
# ==========================================
scale_factor = 4
w_in = (58.66 / 25.4) * scale_factor  
h_in = (56.66 / 25.4) * scale_factor  

groups_to_plot = ['Isocortex', 'HPF', 'CTXsp', 'CNU', 'IB', 'MB', 'VS', 'Whole Brain']

for target_group in groups_to_plot:
    print(f"Generating high-res proportional plot for {target_group}...")
    
    fig, ax = plt.subplots(figsize=(w_in, h_in), dpi=600) 
    
    x_all = df['Avg_Cohens_d']
    y_all = df['Avg_Log2_Std_Ratio']
    
    if target_group == 'Whole Brain':
        target_mask = pd.Series([True]*len(df), index=df.index)
        group_color = df['Plot_Color'] 
    else:
        target_mask = df['Base_Group'] == target_group
        group_color = allen_colors[target_group]
        
    df_target = df[target_mask]
    
    x_min, x_max = df_target['Avg_Cohens_d'].min(), df_target['Avg_Cohens_d'].max()
    y_min, y_max = df_target['Avg_Log2_Std_Ratio'].min(), df_target['Avg_Log2_Std_Ratio'].max()
    
    x_pad = (x_max - x_min) * 0.15
    y_pad = (y_max - y_min) * 0.15
    if x_pad == 0: x_pad = 0.5
    if y_pad == 0: y_pad = 0.5
        
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)
    
    if target_group != 'Whole Brain':
        ax.scatter(x_all[~target_mask], y_all[~target_mask], color='#D3D3D3', s=50, alpha=0.4, edgecolors='none')
    
    dot_size = 60 if target_group == 'Whole Brain' else 180
    dot_alpha = 0.7 if target_group == 'Whole Brain' else 0.55
    ax.scatter(df_target['Avg_Cohens_d'], df_target['Avg_Log2_Std_Ratio'], color=group_color, s=dot_size, alpha=dot_alpha, edgecolors='white', linewidth=0.5)
    
    if len(df_target) > 2:
        rho, p_val = spearmanr(df_target['Avg_Cohens_d'], df_target['Avg_Log2_Std_Ratio'])
        slope, intercept = np.polyfit(df_target['Avg_Cohens_d'], df_target['Avg_Log2_Std_Ratio'], 1)
        
        x_vals = np.array(ax.get_xlim()) 
        y_vals = intercept + slope * x_vals
        
        line_color = 'black' if target_group == 'Whole Brain' else group_color
        ax.plot(x_vals, y_vals, color=line_color, linewidth=3.0, linestyle='-', zorder=1)
        
        stat_text = f"$\\rho$ = {rho:.3f}\n$p$ {format_pval(p_val)}"
    else:
        stat_text = "N/A"

    if target_group != 'Whole Brain':
        for _, row in df_target.iterrows():
            stacked_name = format_label(row['Region'])
            txt = ax.text(
                row['Avg_Cohens_d'], row['Avg_Log2_Std_Ratio'], stacked_name, 
                fontsize=8, fontweight='normal', color='black', 
                ha='center', va='center', 
                transform=ax.transData, zorder=10 
            )
            txt.set_path_effects([PathEffects.withStroke(linewidth=1.5, foreground='white', alpha=0.8)])

    if ax.get_xlim()[0] < 0 < ax.get_xlim()[1]:
        ax.axvline(0, color='black', linestyle=':', alpha=0.5, zorder=0, lw=1.5)
    if ax.get_ylim()[0] < 0 < ax.get_ylim()[1]:
        ax.axhline(0, color='black', linestyle=':', alpha=0.5, zorder=0, lw=1.5)
    
    ax.set_xlabel("Cohen's d", fontsize=16, fontweight='bold', labelpad=6)
    ax.set_ylabel("Log2 Std Ratio", fontsize=16, fontweight='bold', labelpad=6)
    ax.set_title(target_group, fontsize=18, fontweight='bold', pad=12)
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray', lw=1.0)
    ax.text(0.03, 0.97, stat_text, transform=ax.transAxes, fontsize=14, verticalalignment='top', bbox=props, zorder=5)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    filename = f"{output_dir}{reporter}_QuadPlot_{target_group.replace(' ', '_')}.png"
    plt.savefig(filename, dpi=600, bbox_inches='tight')
    plt.close()

print(f"Success! All 8 perfectly proportioned plots saved to {output_dir}")