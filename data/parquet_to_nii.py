#!/usr/bin/env python3

"""
Use ``voxel_info.py`` from UNRAVEL to compute voxel statistics for a set of images.

Creates df with columns:
    - x, y, z, (voxel coordinates)
    - <group>_??, (column name: group name, sample number; values: intensities)
    - <group1>_mean, <group2>_mean, 
    - <group1>_var, var_<group2>_var, 
    - mask,
    - vox_p_tstat1, vox_p_tstat2,
    - rev_cluster_index_1, rev_cluster_index_2, ... (cluster IDs for each voxel)
    - region_id,
    - region (OLF, HPF, RHP, CTXsp, STR, PAL, TH, HY, MB, and one subdivision below ISO (VIS, AUD, ORB etc.))

Usage:
------
    voxel_info -i '<asterisk>.nii.gz' 
"""

import nibabel as nib
import numpy as np
import pandas as pd
from pathlib import Path
from rich import print
from rich.traceback import install

from unravel.core.img_io import load_nii
from unravel.core.help_formatter import RichArgumentParser, SuppressMetavar, SM
from unravel.core.config import Configuration
from unravel.core.utils import log_command, verbose_start_msg, verbose_end_msg


def parse_args():
    parser = RichArgumentParser(formatter_class=SuppressMetavar, add_help=False, docstring=__doc__)

    reqs = parser.add_argument_group('Required arguments')
    reqs.add_argument('-i', '--input', help='Output parquet file path.', required=True, action=SM)
    reqs.add_argument('-c', '--col', help='Column name for the intensities.', required=True, action=SM)
    reqs.add_argument('-r', '--ref_nii', help='Reference nii.gz path.', required=True, action=SM)
    reqs.add_argument('-o', '--output', help='Output img.nii.gz path.', required=True, action=SM)

    general = parser.add_argument_group('General arguments')
    general.add_argument('-v', '--verbose', help='Increase verbosity. Default: False', action='store_true', default=False)

    return parser.parse_args()


def parquet_to_img(parquet_path, img_shape, col_name):
    """Load the voxel coordinates from a CSV file.

    Parameters:
        - csv_path (str): the path to the CSV file
        - img_shape (tuple): the shape of the 3D ndarray to be created (z, y, x)
        - col_name (str): the name of the column containing the intensities

    Returns:
        - df (pd.DataFrame): the DataFrame containing the voxel coordinates
    """
    df = pd.read_parquet(parquet_path, engine="pyarrow", columns=['x', 'y', 'z', col_name])
    img = np.zeros(img_shape)
    img[df['z'], df['y'], df['x']] = df[col_name]
    return img


@log_command
def main():
    install()
    args = parse_args()
    Configuration.verbose = args.verbose
    verbose_start_msg()

    # Load the reference image
    ref_img = load_nii(args.ref_nii)
    
    img = parquet_to_img(args.input, ref_img.shape, args.col)

    # Save the image
    ref_nii = nib.load(args.ref_nii)
    nii_img = nib.Nifti1Image(img, affine=ref_nii.affine, header=ref_nii.header)
    nib.save(nii_img, str(args.output))

    verbose_end_msg()


if __name__ == '__main__':
    main()