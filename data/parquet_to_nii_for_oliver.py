#!/usr/bin/env python3

"""
Use ``parquet_to_nii.py`` to convert a parquet file to a 3D .nii.gz image.

Notes:
    - The parquet file should contain the columns: x, y, z, col_name
    - x, y, z are the voxel coordinates
    - col_name is the column containing the intensities

Usage:
------
    parquet_to_nii -i path/to/parquet -c column_name -r path/to/ref_nii -o path/to/output_img.nii.gz
"""

import argparse
import nibabel as nib
import numpy as np
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('-i', '--input', help='Output parquet file path.', required=True)
    parser.add_argument('-c', '--col', help='Column name for the intensities.', required=True)
    parser.add_argument('-r', '--ref_nii', help='Reference nii.gz path.', required=True)
    parser.add_argument('-o', '--output', help='Output img.nii.gz path.', required=True)

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


def main():
    args = parse_args()

    # Load the reference image
    ref_nii = nib.load(args.ref_nii)
    ref_img = np.asanyarray(ref_nii.dataobj, dtype=ref_nii.header.get_data_dtype()).squeeze()
    
    # Convert the parquet column to a 3D image
    img = parquet_to_img(args.input, ref_img.shape, args.col)

    # Save the image
    nii_img = nib.Nifti1Image(img, affine=ref_nii.affine, header=ref_nii.header)
    nib.save(nii_img, str(args.output))


if __name__ == '__main__':
    main()