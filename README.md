# ai14-ai34-light-dark-whole-brain
# Whole-Brain Microscale Stereotypy & Reorganization

This repository contains the data analysis and visualization code for the manuscript:
**"Microscale stereotypy and experience-dependent reorganization of cFos-tagged cells and their presynaptic terminals across the brain."**
*Oliver J. L’Esperance, Daniel Ryskamp Rijsketić, Nicholas Pritchett, Tristan Meier, Ethan Hunt, Boris D. Heifets, & Jaichandar Subramanian*

## Overview
This project provides a robust computational pipeline to analyze brain-wide experience-dependent functional reorganization. It processes `.nii.gz` functional maps and `.parquet` cell centroid data utilizing the UNRAVEL framework, generating voxel-wise Pitman-Morgan statistical maps, False Discovery Rate (FDR) corrections, and comprehensive formatted data tables.

## Repository Structure
* `src/`: Core utility modules for anatomical mapping definitions and atlas dictionaries.
* `scripts/`: Standalone Python scripts for running statistical engines and generating tables/plots.
* `data/`: Directory for input data (e.g., `CCFv3-2020_info.csv`, Parquet datasets). *Note: Raw neuroimaging data is hosted externally.*
* `output/`: Generated Excel tables, CSVs, and high-resolution Quad/Correlation plots.

## Data Availability
Due to file size limits, raw .nii.gz voluments maps and heavy .parquet files are hosted externally. Please download the dataset and place it in the `data/` directory before running the scripts.

## Installation
Ensure you have Python 3.9+ installed. Clone this repository and install the dependencies:
```bash
pip install -r requirements.txt
