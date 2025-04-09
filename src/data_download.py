"""
Script for downloading gene expression data, per https://docs.scvi-tools.org/en/stable/tutorials/notebooks/scrna/tabula_muris.html.
""" 
import scanpy as sc
import os

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Define paths relative to the script location
save_dir = os.path.join(SCRIPT_DIR, "data")
tm_droplet_path = os.path.join(save_dir, "TM_droplet.h5ad")
tm_facs_path = os.path.join(save_dir, "TM_facs.h5ad")

# Create data directory if it doesn't exist
os.makedirs(save_dir, exist_ok=True)

# tm_droplet = sc.read(
#     tm_droplet_path,
#     backup_url="https://figshare.com/ndownloader/files/23938934",
# )
tm_facs = sc.read(
    tm_facs_path,
    backup_url="https://figshare.com/ndownloader/files/23939711",
)