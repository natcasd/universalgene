import scanpy as sc
import pandas as pd
import numpy as np
import anndata as ad
import pickle
import os

print('reading files...')
tm_droplet_data = sc.read(
    r'data/tabula_muris/TM_droplet.h5ad',
)
tm_facs_data = sc.read(
    r'data/tabula_muris/TM_facs.h5ad',
)

print('read in files')

tm_droplet_data = tm_droplet_data[
    (~tm_droplet_data.obs.cell_ontology_class.isna())
].copy()
tm_facs_data = tm_facs_data[
    (~tm_facs_data.obs.cell_ontology_class.isna())
].copy()

# Add technology labels
tm_droplet_data.obs["tech"] = "10x"
tm_facs_data.obs["tech"] = "SS2"

print('reading gene length...')
gene_len = pd.read_csv(
    "https://raw.githubusercontent.com/chenlingantelope/HarmonizationSCANVI/master/data/gene_len.txt",
    delimiter=" ",
    header=None,
    index_col=0,
)
print('read gene length')

gene_len = gene_len.reindex(tm_facs_data.var.index).dropna()

tm_facs_data = tm_facs_data[:, gene_len.index].copy() # break the view

gene_len_vec = gene_len[1].values.astype(np.float32)
median_len = np.median(gene_len_vec)

print('scaling...')
# column‑wise scaling in CSC format
X = tm_facs_data.X.tocsc(copy=True) # -> (n_cells × n_genes)
X = X.multiply(1.0 / gene_len_vec) # divide each column by its length
X = X.multiply(median_len) # multiply by the median length
X.data = np.rint(X.data) # round only the non‑zero entries

tm_facs_data.X = X.tocsr() # store back as CSR (Scanpy’s default)

print('concatenating...')
tm_adata = ad.concat([tm_droplet_data, tm_facs_data])

tm_adata.obs['cell_ontology_class'].replace(
    to_replace='pancreatic ductal cel',
    value='pancreatic ductal cell',
    inplace=True
)

print('normalizing...')
tm_adata = ad.concat([tm_droplet_data, tm_facs_data])
print(tm_adata.shape)
tm_adata.layers["counts"] = tm_adata.X.copy()
sc.pp.normalize_total(tm_adata, target_sum=1e4)
sc.pp.log1p(tm_adata)
tm_adata.raw = tm_adata  # keep full dimension safe
sc.pp.highly_variable_genes(
    tm_adata,
    flavor="seurat_v3",
    n_top_genes=2000,
    layer="counts",
    batch_key="tech",
    subset=True,
)

print('splitting...')       
tm_droplet_data_tissues = set(tm_droplet_data.obs.tissue)
tm_facs_data_tissues = set(tm_facs_data.obs.tissue)
tm_all_tissues = tm_droplet_data_tissues | tm_facs_data_tissues

test_tissues={'Skin', 'Liver', 'Limb_Muscle', 'Pancreas'}
train_tissues = tm_all_tissues.difference(test_tissues)

tm_adata_train = tm_adata[
    tm_adata.obs['tissue'].isin(train_tissues)
]
tm_adata_test = tm_adata[
    tm_adata.obs['tissue'].isin(test_tissues)
]

os.makedirs(r'data/tabula_muris/preprocessed', exist_ok=True)
with open(r'data/tabula_muris/preprocessed/tm_adata_train.pkl', 'wb') as f:
    pickle.dump(tm_adata_train, f)
with open(r'data/tabula_muris/preprocessed/tm_adata_test.pkl', 'wb') as f:
    pickle.dump(tm_adata_test, f)
with open(r'data/tabula_muris/preprocessed/tm_adata_all.pkl', 'wb') as f:
    pickle.dump(tm_adata, f)

print('done')