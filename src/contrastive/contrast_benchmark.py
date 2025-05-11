from train_contrastive import load_data

path = "../data/tabula_muris/preprocessed/tm_adata_test.pkl"
path_r = "../data/tabula_muris/preprocessed_reduced/tm_adata_test.pkl"

print("Loading data")
x, domains, cells, adata = load_data(path)
x_r, domains_r, cells_r, adata_r = load_data(path_r)

from contrastive_model import ContrastiveModel

CONTRASTIVE_TRANSFORMER_KEY="contrastive_transformer"
CONTRASTIVE_MLP_KEY="contrastive_mlp"

c_transformer_path = "runs/attention_05-10-23:02/final.ckpt"
c_mlp_path = "runs/dense_05-11-12:36/final.ckpt"


c_transformer = ContrastiveModel.load_from_checkpoint(c_transformer_path)
c_mlp = ContrastiveModel.load_from_checkpoint(c_mlp_path)

import torch
from tqdm import tqdm

batch_size = 32  # Define a suitable batch size
ct_embeddings_list = []
cm_embeddings_list = []

c_transformer.eval()
c_mlp.eval()

x = torch.tensor(x, dtype=torch.float32)
x_r = torch.tensor(x_r, dtype=torch.float32)


print("Embedding data")
with torch.no_grad():
    for i in tqdm(range(0, x.size(0), batch_size), desc="Processing Batches"):
        x_batch = x[i:i + batch_size]
        x_r_batch = x_r[i:i + batch_size]
        
        ct_embeddings_batch = c_transformer.encoder(x_r_batch)
        cm_embeddings_batch = c_mlp.encoder(x_batch)
        
        ct_embeddings_list.append(ct_embeddings_batch)
        cm_embeddings_list.append(cm_embeddings_batch)

ct_embeddings = torch.cat(ct_embeddings_list, dim=0)
cm_embeddings = torch.cat(cm_embeddings_list, dim=0)

adata.obsm[CONTRASTIVE_TRANSFORMER_KEY] = ct_embeddings.cpu().numpy()
adata.obsm[CONTRASTIVE_MLP_KEY] = cm_embeddings.cpu().numpy()

import scanpy as sc

print('neighbors and umap')
sc.pp.neighbors(adata, use_rep=CONTRASTIVE_TRANSFORMER_KEY)  # Compute neighbors using latent space
sc.tl.umap(adata)

sc.pp.neighbors(adata, use_rep=CONTRASTIVE_MLP_KEY)  # Compute neighbors using latent space
sc.tl.umap(adata)

print('benchmarking')
from scib_metrics.benchmark import Benchmarker

bm = Benchmarker(
    adata,
    batch_key='tech',
    label_key='cell_ontology_class',
    embedding_obsm_keys=[CONTRASTIVE_TRANSFORMER_KEY, CONTRASTIVE_MLP_KEY],
    n_jobs=-1
)
bm.benchmark()

import os
print('saving')
save_path = "benchmark_res"
os.makedirs(save_path, exist_ok=True)
bm.plot_results_table(save_dir=save_path)

