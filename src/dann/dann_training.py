#!/usr/bin/env python
# dann_training.py

import argparse
import os
import torch, pickle, scipy.sparse as sp
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import pytorch_lightning as pl

from lightning_dann import DANNModule
from viz_callback import PlotAndEmbed

import os, sys
from pathlib import Path
script_dir = Path(__file__).resolve().parent
repo_dir = script_dir.parent
# src_dir = repo_dir / "src"
data_dir = repo_dir / "data"
# sys.path.append(str(src_dir))
sys.path.append(str(data_dir))


def main():
    # ─── 1 · Parse arguments ─────────────────────────────────────
    parser = argparse.ArgumentParser(
        description="Train a DANN on Tabula Muris data")
    parser.add_argument(
        "--epochs", "-e",
        type=int,
        default=1,
        help="number of training epochs")
    args = parser.parse_args()

    # ─── 2 · Data loading & prep ─────────────────────────────────
    data_path = data_dir / 'tabula_muris' / 'preprocessed' / 'tm_adata_train.pkl'
    with open('./src/data/tabula_muris/preprocessed/tm_adata_train.pkl', 'rb') as f:
    # with open(data_path, 'rb') as f:
        adata = pickle.load(f)

    X = adata.X.toarray() if sp.issparse(adata.X) else adata.X
    domains = adata.obs['tech'].astype('category').cat.codes.values
    cells   = adata.obs['cell_ontology_class'].astype('category').cat.codes.values
    num_cell_classes = len(np.unique(cells))

    scaler = StandardScaler(with_mean=False)
    X = scaler.fit_transform(X).astype('float32')

    X_train, X_val, d_train, d_val, c_train, c_val = train_test_split(
        X, domains, cells,
        test_size=0.2, random_state=42, stratify=domains)

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train),
                      torch.tensor(d_train, dtype=torch.long),
                      torch.tensor(c_train, dtype=torch.long)),
        batch_size=64, shuffle=True)
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val),
                      torch.tensor(d_val, dtype=torch.long),
                      torch.tensor(c_val, dtype=torch.long)),
        batch_size=64)

    # ─── 3 · Callback for figures ────────────────────────────────
    OUTPUT_DIR = "./src/dann/figures"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    full_X = torch.tensor(X, dtype=torch.float32)
    viz_cb = PlotAndEmbed(outdir=OUTPUT_DIR,
                          full_X=full_X,
                          domain_labels=domains)

    # ─── 4 · Model & Trainer ────────────────────────────────────
    model = DANNModule(input_dim=X.shape[1],
                       num_cell_classes=num_cell_classes,
                       hidden_dim=512,
                       lambda_domain=1.0,
                       lr=1e-4)

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu",
        devices=-1,
        callbacks=[viz_cb],
        log_every_n_steps=10,
        deterministic=True,
    )
    trainer.fit(model, train_loader, val_loader)

    # ─── 5 · Checkpoint ─────────────────────────────────────────
    ckpt_dir = repo_dir / "checkpoints"
    sys.path.append(str(ckpt_dir))
    os.makedirs(ckpt_dir, exist_ok=True)
    trainer.save_checkpoint(
        os.path.join(ckpt_dir, f"dann_train_tissue_epochs{args.epochs}.ckpt"))

if __name__ == "__main__":
    main()
