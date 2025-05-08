import os

from sklearn.model_selection import train_test_split

os.environ["PL_DISABLE_SLURM"] = "1"
for var in ["SLURM_NTASKS", "SLURM_NPROCS", "SLURM_STEP_NUM_TASKS"]:
    os.environ.pop(var, None)
    
import argparse
import pickle
from viz_callback import PlotAndEmbed
import scipy.sparse as sp
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import torch
torch.set_float32_matmul_precision('medium')
from contrastive_model import ContrastiveModel
import pytorch_lightning as pl

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default="data/tabula_muris/preprocessed/tm_adata_train.pkl")
    parser.add_argument("--val_path", type=str, default="data/tabula_muris/preprocessed/tm_adata_test.pkl")
    parser.add_argument("--all_path", type=str, default="data/tabula_muris/preprocessed/tm_adata_all.pkl")
    parser.add_argument("--randomsplit", action="store_true")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--cls_token", action="store_true")
    parser.add_argument("--multiply_by_expr", action="store_true")
    parser.add_argument("--outdir", type=str, default="models/contrastive/")
    parser.add_argument("--nworkers", type=int, default=1)
    return parser.parse_args()

def load_data(path):
    with open(path, "rb") as f:
        adata = pickle.load(f)
    X = adata.X.toarray() if sp.issparse(adata.X) else adata.X
    domains = adata.obs["tech"].astype("category").cat.codes.values
    cells = adata.obs["cell_ontology_class"].astype("category").cat.codes.values
    scaler = StandardScaler(with_mean=False)
    X = scaler.fit_transform(X).astype("float32")
    return X, domains, cells

def main(args):
    # load data
    if args.randomsplit:
        X, domains, cells = load_data(args.all_path)
        train_X, val_X, train_domains, val_domains, train_cells, val_cells = train_test_split(
        X, domains, cells, test_size=0.2, random_state=42, stratify=domains
        )
    else:
        train_X, train_domains, train_cells = load_data(args.train_path)
        val_X, val_domains, val_cells = load_data(args.val_path)

    train_loader = DataLoader(
                TensorDataset(
                    torch.tensor(train_X),
                    torch.tensor(train_domains, dtype=torch.long),
                    torch.tensor(train_cells, dtype=torch.long),
                ),
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.nworkers,
                pin_memory=True,
                persistent_workers=True
            )
    val_loader = DataLoader(
            TensorDataset(
                torch.tensor(val_X),
                torch.tensor(val_domains, dtype=torch.long),
                torch.tensor(val_cells, dtype=torch.long),
            ),
            batch_size=args.batch_size,
            num_workers=args.nworkers,
            pin_memory=True,
            persistent_workers=True
        )
        
    full_X = torch.cat((torch.tensor(train_X, dtype=torch.float32), torch.tensor(val_X, dtype=torch.float32)), dim=0)
    full_domains = torch.cat((torch.tensor(train_domains), torch.tensor(val_domains)), dim=0)
    os.makedirs(args.outdir, exist_ok=True)
    viz_cb = PlotAndEmbed(outdir=args.outdir, full_X=full_X, domain_labels=full_domains)

    model = ContrastiveModel(
        n_genes=train_X.shape[1],
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        lr=args.lr,
        cls_token=args.cls_token,
        multiply_by_expr=args.multiply_by_expr,
        temperature=args.temperature,
        outdir=args.outdir
    )
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu",
        devices=-1,
        callbacks=[viz_cb],
        log_every_n_steps=10,
        deterministic=True,
        enable_model_summary=False,
        precision="16-mixed",
        strategy="auto"
    )

    trainer.fit(model, train_loader, val_loader)

    trainer.save_checkpoint(args.outdir / f"val_loss{trainer.callback_metrics['val/loss'].item():.3f}.ckpt")

if __name__ == "__main__":
    args = parse_args()
    main(args)
