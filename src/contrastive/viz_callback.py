# viz_callback.py
from pytorch_lightning.callbacks import Callback
import matplotlib as mpl
mpl.use('Agg')
import torch, os, matplotlib.pyplot as plt, seaborn as sns
import scanpy as sc
from threadpoolctl import threadpool_limits
mpl.rcParams['figure.autolayout'] = False
mpl.rcParams['savefig.bbox'] = 'tight'
mpl.rcParams['figure.dpi'] = 600
plt.rcParams["figure.figsize"] = (12, 8)


class PlotAndEmbed(Callback):
    """
    Collects epoch-level metrics; on_train_end() makes three PNG files:
        loss_curves.png, accuracy_curves.png, embeddings.png
    """
    def __init__(self, outdir: str,
                 full_X: torch.Tensor,
                 domain_labels, batch_size, full_data):
        super().__init__()
        self.outdir = outdir
        os.makedirs(outdir, exist_ok=True)

        # store full data for embeddings
        self.full_X = full_X
        self.full_data = full_data
        self.domain_labels = domain_labels

        # will be filled during training
        self.train_loss, self.val_loss = [], []
        self.batch_size = batch_size

    # ---- one value per epoch ----
    def on_train_epoch_end(self, trainer, pl_module):
        self.train_loss.append( trainer.callback_metrics["train/loss"].item() )

    def on_validation_epoch_end(self, trainer, pl_module):
        self.val_loss.append( trainer.callback_metrics["val/loss"].item() )

    # ---- after the very last epoch ----
    def on_train_end(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return
        # 1. loss curves
        plt.figure(figsize=(6,4))
        plt.plot(self.train_loss, label="train")
        plt.plot(self.val_loss,   label="val")
        plt.xlabel("epoch"); plt.ylabel("loss"); plt.title("Loss curves"); plt.legend()
        plt.tight_layout();  plt.savefig(f"{self.outdir}/loss_curves.png"); plt.close()
        
        # 3. embeddings (first two dims)
        pl_module.eval()
        emb = []
        with torch.no_grad():
            for xb in torch.split(self.full_X, self.batch_size):
                xb = xb.to(pl_module.device)
                emb.append(pl_module.encoder(xb).cpu())
        emb = torch.cat(emb).numpy()

        plt.figure(figsize=(7,5))
        sns.scatterplot(x=emb[:,0], y=emb[:,1],
                        hue=self.domain_labels, palette="tab10", s=5, linewidth=0)
        plt.title("Encoder embeddings (dim 0 vs 1)")
        plt.legend(bbox_to_anchor=(1.04,1), borderaxespad=0)
        plt.tight_layout()
        plt.savefig(f"{self.outdir}/embeddings.png")
        plt.close()

        adata = self.full_data.copy()
        adata.obsm['X_emb'] = emb
        print("computing neighbours", flush=True)
        with threadpool_limits(limits=1):             
            sc.pp.neighbors(adata, use_rep="X_emb")
        print("neighbours finished", flush=True)
        print("computing umap", flush=True)
        sc.tl.umap(adata, n_components=2)
        print("umap finished", flush=True)
        fig3 = sc.pl.umap(adata, color='cell_ontology_class', show=False, return_fig=True)
        fig3.savefig(f"{self.outdir}/umap_cell_type.png")
        plt.close(fig3)
        print("umap cell type saved", flush=True)

        fig4 = sc.pl.umap(adata, color='tech', show=False, return_fig=True)
        fig4.savefig(f"{self.outdir}/umap_tech.png")
        plt.close(fig4)
        print("umap tech saved", flush=True)
        