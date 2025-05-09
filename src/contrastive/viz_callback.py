# viz_callback.py
from pytorch_lightning.callbacks import Callback
import matplotlib as mpl
mpl.use('Agg')
import torch, os, matplotlib.pyplot as plt, seaborn as sns
import scanpy as sc
mpl.rcParams['figure.autolayout'] = False
mpl.rcParams['savefig.bbox'] = 'tight'
mpl.rcParams['figure.dpi'] = 600
plt.rcParams["figure.figsize"] = (12, 8)


class PlotAndEmbed(Callback):
    """
    Collects epoch-level metrics; on_train_end() makes three PNG files:
        loss_curves.png, accuracy_curves.png, embeddings.png
    """
    def __init__(self, outdir: str, batch_size, data, x_data):
        super().__init__()
        self.outdir = outdir
        os.makedirs(outdir, exist_ok=True)

        # store full data for embeddings
        self.test_data = data
        self.x_data = x_data
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
        print('embedding...',flush=True)
        pl_module.eval()
        emb = []
        with torch.no_grad():
            for xb in torch.split(self.x_data, self.batch_size):
                xb = xb.to(pl_module.device)
                emb.append(pl_module.encoder(xb).cpu())
        emb = torch.cat(emb).numpy()
        print('calculating neighbors...',flush=True)
        adata = self.test_data.copy()
        adata.obsm['X_emb'] = emb
        sc.pp.neighbors(adata, use_rep="X_emb")
        print('calculating umap...',flush=True)
        sc.tl.umap(adata, n_components=2)
        fig3 = sc.pl.umap(adata, color='cell_ontology_class', size=10, show=False, return_fig=True)
        fig3.savefig(f"{self.outdir}/umap_cell_type.png")
        plt.close(fig3)
        fig4 = sc.pl.umap(adata, color='tech', size=20, show=False, return_fig=True)
        fig4.savefig(f"{self.outdir}/umap_tech.png")
        plt.close(fig4)
        fig5 = sc.pl.umap(adata, color='tissue', size=20, show=False, return_fig=True)
        fig5.savefig(f"{self.outdir}/umap_tissue.png")
        plt.close(fig5)
        fig6 = sc.pl.umap(adata, color='celltype_tech_availability', size=10, show=False, return_fig=True)
        fig6.savefig(f"{self.outdir}/umap_tech_availability.png")
        plt.close(fig6)
        print('done!',flush=True)
        