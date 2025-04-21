# viz_callback.py
from pytorch_lightning.callbacks import Callback
import torch, os, matplotlib.pyplot as plt, seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
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
                 domain_labels):
        super().__init__()
        self.outdir = outdir
        os.makedirs(outdir, exist_ok=True)

        # store full data for embeddings
        self.full_X = full_X
        self.domain_labels = domain_labels

        # will be filled during training
        self.train_loss, self.val_loss = [], []
        self.train_acc,  self.val_acc  = [], []

    # ---- one value per epoch ----
    def on_train_epoch_end(self, trainer, pl_module):
        self.train_loss.append( trainer.callback_metrics["train/loss"].item() )
        self.train_acc .append( trainer.callback_metrics["train/cell_acc"].item() )

    def on_validation_epoch_end(self, trainer, pl_module):
        self.val_loss.append( trainer.callback_metrics["val/loss"].item() )
        self.val_acc .append( trainer.callback_metrics["val/cell_acc"].item() )


    # ---- after the very last epoch ----
    def on_train_end(self, trainer, pl_module):
        # 1. loss curves
        plt.figure(figsize=(6,4))
        plt.plot(self.train_loss, label="train")
        plt.plot(self.val_loss,   label="val")
        plt.xlabel("epoch"); plt.ylabel("loss"); plt.title("Loss curves"); plt.legend()
        plt.tight_layout();  plt.savefig(f"{self.outdir}/loss_curves.png"); plt.close()

        # 2. accuracy curves
        plt.figure(figsize=(6,4))
        plt.plot(self.train_acc, label="train cell-acc")
        plt.plot(self.val_acc,   label="val cell-acc")
        plt.xlabel("epoch"); plt.ylabel("accuracy"); plt.title("Accuracy curves"); plt.legend()
        plt.tight_layout();  plt.savefig(f"{self.outdir}/accuracy_curves.png"); plt.close()

        # 3. embeddings (first two dims)
        pl_module.eval()
        emb = []
        with torch.no_grad():
            for xb in torch.split(self.full_X.to(pl_module.device), 8192):
                emb.append( pl_module.encoder(xb).cpu() )
        emb = torch.cat(emb).numpy()

        plt.figure(figsize=(7,5))
        sns.scatterplot(x=emb[:,0], y=emb[:,1],
                        hue=self.domain_labels, palette="tab10", s=5, linewidth=0)
        plt.title("Encoder embeddings (dim 0 vs 1)")
        plt.legend(bbox_to_anchor=(1.04,1), borderaxespad=0)
        plt.tight_layout()
        plt.savefig(f"{self.outdir}/embeddings.png")
        plt.close()
