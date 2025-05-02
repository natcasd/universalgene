import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics  # <-- NEW
from dann_modules import CDANHead, CellClassifier, MLPEncoder  # <- your existing code
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader


class CDANModule(pl.LightningModule):
    def __init__(
        self,
        input_dim: int,
        num_cell_classes: int,
        hidden_dim: int = 512,
        lambda_domain: float = 1.0,
        lambda_mmd: float = 0.5,
        lr: float = 1e-4,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.encoder = MLPEncoder(input_dim, hidden_dim)
        self.domain_clf = CDANHead(128, num_cell_classes)
        self.cell_clf = CellClassifier(128, num_cell_classes)

        # metrics you want to track
        self.train_acc_cell = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_cell_classes
        )
        self.train_acc_domain = torchmetrics.Accuracy(task="multiclass", num_classes=2)
        self.val_acc_cell = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_cell_classes
        )
        self.val_acc_domain = torchmetrics.Accuracy(task="multiclass", num_classes=2)

    def smoothed_nll(self, logp, y, ε=0.1):
        n_cls = logp.size(1)
        with torch.no_grad():
            true_dist = torch.full_like(logp, ε / (n_cls - 1))
            true_dist.scatter_(1, y.unsqueeze(1), 1 - ε)
        return torch.mean(torch.sum(-true_dist * logp, dim=1))

    def mmd_lin(self, f_s, f_t):
        """Linear MMD"""
        delta = f_s.mean(0) - f_t.mean(0)
        return delta.dot(delta)

    def λ_domain(self, ep, tot):
        base = 2 / (1 + np.exp(-10 * ep / tot)) - 1  # ramps 0→1
        tri = 0.5 * (1 + np.sin(2 * np.pi * ep / tot))  # triangle wave
        return base * tri

    # ---------- Forward ----------
    def forward(self, x):
        return self.encoder(x)

    # ---------- Training ----------
    def training_step(self, batch, batch_idx):
        x, d_lbl, c_lbl = batch
        d_lbl, c_lbl = d_lbl.long(), c_lbl.long()

        α = self.λ_domain(self.current_epoch, self.trainer.max_epochs)

        f = self.encoder(x)
        log_pC = self.cell_clf(f)
        p_soft = log_pC.exp().detach()
        log_pD = self.domain_clf(f, p_soft, α=α)

        # Split batch into src/tgt by domain label (0 vs 1) for MMD
        f_s, f_t = f[d_lbl == 0], f[d_lbl == 1]
        loss_mmd = self.mmd_lin(f_s, f_t) if len(f_s) > 0 and len(f_t) > 0 else 0.0

        # Calculate losses
        loss_d = F.nll_loss(log_pD, d_lbl)
        loss_c = F.nll_loss(log_pC, c_lbl)
        loss = (
            loss_c
            + self.hparams.lambda_domain * loss_d
            + self.hparams.lambda_mmd * loss_mmd
        )

        # Compute metrics
        self.train_acc_cell(log_pC, c_lbl)
        self.train_acc_domain(log_pD, d_lbl)

        # Log metrics - Lightning handles back-prop + opt step
        self.log_dict(
            {
                "train/loss": loss,
                "train/cell_acc": self.train_acc_cell,
                "train/domain_acc": self.train_acc_domain,
                "train/mmd_loss": loss_mmd,
            },
            prog_bar=True,
            on_epoch=True,
            on_step=False,
        )

        return loss

    # ---------- Validation ----------
    def validation_step(self, batch, batch_idx):
        x, d_lbl, c_lbl = batch
        d_lbl, c_lbl = d_lbl.long(), c_lbl.long()

        f = self.encoder(x)
        log_pC = self.cell_clf(f)
        p_soft = log_pC.exp().detach()
        log_pD = self.domain_clf(f, p_soft, α=1.0)

        # Split batch into src/tgt by domain label (0 vs 1) for MMD
        f_s, f_t = f[d_lbl == 0], f[d_lbl == 1]
        loss_mmd = self.mmd_lin(f_s, f_t) if len(f_s) > 0 and len(f_t) > 0 else 0.0

        loss_d = F.nll_loss(log_pD, d_lbl)
        loss_c = F.nll_loss(log_pC, c_lbl)
        loss = (
            loss_c
            + self.hparams.lambda_domain * loss_d
            + self.hparams.lambda_mmd * loss_mmd
        )

        self.val_acc_cell(log_pC, c_lbl)
        self.val_acc_domain(log_pD, d_lbl)

        self.log_dict(
            {
                "val/loss": loss,
                "val/cell_acc": self.val_acc_cell,
                "val/domain_acc": self.val_acc_domain,
                "val/mmd_loss": loss_mmd,
            },
            prog_bar=True,
            on_epoch=True,
            on_step=False,
        )

    # ---------- Optimiser ----------
    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        sched = ReduceLROnPlateau(opt, mode="min", patience=5, factor=0.5)
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, "monitor": "val/loss"},
        }

    # ---------- (Option A) save encoder when training finishes ----------
    def on_train_end(self):
        torch.save(self.encoder.state_dict(), "dann_encoder.pth")
