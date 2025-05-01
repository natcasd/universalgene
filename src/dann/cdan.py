import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics  # <-- NEW
from dann_modules import CDANHead, CellClassifier, MLPEncoder  # <- your existing code
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader


class CDANodule(pl.LightningModule):
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
        self.scaler = torch.amp.GradScaler()

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
        opt, _ = self.optimizers()
        x, d_lbl, c_lbl = batch
        d_lbl = d_lbl.long()
        c_lbl = c_lbl.long()
        p = self.current_epoch / max(1, self.trainer.max_epochs - 1)
        alpha = self.λ_domain(p, self.trainer.max_epochs)

        # Forward pass
        f = self.encoder(x)
        lp_c = self.cell_clf(f)
        p_soft = lp_c.exp().detach()  # stop gradients
        lp_d = self.domain_clf(f, p_soft, α=alpha)

        # Split batch into src/tgt by domain label (0 vs 1) for MMD
        f_s, f_t = f[d_lbl == 0], f[d_lbl == 1]
        loss_mmd = self.mmd_lin(f_s, f_t) if len(f_s) > 0 and len(f_t) > 0 else 0.0

        # Calculate losses
        d_loss = F.nll_loss(lp_d, d_lbl)
        c_loss = F.nll_loss(lp_c, c_lbl)
        loss = (
            c_loss
            + self.hparams.lambda_domain * d_loss
            + self.hparams.lambda_mmd * loss_mmd
        )

        # Backward pass
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.scaler.step(opt)
        self.scaler.update()
        opt.zero_grad()

        # log & update metrics
        self.train_acc_cell(lp_c, c_lbl)
        self.train_acc_domain(lp_d, d_lbl)

        self.log("train/loss", loss, on_step=False, on_epoch=True)
        self.log("train/cell_acc", self.train_acc_cell, on_step=False, on_epoch=True)
        self.log(
            "train/domain_acc", self.train_acc_domain, on_step=False, on_epoch=True
        )
        self.log("train/mmd_loss", loss_mmd, on_step=False, on_epoch=True)
        return loss

    # ---------- Validation ----------
    def validation_step(self, batch, batch_idx):
        x, d_lbl, c_lbl = batch
        d_lbl = d_lbl.long()
        c_lbl = c_lbl.long()

        f = self.encoder(x)
        lp_c = self.cell_clf(f)
        p_soft = lp_c.exp().detach()  # stop gradients
        lp_d = self.domain_clf(f, p_soft, α=1.0)

        # Split batch into src/tgt by domain label (0 vs 1) for MMD
        f_s, f_t = f[d_lbl == 0], f[d_lbl == 1]
        loss_mmd = self.mmd_lin(f_s, f_t) if len(f_s) > 0 and len(f_t) > 0 else 0.0

        d_loss = F.nll_loss(lp_d, d_lbl)
        c_loss = F.nll_loss(lp_c, c_lbl)
        loss = (
            c_loss
            + self.hparams.lambda_domain * d_loss
            + self.hparams.lambda_mmd * loss_mmd
        )

        self.val_acc_cell(lp_c, c_lbl)
        self.val_acc_domain(lp_d, d_lbl)

        self.log("val/loss", loss, on_step=False, on_epoch=True)
        self.log("val/cell_acc", self.val_acc_cell, on_step=False, on_epoch=True)
        self.log("val/domain_acc", self.val_acc_domain, on_step=False, on_epoch=True)
        self.log("val/mmd_loss", loss_mmd, on_step=False, on_epoch=True)

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
