# lightning_dann.py
import pytorch_lightning as pl
import torchmetrics                        # <-- NEW
import torch, torch.nn.functional as F
from dann_modules import Encoder, DomainClassifier, CellClassifier          # <- your existing code
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau


class DANNModule(pl.LightningModule):
    def __init__(self,
                 input_dim: int,
                 num_cell_classes: int,
                 hidden_dim: int = 512,
                 lambda_domain: float = 1.0,
                 lr: float = 1e-4):
        super().__init__()
        self.save_hyperparameters()

        self.encoder = Encoder(input_dim, hidden_dim)
        self.domain_clf = DomainClassifier(128)
        self.cell_clf   = CellClassifier(128, num_cell_classes)

        # metrics you want to track
        self.train_acc_cell   = torchmetrics.Accuracy(task="multiclass",
                                                      num_classes=num_cell_classes)
        self.train_acc_domain = torchmetrics.Accuracy(task="multiclass", num_classes=2)
        self.val_acc_cell     = torchmetrics.Accuracy(task="multiclass",
                                                      num_classes=num_cell_classes)
        self.val_acc_domain   = torchmetrics.Accuracy(task="multiclass", num_classes=2)


    # ---------- Forward ----------
    def forward(self, x):
        return self.encoder(x)

    # ---------- Training ----------
    def training_step(self, batch, batch_idx):
        x, d_lbl, c_lbl = batch
        d_lbl = d_lbl.long(); c_lbl = c_lbl.long()
        # alpha = 2. / (1. + torch.exp(-10 * self.current_epoch / self.trainer.max_epochs)) - 1
        p = self.current_epoch / max(1, self.trainer.max_epochs - 1)
        alpha = 2.0 / (1.0 + torch.exp(torch.tensor(-10 * p, device=self.device))) - 1.0


        z           = self.encoder(x)
        domain_pred = self.domain_clf(z, alpha=alpha)
        cell_pred   = self.cell_clf(z)

        d_loss = F.nll_loss(domain_pred, d_lbl)
        c_loss = F.nll_loss(cell_pred,   c_lbl)
        loss   = c_loss + self.hparams.lambda_domain * d_loss

        # log & update metrics
        self.train_acc_cell (cell_pred,   c_lbl)
        self.train_acc_domain(domain_pred, d_lbl)
        # self.log_dict({"train/loss": loss,
        #                "train/cell_acc":  self.train_acc_cell,
        #                "train/domain_acc":self.train_acc_domain},
        #               prog_bar=True, on_step=False, on_epoch=True)

        self.log("train/loss",  loss,  on_step=False, on_epoch=True)
        self.log("train/cell_acc", self.train_acc_cell, on_step=False, on_epoch=True)
        self.log("train/domain_acc", self.train_acc_domain, on_step=False, on_epoch=True)
        return loss

    # ---------- Validation ----------
    def validation_step(self, batch, batch_idx):
        x, d_lbl, c_lbl = batch
        d_lbl = d_lbl.long(); c_lbl = c_lbl.long()
        z           = self.encoder(x)
        domain_pred = self.domain_clf(z, alpha=1.0)
        cell_pred   = self.cell_clf(z)

        d_loss = F.nll_loss(domain_pred, d_lbl)
        c_loss = F.nll_loss(cell_pred,   c_lbl)
        loss   = c_loss + self.hparams.lambda_domain * d_loss

        self.val_acc_cell (cell_pred,   c_lbl)
        self.val_acc_domain(domain_pred, d_lbl)

        # self.log_dict({"val/loss":   loss,
        #                "val/cell_acc":   self.val_acc_cell,
        #                "val/domain_acc": self.val_acc_domain},
        #               prog_bar=True, on_step=False, on_epoch=True)

        self.log("val/loss",   loss,  on_step=False, on_epoch=True)
        self.log("val/cell_acc",   self.val_acc_cell, on_step=False, on_epoch=True)
        self.log("val/domain_acc",   self.val_acc_domain, on_step=False, on_epoch=True)


    # ---------- Optimiser ----------
    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        sched = ReduceLROnPlateau(opt, mode='min', patience=5, factor=0.5)
        return {"optimizer": opt,
                "lr_scheduler": {"scheduler": sched,
                                 "monitor": "val/loss"}}

    # ---------- (Option A) save encoder when training finishes ----------
    def on_train_end(self):
        torch.save(self.encoder.state_dict(), "dann_encoder.pth")
