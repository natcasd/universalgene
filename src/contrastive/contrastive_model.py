import pytorch_lightning as pl
from contrastive_modules import AttentionEncoder
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

# implementation of https://arxiv.org/abs/2004.11362
def supervised_contrastive_loss(embeddings, cell_types, temperature):
    similarity_matrix = torch.matmul(embeddings, embeddings.T) / temperature
    logits_mask = torch.ones_like(similarity_matrix) - torch.eye(embeddings.size(0), device=embeddings.device)

    cell_types = cell_types.contiguous().view(-1, 1)
    positive_mask = torch.eq(cell_types, cell_types.T).float() * logits_mask

    exp_logits = torch.exp(similarity_matrix) * logits_mask
    log_prob = similarity_matrix - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

    mean_log_prob_pos = (positive_mask * log_prob).sum(1) / (positive_mask.sum(1) + 1e-12)
    loss = -mean_log_prob_pos.mean()
    return loss
    
class ContrastiveModel(pl.LightningModule):
    def __init__(self, n_genes, d_model, n_heads, n_layers, lr, cls_token=False, multiply_by_expr=False, temperature=1.0, outdir=None):
        super(ContrastiveModel, self).__init__()
        self.save_hyperparameters()

        self.encoder = AttentionEncoder(n_genes, d_model, n_heads, n_layers, cls_token, multiply_by_expr)
        self.temperature = temperature
        self.outdir = outdir
    def forward(self, x):
        return self.encoder(x)
    
    def training_step(self, batch, batch_idx):
        x, _, cell_types = batch
        embeddings = self.encoder(x)
        embeddings = F.normalize(embeddings, dim=1)
        loss = supervised_contrastive_loss(embeddings, cell_types, self.temperature)
        self.log("train/loss",  loss,  on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, _, cell_types = batch
        embeddings = self.encoder(x)
        embeddings = F.normalize(embeddings, dim=1)
        loss = supervised_contrastive_loss(embeddings, cell_types, self.temperature)
        self.log("val/loss",  loss,  on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        sched = ReduceLROnPlateau(opt, mode='min', patience=5, factor=0.5)
        return {"optimizer": opt,
                "lr_scheduler": {"scheduler": sched,
                                 "monitor": "val/loss"}}
    def on_train_end(self):
        torch.save(self.encoder.state_dict(), self.outdir / "contrastive_encoder.pth")

    def on_fit_start(self):
        print("Model is on device:", next(self.parameters()).device)
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(self) 