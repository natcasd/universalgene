import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionEncoder(nn.Module):
    def __init__(self, n_genes, d_model, n_heads, n_layers, cls_token=False, multiply_by_expr=False, projection=False):
        super(AttentionEncoder, self).__init__()
        self.expr_proj = nn.Linear(1, d_model)
        self.gene_embedding = nn.Parameter(torch.randn(n_genes, d_model))
        enc_layer = nn.TransformerEncoderLayer(d_model, n_heads,
                                               dim_feedforward=4*d_model,
                                               dropout=0.1,
                                               batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, n_layers)
        self.use_cls = cls_token
        if cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.multiply_by_expr = multiply_by_expr
        self.projection = projection
        if projection:
            self.projection_head = ProjectionHead(d_model)

    def forward(self, x):
        # project expression scalar to model/gene embedding dim (learn this with a linear layer)
        x = self.expr_proj(x.unsqueeze(-1)) # [B, n_genes] → [B, n_genes, 1] → [B, n_genes, d_model]
        # embed gene indices
        gene_embed = self.gene_embedding[None, :, :]  # [1, n_genes, d_model]
        # inject expression level
        if self.multiply_by_expr:
            x = x * gene_embed # [B, n_genes, d_model]
        else:
            x = x + gene_embed # [B, n_genes, d_model]
        # if passed, cls token for getting cell representation
        if self.use_cls:
            cls = self.cls_token.expand(x.size(0), -1, -1)  # [B, 1, d_model]
            x = torch.cat([cls, x], dim=1)  # [B, G+1, d_model]
        # pass through transformer encoder
        x = self.encoder(x)
        if self.projection:
            x = self.projection_head(x)
        return x[:, 0] if self.use_cls else x.mean(dim=1)

class DenseEncoder(nn.Module):
    def __init__(self, n_genes, d_model, n_layers, dropout):
        super(DenseEncoder, self).__init__()
        layers = []

        layers.append(nn.Linear(n_genes, d_model))
        layers.append(nn.LeakyReLU())
        layers.append(nn.LayerNorm(d_model))
        layers.append(nn.Dropout(dropout))

        for _ in range(n_layers - 1):
            layers.append(nn.Linear(d_model, d_model))
            layers.append(nn.LeakyReLU())
            layers.append(nn.LayerNorm(d_model))
            layers.append(nn.Dropout(dropout))

        self.encoder = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.encoder(x)
    
class ProjectionHead(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model*4),
            nn.LeakyReLU(inplace=True),
            nn.Linear(d_model*4, d_model)
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)
        
        

    
        