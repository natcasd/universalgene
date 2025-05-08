import torch
import torch.nn as nn

class AttentionEncoder(nn.Module):
    def __init__(self, n_genes, d_model, n_heads, n_layers, cls_token=False, multiply_by_expr=False):
        super(AttentionEncoder, self).__init__()
        self.expr_proj = nn.Linear(1, d_model)
        self.gene_embed = nn.Embedding(n_genes, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model, n_heads,
                                               dim_feedforward=4*d_model,
                                               dropout=0.1,
                                               batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, n_layers)
        self.use_cls = cls_token
        if cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.multiply_by_expr = multiply_by_expr

    def forward(self, x):
        # project expression scalar to model/gene embedding dim (learn this with a linear layer)
        x = self.expr_proj(x.unsqueeze(-1)) # [B, n_genes] → [B, n_genes, 1] → [B, n_genes, d_model]
        # create index for each gene
        idx = torch.arange(x.size(1), device=x.device) # [n_genes]
        # embed gene indices
        gene_embedding = self.gene_embed(idx)[None, ...] # [1, n_genes, d_model]
        # inject expression level
        if self.multiply_by_expr:
            x = x * gene_embedding # [B, n_genes, d_model]
        else:
            x = x + gene_embedding # [B, n_genes, d_model]
        # if passed, cls token for getting cell representation
        if self.use_cls:
            cls = self.cls_token.expand(x.size(0), -1, -1)  # [B, 1, d_model]
            x = torch.cat([cls, x], dim=1)  # [B, G+1, d_model]
        # pass through transformer encoder
        x = self.encoder(x)
        if self.use_cls:
            # return cls token representation
            return x[:, 0]
        else:
            # return mean of all token representations
            return x.mean(dim=1)
       # could also think about using projection head here

    
        