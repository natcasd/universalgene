import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

'''
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads=4):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        # added multihead attention layer
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, batch_first=True
        )
        self.fc2 = nn.Linear(hidden_dim, 128)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = x.unsqueeze(1)  # Add a sequence dimension for multihead attention
        x, _ = self.attention(x, x, x)
        x = x.squeeze(1)  # Remove the sequence dimension
        return F.relu(self.fc2(x))
'''
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim//2)
        self.fc3 = nn.Linear(hidden_dim//2, hidden_dim//4)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.relu(self.fc3(x))

class TransformerEncoder(nn.Module):
    def __init__(self, d_in, d_model=512, n_blocks=4, n_heads=8):
        super().__init__()
        self.proj = nn.Linear(d_in, d_model)
        block = nn.TransformerEncoderLayer(
            d_model, n_heads, d_model * 4, batch_first=True, dropout=0.1
        )
        self.tf = nn.TransformerEncoder(block, n_blocks)
        self.out = nn.Linear(d_model, 128)

    def forward(self, x):
        z = self.proj(x).unsqueeze(1)  # [B,1,d_model]
        z = self.tf(z).squeeze(1)  # [B,d_model]
        return F.relu(self.out(z))


class MLPEncoder(nn.Module):
    def __init__(self, d_in, hidden=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Linear(hidden, 128),
        )

    def forward(self, x):
        return F.relu(self.net(x))


class CDANHead(nn.Module):
    def __init__(self, f_dim, n_cls):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(f_dim * n_cls, 512), nn.ReLU(), nn.Linear(512, 2)
        )

    def forward(self, f, p_soft, α=1.0):
        # f: [B,128], p_soft: [B,n_cls]
        op = torch.bmm(p_soft.unsqueeze(2), f.unsqueeze(1)).view(f.size(0), -1)
        return F.log_softmax(
            self.mlp(GradientReversalLayer.apply(op, α)), 1
        )  # this might not work!


class DomainClassifier(nn.Module):
    def __init__(self, input_dim):
        super(DomainClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        # added extra layer
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 2)  # Binary classification: Sampling Technique 1 vs 2

    def forward(self, x, alpha=1.0):
        x = GradientReversalLayer.apply(x, alpha)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.log_softmax(self.fc3(x), dim=1)


class CellClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(CellClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        # Added extra layer
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.log_softmax(self.fc3(x), dim=1)


class SingleCellDataset(Dataset):
    def __init__(self, X, domain_labels, cell_labels):
        self.X = X
        self.domain_labels = domain_labels
        self.cell_labels = cell_labels

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.domain_labels[idx], self.cell_labels[idx]
