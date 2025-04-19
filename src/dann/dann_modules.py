import torch
import numpy as np
import scanpy as sc
import pickle
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import os
import scipy.sparse as sp
from torch.utils.data import Dataset

class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 128)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.relu(self.fc2(x))
class DomainClassifier(nn.Module):
    def __init__(self, input_dim):
        super(DomainClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 2)  # Binary classification: Sampling Technique 1 vs 2

    def forward(self, x, alpha=1.0):
        x = GradientReversalLayer.apply(x, alpha)
        x = F.relu(self.fc1(x))
        return F.log_softmax(self.fc2(x), dim=1)
class CellClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(CellClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.log_softmax(self.fc2(x), dim=1)

class SingleCellDataset(Dataset):
    def __init__(self, X, domain_labels, cell_labels):
        self.X = X
        self.domain_labels = domain_labels
        self.cell_labels = cell_labels

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.domain_labels[idx], self.cell_labels[idx]