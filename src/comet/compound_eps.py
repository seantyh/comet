import torch
import torch.nn as nn
import torch.nn.functional as F

class CompoundEps(nn.Module):
  def __init__(self, emb_dim, h_dim=800, dropout=0.1):
    super().__init__()
    in_dim = emb_dim*2
    out_dim = emb_dim
    self.lin_proj = nn.Parameter(torch.randn(in_dim, out_dim))
    self.fc_mod = nn.Sequential(
      nn.Linear(in_dim, h_dim),
      nn.Tanh(),
      nn.Dropout(dropout),
      nn.Linear(h_dim, out_dim),
      nn.Tanh()
    )

  def forward(self, x, true_vecs=None):
    lin = torch.matmul(x, self.lin_proj)
    eps = self.fc_mod(x)
    pred = lin+eps
    if true_vecs is not None:
      loss = F.mse_loss(pred, true_vecs)
      return pred, loss
    return pred