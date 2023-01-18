from tqdm.auto import tqdm
import numpy as np
from sklearn.neighbors import NearestNeighbors
import torch
from torch.utils.data import DataLoader


def compute_knn_accuracies(model,
    loader: DataLoader, 
    knn: NearestNeighbors, 
    device: str="cpu"):
  """Compute the top-k accuracies of the model using k-nearest neighbors."""

  model.eval()
  n_instances = len(loader.dataset)   # type: ignore
  with torch.no_grad():
    n_correct = 0
    topk_corrects = np.zeros((15,)) 
    for batch in tqdm(loader, desc="Computing knn accuracy"):
      consts = batch["consts"].to(device)
      compound = batch["compound"].to(device)
      pred = model(consts)
      _, idx = knn.kneighbors(pred.cpu())
      batch_true_idxs = batch["idx"].numpy()
      for k in range(15):
        correct_mat = (idx[:, :k+1] == batch_true_idxs[:, np.newaxis])
        topk_corrects[k] += correct_mat.sum().item()
  topk_corrects = [x/n_instances for x in topk_corrects]
  return topk_corrects
