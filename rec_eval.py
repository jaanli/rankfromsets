import bottleneck as bn
import numpy as np
import bottleneck as bn
#import mxnet as mx
import logging
from tqdm import tqdm


def precision_recall_at_k_batch(pred, idx, true_idx_list, k):
  """Precision and recall at k for a batch of predictions."""
  # top k items are recommended
  pred_binary = np.zeros_like(pred, dtype=bool)
  pred_binary[np.arange(len(pred))[:, np.newaxis], idx[:, :k]] = True
  true_binary = np.zeros_like(pred, dtype=bool)
  for i, true_idx in enumerate(true_idx_list):
    true_binary[i, true_idx] = True
  true_positives = np.logical_and(true_binary, pred_binary).sum(axis=1).astype(np.float32)
  # precision = true positives / true positives + false positives.
  # for precision@k we make k predictions, so the denominator is simply k.
  precision = true_positives / k
  # recall = true positives / true positives + false negatives
  recall = true_positives / true_binary.sum(axis=1)
  return precision, recall

def user_idx_generator(users, batch_size):
  """helper function to generate the user index to loop through the dataset"""
  n_users = len(users)
  for start in range(0, n_users, batch_size):
    end = min(n_users, start + batch_size)
    yield users[start:end]
