import logging
import yaml
import pathlib
import pandas as pd
import numpy as np
import collections
import numpy as np
import torch
import copy
import bottleneck as bn
from tqdm import tqdm
import addict

import rec_eval
import data
import util
import models
import torch_fit

logger = logging.getLogger(__name__)


def compute_precision_recall(cfg, step, model, sample_users, heldout_data, heldout_name, item_idx, eval_mode):
  """Calculate precision/recall on a subset of users."""
  model.eval()
  # create a temporary dataset and model on CPU (GPU memory cannot fit all items)
  dataset = copy.deepcopy(heldout_data)
  # we overwrite the user below
  dataset.users = item_idx * 0
  dataset.items = item_idx
  batch_size = int(cfg.eval_batch_size) if cfg.eval_batch_size is not None else len(dataset)
  eval_data = torch.utils.data.DataLoader(
    dataset, batch_size=batch_size,
    collate_fn=data.collate_fn, num_workers=cfg.num_workers, 
    pin_memory=cfg.pin_memory if cfg.eval_batch_size is not None else False)
  pred_list = []

  device = torch.device('cuda' if next(model.parameters()).is_cuda else 'cpu')

  if cfg.eval_batch_size is None:
    batch = next(iter(eval_data))
    users, items, item_attributes, num_attributes = batch
    model.to('cpu')

  # make predictions for every user and calculate recall
  precision_dict = collections.defaultdict(lambda: list())
  recall_dict = collections.defaultdict(lambda: list())
  # number of recommendations
  k_list = cfg.number_of_recommendations
  for batch_users in tqdm(rec_eval.user_idx_generator(sample_users, 100)):
    if isinstance(model, models.InnerProduct):
      users = torch.tensor(batch_users, dtype=torch.long)
      with torch.no_grad():
        item_idx_pred = model(users, items, item_attributes, num_attributes, pairwise=True).cpu().numpy()
    else:
      item_idx_pred = []
      for row, user in tqdm(enumerate(batch_users), 'evaluating one batch of users'):
        user_pred = []
        for batch in eval_data:
          users, items, item_attributes, num_attributes = batch
          users = users.to(device)
          items = items.to(device)
          item_attributes = item_attributes.to(device)
          num_attributes = num_attributes.to(device)
          with torch.no_grad():
            user_pred.append(model(users, items, item_attributes, num_attributes).cpu().numpy())
            torch.cuda.empty_cache()
        item_idx_pred.append(np.hstack(user_pred))
      item_idx_pred = np.vstack(item_idx_pred)
    # we only predict either in-matrix or out-matrix items. all else is minus inf
    pred = np.full((len(batch_users), heldout_data.item_attributes.shape[0]), -np.inf)
    pred[:, item_idx] = item_idx_pred
    # get the items the user consumed in the data
    true_idx_list = []
    for i, user in enumerate(batch_users):
      user_item_idx = (heldout_data.users == user)
      unique_sets = np.unique(heldout_data.items[user_item_idx])
      true_idx_list.append(unique_sets)
    sorted_idx = bn.argpartition(-pred, max(k_list), axis=1)
    for k in k_list:
      precision, recall = rec_eval.precision_recall_at_k_batch(pred, sorted_idx, true_idx_list, k)
      precision_dict[k].append(precision)
      recall_dict[k].append(recall)

  # compute average precision and recall across users for every k
  mean_precision_list = []
  mean_recall_list = []
  for k in k_list:
    mean_precision_list.append(np.mean(np.hstack(precision_dict[k])))
    mean_recall_list.append(np.mean(np.hstack(recall_dict[k])))

  # write the results
  name = '_'.join([heldout_name, eval_mode, 'precision'])
  fname = (cfg.train_dir / name).with_suffix('.csv')
  util.write_row(fname, [step] + mean_precision_list)
  strings = ' '.join([f'{x:.3e}' for x in mean_precision_list])
  logger.info(f'step {step}\t{heldout_name}\t{eval_mode}')
  logger.info(f'\t\tmean precision for k={str(k_list)}:\t{strings}')

  name = '_'.join([heldout_name, eval_mode, 'recall'])
  fname = (cfg.train_dir / name).with_suffix('.csv')
  util.write_row(fname, [step] + mean_recall_list)
  strings = ' '.join([f'{x:.3e}' for x in mean_recall_list])
  logger.info(f'\t\tmean recall for k={str(k_list)}:\t{strings}')

  # return the model to the original device
  model.to(device)
  torch.cuda.empty_cache()
  # early stopping is based on recall
  return mean_recall_list[-1]

def drop_inactive_users(heldout_data, sample_users, item_idx):
  """Filter users and return only those users that have an observation in heldout data.

  c.f. https://github.com/premgopalan/collabtm/blob/cfc9a83b21cd3fedc1bf13bbda86a8b239ef517d/src/collabtm.cc#L352
  (Otherwise, recall is undefined because of division by zero.)
  """
  # boolean indices of cold-start items in the heldout data
  idx = np.isin(heldout_data.items, item_idx)
  # count how many times a user consumed a cold-start item in the heldout data
  # users has no zero counts, so this is the filtering step we need
  filtered_users = np.unique(heldout_data.users[idx])
  idx = np.isin(filtered_users, sample_users)
  sample_users_filtered = filtered_users[idx].tolist()
  return sample_users_filtered

def evaluate(cfg, step, model, train_data, heldout_data, heldout_name, sample_users):
  """Evaluate both in-matrix and out-matrix precision and recall."""
  # get in-matrix items, those that have at least one click in the data
  train_docs = np.unique(train_data.items)

  # only keep heldout clicks on docs with zero elements if they have no clicks in training
  heldout_docs = np.unique(heldout_data.items)
  heldout_zero_docs = heldout_docs[np.isin(heldout_docs, heldout_data.items_with_zero_attributes)]
  overlap_idx = np.isin(heldout_zero_docs, train_docs)
  heldout_docs_to_drop = heldout_zero_docs[~np.isin(heldout_zero_docs, overlap_idx)]
  idx_to_drop = np.isin(heldout_data.items, heldout_docs_to_drop)
  heldout_data.items = heldout_data.items[~idx_to_drop]
  heldout_data.users = heldout_data.users[~idx_to_drop]

  # compute in-matrix precision and recall
  sample_users_filtered = drop_inactive_users(heldout_data, sample_users, train_docs)
  if heldout_name != 'valid':
    recall = compute_precision_recall(cfg, step, model, sample_users_filtered, heldout_data, heldout_name, train_docs, 'in_matrix')

  # get out-matrix items, that have not been rated by any users
  all_docs = set(np.arange(train_data.item_attributes.shape[0]))
  out_matrix_item_idx = all_docs.difference(train_docs)
  out_matrix_item_idx = np.array(list(out_matrix_item_idx), dtype=int)
  print('number of out-matrix items', len(out_matrix_item_idx))
  print('total number of items', len(all_docs))
  # remove items that have no attributes; they have no clicks in the training data and no content (so no way to predict)
  out_matrix_item_idx = out_matrix_item_idx[~np.isin(out_matrix_item_idx, heldout_data.items_with_zero_attributes)]

  # compute out-matrix precision and recall
  sample_users_filtered = drop_inactive_users(heldout_data, sample_users, out_matrix_item_idx)
  recall = compute_precision_recall(cfg, step, model, sample_users_filtered, heldout_data, heldout_name, out_matrix_item_idx, 'out_matrix')
  return recall


if __name__ == '__main__':
  with open('config.yml', 'r') as f:
    cfg = addict.Dict(yaml.load(f))
  cfg.parse_args()
  train_data, valid_data, test_data = [data.ConsumptionData(cfg.item_attributes, cfg[tsv]) \
                        for tsv in ['train_tsv', 'valid_tsv', 'test_tsv']]

  if cfg.test_users_tsv is None:
    np.random.seed(2423)
    heldout_users = np.union1d(valid_data.users, test_data.users)
    sample_users = np.random.choice(heldout_users, min(10000, len(heldout_users)), replace=False)
    print('sampled %d users for testing', len(sample_users))
  else:
    sample_users = pd.read_csv(cfg.test_users_tsv, header=None).values[:, 0]
  
  eval_hook = lambda step, model, heldout_data, heldout_name: evaluate(cfg, step, model, train_data, heldout_data, heldout_name, sample_users)
  
  torch_fit.fit(cfg, train_data, valid_data, test_data, eval_hook)
