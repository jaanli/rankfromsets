import numpy as np
import scipy.sparse
import pathlib
import pandas as pd
import datetime
from tqdm import tqdm
import torch.utils.data
import torch.nn

import util

def set_id_to_tuples(set_id):
  """Convert set id array to tuples."""
  user, days, meal_types = np.hsplit(set_id, 3)
  user = np.squeeze(user)
  days = np.squeeze(days)
  meal_types = np.squeeze(meal_types)
  return zip(user, days, meal_types)


class CrowdSourcedSetData(torch.utils.data.Dataset):
  """Crowd-sourced user consumption of sets data. The number of sets is combinatorially huge."""
  def __init__(self, sparse_mat_file, set_id_file, name):
    sparse_mat_file = pathlib.Path(sparse_mat_file)
    set_id_file = pathlib.Path(set_id_file)
    self.item_counts = scipy.sparse.load_npz(
        sparse_mat_file.with_suffix(f'.{name}.npz'))
    self.item_counts_nnz = self.item_counts.getnnz(axis=1)
    self.set_map = np.load(set_id_file.with_suffix(f'.{name}.npy'))[:, 0]
    self.user_map = self.set_map
    uniques, counts = np.unique(self.set_map, return_counts=True)
    user2count = dict(zip(uniques, counts))
    self._n_users = len(uniques)
    self._user2count = user2count

    # load metadata needed for intercepts
    self.meal_food_counts = scipy.sparse.load_npz(
        sparse_mat_file.with_suffix(f'.meal_food_counts.{name}.npz'))
    self.meal_food_counts_nnz = self.meal_food_counts.getnnz(axis=1)

  def __getitem__(self, set_idx):
    user = self.set_map[set_idx]
    meal_mat = self.item_counts[set_idx]
    items = meal_mat.nonzero()[1]
    counts = meal_mat.data
    set_size = self.item_counts_nnz[set_idx]
    bias_idx = self.meal_food_counts[set_idx].nonzero()[1]
    bias_sizes = self.meal_food_counts_nnz[set_idx]
    return user, items, counts, set_size, bias_idx, bias_sizes

  @property
  def n_users(self):
    return self._n_users

  @property
  def n_items(self):
    return self.item_counts.shape[1]

  @property
  def n_sets(self):
    """Number of intercepts. In this case, number of unique foodids."""
    return self.meal_food_counts.shape[-1]
  
  @property
  def n_meals(self):
    return self.item_counts.shape[0]

  def __len__(self):
    return self.item_counts.shape[0]


start_date = datetime.date(2000, 12, 31)

def days_since_to_weekday(days_since):
  date = start_date + datetime.timedelta(days=int(days_since))
  return date.weekday()

  

class ConsumptionData(torch.utils.data.Dataset):
  """User consumption data, where users consume sets from a fixed number of possible sets."""
  def __init__(self, item_attributes_path, tsv_file):
    super().__init__()
    # load the csr matrix of shape (n_items, n_item_attributes)
    self.item_attributes = scipy.sparse.load_npz(item_attributes_path)
    self.item_attribute_counts = np.array(self.item_attributes.getnnz(axis=1)).flatten()
    self.items_with_zero_attributes = np.where(self.item_attribute_counts == 0)[0]
    df = pd.read_csv(tsv_file, sep='\t', header=None)
    # a datapoint is a tuple: (user, item) interaction
    self.users = df.iloc[:, 0].values
    self.items = df.iloc[:, 1].values
    assert len(self.users) == len(self.items)

  @property
  def n_users(self):
    return max(self.users) + 1

  @property
  def n_items(self):
    return self.item_attributes.shape[0]
    
  @property
  def n_attributes(self):
    return self.item_attributes.shape[1]

  def __len__(self):
    """Number of datapoints; user, item interactions."""
    return len(self.users)
  
  def __getitem__(self, data_idx):
    user = self.users[data_idx]
    item_idx = self.items[data_idx]
    item_row = self.item_attributes[item_idx]
    _, item_attributes = item_row.nonzero()
    num_attributes = self.item_attribute_counts[item_idx]
    return user, item_idx, item_attributes, num_attributes


def collate_with_neg_fn(generator):
  """Collate a list of datapoints into a batch, with negative samples in last half of batch."""
  users, items, item_attr, num_attr = collate_fn(generator)
  users[len(users) // 2:] = users[:len(users) // 2]
  return users, items, item_attr, num_attr

def collate_fn(generator):
  """Collate a list of datapoints into a batch"""
  users, items, item_attributes, num_attributes = zip(*generator)
  # increment attribute index by one (index zero is reserved for the padding index)
  item_attr = [torch.tensor(x + 1, dtype=torch.long) for x in item_attributes]
  item_attr = torch.nn.utils.rnn.pad_sequence(item_attr, batch_first=True, padding_value=0)
  users = torch.tensor(users, dtype=torch.long)
  items = torch.tensor(items, dtype=torch.long)
  num_attributes = torch.tensor(num_attributes, dtype=torch.float)
  return users, items, item_attr, num_attributes


def collate_sorted_seq_fn(lst):
  """Collate examples into a packed sequence. Return padded indices."""
  users, item_attr, _, set_sizes, _, _ = zip(*lst)
  # sort in decreasing order
  batch_size = len(users)
  users = list(users)
  users[batch_size:] = users[:batch_size]
  idx = np.argsort(set_sizes)[::-1]
  unsort_idx = torch.from_numpy(np.argsort(idx))
  users = [users[i] for i in idx]
  item_attr = [item_attr[i] for i in idx]
  set_sizes = [set_sizes[i] for i in idx]
  for x in item_attr:
    np.random.shuffle(x)
  item_attr = [torch.tensor(x, dtype=torch.long) for x in item_attr]
  item_attr = torch.nn.utils.rnn.pad_sequence(item_attr, batch_first=True)
  return torch.tensor(users, dtype=torch.long), item_attr, set_sizes, unsort_idx
