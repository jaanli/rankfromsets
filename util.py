import logging
import pandas as pd
import numpy as np
import os
import time
import json
import csv
import pathlib
#from mxnet import gluon
from joblib import Parallel
from joblib import delayed

import data

def cycle(iterable):
  while True:
    for x in iterable:
      yield x

def build_mixed_weight_decay_trainer(cfg, params):
  bias = {k: v for k, v in params.items() if 'bias' in k}
  items = {k: v for k, v in params.items() if 'item' in k}
  other_params = {k: v for k, v in params.items() if ('bias' not in k and 'item' not in k)}
  print('bias parameters', bias)
  print('item parameters', items)
  print('others', other_params)
  cfg['weight_decay'] = 0.0
  trainer_other = build_trainer(cfg, other_params)

  cfg['weight_decay'] = cfg['regularizer_items']
  trainer_items = build_trainer(cfg, items)

  cfg['weight_decay'] = cfg['regularizer_bias']
  trainer_bias = build_trainer(cfg, bias)
  return trainer_other, trainer_items, trainer_bias

def build_trainer(cfg, params):
  """Build a gluon trainer for the parameters based on configuration."""
  if cfg['linear_learning_rate_decay']:
    lr_scheduler = LearningRateScheduler(cfg['learning_rate'], cfg['max_iterations'])
  else:
    lr_scheduler = None

  if cfg['optimizer'] == 'sgd':
    trainer_kwargs = {'optimizer': cfg['optimizer'],
                      'optimizer_params': {'learning_rate': cfg['learning_rate'],
                                           'momentum': cfg['momentum'],
                                           'wd': cfg['weight_decay'],
                                           'lr_scheduler': lr_scheduler}}
  elif cfg['optimizer'] == 'rmsprop':
    trainer_kwargs = {'optimizer': cfg['optimizer'],
                       'optimizer_params': {'learning_rate': cfg['learning_rate'],
                                            'centered': cfg['rmsprop_centered'],
                                            'lr_scheduler': lr_scheduler}}

  return gluon.Trainer(params, **trainer_kwargs)

# class SetEmbedding(gluon.HybridBlock):
#   """Get set embedding from batch item embeddings and counts."""

#   def __init__(self, include_item_weight, average):
#     super().__init__()
#     self.include_item_weight = include_item_weight
#     self.average = average


#   def hybrid_forward(self, F, item_emb, item_weight, set_sizes):
#     if self.include_item_weight:
#       item_emb = F.broadcast_mul(item_emb, F.expand_dims(item_weight, -1))
#     # transpose so set_size is first dimension. needed for mask
#     item_emb = F.transpose(item_emb, axes=(1, 0, 2))
#     # mask the embeddings
#     mask = F.SequenceMask(F.ones_like(item_emb),
#                           sequence_length=set_sizes, use_sequence_length=True)
#     item_emb = mask * item_emb
#     # sum over the item embeddings
#     set_emb = F.sum(item_emb, axis=0)
#     if self.average:
#       set_emb = F.broadcast_div(set_emb, F.expand_dims(set_sizes, -1))
#     return set_emb


# class AttentionSetEmbedding(gluon.HybridBlock):
#   """Attention(Q, K, V) = Softmax(QK^\top/sqrt(d)) V.

#   This uses self-attention (queries, keys, values all item embeddings),
#   and user-attention (queries are user embeddings, keys and values are item embeddings).
#   c.f. http://nlp.seas.harvard.edu/2018/04/03/attention.html#attention
#   """

#   def __init__(self):
#     super().__init__(include_user_attention, dim)
#     self.include_user_attention = include_user_attention
#     with self.name_scope():
#       self.attention = ScaledDotProductAttention(dim=dim)

#   def hybrid_forward(self, F, user_emb, item_emb, set_sizes):
#     if self.include_user_attention:
#       second_queries = user_emb
#     else:
#       second_queries = None
#     contexts = self.attention(first_queries=item_emb,
#                            second_queries=second_queries,
#                            values=item_emb,
#                            lengths=set_sizes)
    
#     # transpose so set_size is first dimension. needed for mask
#     item_emb = F.transpose(item_emb, axes=(1, 0, 2))
#     # mask the embeddings
#     mask = F.SequenceMask(F.ones_like(item_emb),
#                           sequence_length=set_sizes, use_sequence_length=True)
#     item_emb = mask * item_emb
#     # sum over the item embeddings
#     set_emb = F.sum(item_emb, axis=0)
#     if self.average:
#       set_emb = F.broadcast_div(set_emb, F.expand_dims(set_sizes, -1))
#     return set_emb


# class ScaledDotProductAttention(gluon.HybridBlock):
#   """c.f. http://nlp.seas.harvard.edu/2018/04/03/attention.html#attention"""
#   def __init__(self, dim):
#     super().__init__()
#     self.sqrt_dim = np.sqrt(dim)
#     self.LARGE_NEGATIVE_VALUE = -99999999.
  
#   def hybrid_forward(self, F, first_queries, second_queries, keys, values, lengths):
#     """
#     c.f. https://github.com/awslabs/sockeye/blob/master/sockeye/layers.py
#     Computes dot attention for a set of queries, keys, and values.
#     :param queries: Attention queries. Shape: (n, lq, d).
#     :param keys: Attention keys. Shape: (n, lk, d).
#     :param values: Attention values. Shape: (n, lk, dv).
#     :param lengths: Optional sequence lengths of the keys. Shape: (n,).
#     :param dropout: Dropout probability.
#     :param bias: Optional 3d bias tensor.
#     :param prefix: Optional prefix
#     :return: 'Context' vectors for each query. Shape: (n, lq, dv).
#     """
#     # (n, lq, d) x (n, lk, d) -> (n, lq, lk)
#     logits = F.batch_dot(lhs=queries, rhs=keys, transpose_b=True)
#     if second_queries is not None:
#       logits = logits + F.batch_dot(lhs=second_queries, rhs=keys, transpose_b=True)
#     logits = logits / self.sqrt_dim
#     # mask lk dimension; this is the set dimension in (batch_size, max_set_size, dim)
#     logits = F.transpose(logits, (1, 0, 2))
#     # large negative value ensures values are set to zero
#     logits = F.SequenceMask(data=logits,
#                             use_sequence_length=True,
#                             sequence_length=lengths,
#                             value=self.LARGE_NEGATIVE_VALUE)
#     logits = F.transpose(logits, (0, 1, 2))
#     # add bias term here? or dropout?
#     probs = F.softmax(logits, dim=-1)
#     # (n, lq, lk) x (n, lk, dv) -> (n, lq, dv)
#     return F.batch_dot(lhs=probs, rhs=values)


# class LearningRateScheduler(object):
#   def __init__(self, init_lr, total_iterations):
#     self.total_iterations = total_iterations
#     self.init_lr = init_lr
    
#   def __call__(self, iteration):
#     progress = iteration / self.total_iterations
#     return self.init_lr * (1.0 - progress)
    

# def inv_softrelu(x):
#   return np.log(np.exp(x) - 1.)


# class BatchDot(gluon.HybridBlock):
#   def __init__(self):
#     super().__init__()

#   def hybrid_forward(self, F, x, y):
#     return F.squeeze(F.batch_dot(F.expand_dims(x, 1), F.expand_dims(y, -1)))


# class LogSumExp(gluon.HybridBlock):
#   """Calculate logsumexp for numerical stability."""
#   def __init__(self, axis=0):
#     super().__init__()
#     self.axis = 0

#   def hybrid_forward(self, F, x):
#     max_x = F.max_axis(x, self.axis, keepdims=True)
#     exp = F.exp(F.broadcast_sub(x,  max_x))
#     logsumexp = max_x + F.log(F.sum(exp, self.axis, keepdims=True))
#     return F.squeeze(logsumexp, axis=self.axis)


# def standardize(x, std):
#   return std * (x - x.mean(axis=0)) / x.std(axis=0)


def load_word2vec_embeddings(path):
  # init item embeddings to those from word2vec
  vectors = load_model(path)
  np_vectors = np.zeros((len(vectors), len(vectors['0'])))
  zero_idx = []
  for i in range(len(np_vectors)):
    try:
      np_vectors[i] = vectors[str(i)]
    except:
      zero_idx.append(i)
#  if cfg['standardize_data']:
#    np_vectors = standardize(np_vectors, cfg['standardize_data_std'])
#  if cfg['normalize_data']:
#    np_vectors = np_vectors / np.sum(np_vectors, axis=0)
  mean_emb = np_vectors.mean(axis=0, keepdims=True)
  np_vectors[np.array(zero_idx, dtype=int)] = np.squeeze(mean_emb)
  return np_vectors
# if cfg['emb_size'] < 128:
#   mat = PCA(n_components=cfg['emb_size']).fit_transform(np_vectors)
# model.item_embeddings.weight.set_data(nd.array(mat))
# mat = np_vectors
# if cfg['latent_size_meal'] < 128:
#   mat = PCA(n_components=cfg['latent_size_meal']).fit_transform(np_vectors)
#   mat = standardize(mat)


def summarize(writer, name, array, step):
  np_array = array.asnumpy()
  for stat in ['max', 'min', 'mean', 'linalg.norm']:
    if stat == 'linalg.norm':
      fn = np.linalg.norm
    else:
      fn = getattr(np, stat)
    writer.add_scalar(f'{name}/{stat}', fn(np_array.ravel()), global_step=step)


def summarize_validation_row(writer, name, row, step):
  for k, val in enumerate(row):
    writer.add_scalar(f'{name}/{k}', val, global_step=step)


def summarize(writer, name, array, step):
  np_array = array.asnumpy()
  for stat in ['max', 'min', 'mean', 'linalg.norm']:
    if stat == 'linalg.norm':
      fn = np.linalg.norm
    else:
      fn = getattr(np, stat)
    writer.add_scalar(f'{name}/{stat}', fn(np_array.ravel()), global_step=step)


def load_vocab(vocab_file):
  return pd.read_csv(vocab_file, header=None, index_col=0)[1].to_dict()


def make_log_dir(cfg, log_dir_name=None):
  """Create a date directory and experiment directory."""
  date = time.strftime("%Y-%m-%d")
  log_dir = pathlib.Path(cfg['log_dir'])
  train_dir = log_dir / date / cfg['experiment_name']
  if log_dir_name is not None:
    train_dir = train_dir / log_dir_name
  if not os.path.exists(train_dir):
    os.makedirs(train_dir)
  return train_dir


def write_row(fname, row):
  with open(fname, 'a') as f:
    writer = csv.writer(f)
    writer.writerow(row)


def compat_splitting(line):
  return line.decode('utf8').split()


def load_model(path):
  vectors = {}
  fin = open(str(path), 'rb')
  for i, line in enumerate(fin):
    if i == 0 or i == 1:
      pass
    else:
      try:
        tab = compat_splitting(line)
        vec = np.array(tab[1:], dtype=float)
        word = tab[0]
        if not word in vectors:
            vectors[word] = vec
      except ValueError:
          continue
      except UnicodeDecodeError:
          continue
  fin.close()
  return vectors


def get_file_console_logger(filename):
  filename = str(filename)
  logging.basicConfig(level=logging.INFO,
                      format='%(asctime)s %(name)-4s %(levelname)-4s %(message)s',
                      datefmt='%m-%d %H:%M',
                      filename=filename,
                      filemode='a')
  console = logging.StreamHandler()
  console.setLevel(logging.INFO)
  logger = logging.getLogger('')
  # only add the console handler if there is only the file handler
  if len(logger.handlers) == 1:
    logger.addHandler(console)
  return logger


def pack(items, item_weight, set_sizes):
  """Pack flattened arrays into shape (n_sets, max_set_size)."""
  split_ind = np.zeros(len(set_sizes) - 1, dtype=int)
  split_ind[0] = set_sizes[0]
  for i in range(1, len(set_sizes) - 1):
    split_ind[i] = split_ind[i - 1] + set_sizes[i]
  items_list = np.split(items, split_ind)
  item_weight_list = np.split(item_weight, split_ind)
  return items_list, item_weight_list

      
def build_set_emb(cfg, train_dataset, model):
  """Build set embeddings in a parallelized way."""
  csr = train_dataset.item_counts
  # need to get inputs of shape (n_sets, max_set_size)
  _, items = csr.nonzero()
#  item_weight = csr.data
  item_weight = train_dataset.item_weight.data
  set_sizes = train_dataset.item_counts_nnz
  bias_idx = np.arange(csr.shape[0])
  items_list, item_weight_list = pack(items, item_weight, set_sizes)
  # create a fake list of users to use the batchify_fn
  zero = np.zeros(1, dtype=float)
  fake_users = [zero for _ in range(len(items_list))]
  bias_sizes = set_sizes * 0 + 1
  tup = list(zip(fake_users, items_list, item_weight_list, set_sizes, bias_idx, bias_sizes))
  _, items, item_weight, set_sizes, bias_idx, _ = data.batchify_fn(tup)
  # lookup item embeddings
  item_embeddings = model.item_embeddings.weight.data().asnumpy()
  bias = model.bias.weight.data().asnumpy()
  items, item_weight, set_sizes, bias_idx = (x.asnumpy() for x in [items, item_weight, set_sizes, bias_idx])
  set_sizes = set_sizes.astype(int)
  items = items.astype(int)
  bias_idx = bias_idx.astype(int)

  # this has to be global as joblib.Parallel can't pickle local objects
  global set_embedding
  # def set_embedding(set_idx):
  #   user, items, item_weight, set_size, bias_idx, bias_size = train_dataset[set_idx]
  #   item_emb = item_embeddings[items][:set_size]
  #   if cfg['include_item_weight']:
  #     item_emb = item_emb * item_weight[:set_size]
  #   if cfg['set_emb_average']:
  #     item_emb = item_emb / set_size
  #   set_emb = item_emb.sum(axis=0)
  #   set_bias = np.squeeze(bias[bias_idx])
  #   return set_emb + set_bias

  def set_embedding(items, item_weight, set_size, bias_idx):
    item_emb = item_embeddings[items][:set_size]
    if cfg['include_item_weight']:
      item_emb = item_emb * np.expand_dims(item_weight[:set_size], -1)
    if cfg['set_emb_average']:
      item_emb = item_emb / set_size
    set_emb = item_emb.sum(axis=0)
    set_bias = np.squeeze(bias[bias_idx])
    return set_emb + set_bias

  res = Parallel(n_jobs=1, verbose=3)(delayed(set_embedding)(items_arr, weight, set_size, idx) for items_arr, weight, set_size, idx in zip(items, item_weight, set_sizes, bias_idx))
  _, unique_set_idx = np.unique(train_dataset.set_map, return_index=True)
  
# res = Parallel(n_jobs=4, verbose=3)(delayed(set_embedding)(set_idx)
  #                                     for set_idx in unique_set_idx)

  del item_embeddings
  del set_embedding
  set_emb = np.stack(res)
  return set_emb
