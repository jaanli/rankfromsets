# build train.tsv, test.tsv, valid.tsv, and csr.npz for simulation
# to compare inner product model to residual model

import torch
import numpy as np
import pandas as pd
import yaml
import scipy.sparse
import scipy.special
from tqdm import tqdm
import pathlib
import time
import itertools

import data
import models

def ilogit(x):
  return 1 / (1 + np.exp(-x))

def generate_data(n_users, n_items, n_attributes, emb_size):
  """Generate dataset according to generative process:

  \theta_u ~ Uniform  # user embedding
  \beta_v ~ Uniform  # attribute embedding
  K ~ Poisson(5)  # number of attributes for an item
  p_i ~ Dirichlet  # probability of an item having an attribute
  x_m ~ Multinomial(x_{mi}; K, p_i)  # draw attributes for an item consumed by a user
  y_{um} ~ Bernoulli(\sigma(\theta_u^\top(f(\sum_{v \in x_m} \beta_v))))
  Where f(x) = x + x^2 + x^3
  """
  scale = 0.07
  average_num_attributes_per_item = 20
  theta = np.random.randn(n_users, emb_size) 
  beta = np.random.randn(n_attributes, emb_size)
  dirichlet_scale = np.zeros(n_attributes) + 0.01
  p = np.random.dirichlet(alpha=dirichlet_scale, size=n_items)
  # add 1 to ensure every item has at least a single attribute
  k = np.random.poisson(lam=average_num_attributes_per_item, size=n_items) + 1
  x = []
  beta_sum = np.zeros((n_items, emb_size))
  for m in range(n_items):
    x_m = np.random.multinomial(n=k[m], pvals=p[m], size=1)
    # binarize
    x_m = (x_m > 0.5).astype(np.int)
    x.append(x_m)
    idx = np.where(x_m)[1]
    beta_sum[m, :] = beta[idx].sum(axis=0)
  x = np.vstack(x)
  item_attributes_csr = scipy.sparse.csr_matrix(x)
  # (n_users, K) x (K, n_items) -> (n_users, n_items)
  dot = theta.dot(beta_sum.T)
  logits = np.square(dot)
  logits = (logits - logits.mean(axis=0, keepdims=True)) / logits.std(axis=0, keepdims=True)
  logits = logits - 7
  # keep popularity term fixed for every item
  observations = np.random.random(size=logits.shape) <= ilogit(logits)
  obs_user, obs_item = np.where(observations)
  print('num observations %d\tnum obs per user: %.3f' % (len(obs_user), len(obs_user) / n_users))
  return obs_user, obs_item, item_attributes_csr, theta, beta


def split_data(n_observations):
  """Split a dataset into train, valid, test."""
  idx = np.arange(n_observations)
  train_idx = np.random.choice(idx,
                               size=int(0.8 * n_observations),
                               replace=False)
  valid_idx = np.random.choice(idx[~np.isin(idx, train_idx)],
                               size=int(0.1 * n_observations),
                               replace=False)
  test_idx = idx[~np.isin(idx, np.union1d(train_idx, valid_idx))]
  return train_idx, valid_idx, test_idx


def write_tsv(observation_user, observation_item, path):
  """Write a TSV file where every line is a user, item interaction."""
  df = pd.DataFrame(data={'user': observation_user, 'item': observation_item})
  df.to_csv(path, index=False, header=False, sep='\t')


def logit_fn(theta, beta_sum):
  beta_sum = beta_sum.T
  return np.square(theta.dot(beta_sum))


if __name__ == "__main__":
  for i in range(30):
    data_dir = pathlib.Path('/tmp/dat/set_rec/simulation_%d' % i)
    data_dir.mkdir()

    n_users = 1000
    n_items = 30000
    # unique dataset for every simulated data
    np.random.seed(2423242 * i + i)

    # save test users
    test_users = np.random.choice(np.arange(n_users), size=int(0.1 * n_users), replace=False)
    with (data_dir / 'test_users.tsv').open('w') as f:
      f.writelines('%d\n' % u for u in test_users)


    obs_user, obs_item, item_attributes_csr, theta, beta = generate_data(n_users=n_users,
                                                                                  n_items=n_items,
                                                                                  n_attributes=5000,
                                                                                  emb_size=128)


    # save tsv files
    train_idx, valid_idx, test_idx = split_data(len(obs_user))

    all_docs = set(np.arange(n_items))
    out_matrix_item_idx = all_docs.difference(set(obs_item[train_idx]))
    print('number of out-matrix items: ', len(out_matrix_item_idx))

    for idx, name in [(train_idx, 'train.tsv'), (valid_idx, 'valid.tsv'), (test_idx, 'test.tsv')]:
      write_tsv(obs_user[idx], obs_item[idx], data_dir / name)

    # save item attributes
    scipy.sparse.save_npz(data_dir / 'item_attributes_csr.npz', item_attributes_csr)
