import numpy as np
import torch.utils.data


class BatchSamplerWithNegativeSamples(torch.utils.data.Sampler):
  """Samples batches where first half is positive, second half are negative.

  We discard the last batch and check that we never use the same positive and negative sample.
  """
  def __init__(self, pos_sampler, neg_sampler, batch_size, items):
    self._pos_sampler = pos_sampler
    self._neg_sampler = neg_sampler
    self._items = items
    assert batch_size % 2 == 0, 'Batch size must be divisible by two for negative samples.'
    self._batch_size = batch_size

  def __iter__(self):
    batch, neg_batch = [], []
    neg_sampler = iter(self._neg_sampler)
    for pos_idx in self._pos_sampler:
      batch.append(pos_idx)
      neg_idx = pos_idx
      # keep sampling until we get a true negative sample
      while self._items[neg_idx] == self._items[pos_idx]:
        try:
          neg_idx = next(neg_sampler)
        except StopIteration:
          neg_sampler = iter(self._neg_sampler)
          neg_idx = next(neg_sampler)
      neg_batch.append(neg_idx)
      if len(batch) == self._batch_size // 2:
        batch.extend(neg_batch)
        yield batch
        batch, neg_batch = [], []
    return

  def __len__(self):
    return len(self._pos_sampler) // self._batch_size
