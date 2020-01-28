import time
import numpy as np

import torch
import torch.utils.data
import torch.optim
import torch.multiprocessing

import util
import data
import sampler
import models
import rec_eval


def fit(cfg, train_dataset, valid_dataset, test_dataset, eval_hook):
  """Fit the RankFromSets model to a dataset, with early stopping."""
  device = torch.device("cuda" if cfg.use_gpu else "cpu")
  torch.manual_seed(24232)
  logger = util.get_file_console_logger(cfg.train_dir /  'train.log')

  # positive labels are given to every datapoint in the data
  pos_sampler = torch.utils.data.RandomSampler(train_dataset)

  # negative sampling distribution is uniform over items in the dataset
  prob = np.zeros(len(train_dataset), dtype=np.float32)
  _, unique_item_idx = np.unique(train_dataset.items, return_index=True)
  prob[unique_item_idx] = 1.0 / len(unique_item_idx)
  neg_sampler = torch.utils.data.WeightedRandomSampler(weights=prob, num_samples=len(train_dataset))

  # this sampler yields batches with negative samples in the last half of the batch
  if cfg.batch_size >= len(train_dataset):
    batch_size = len(train_dataset) // 2 * 2
  else:
    batch_size = cfg.batch_size
  batch_sampler = sampler.BatchSamplerWithNegativeSamples(
    pos_sampler=pos_sampler, neg_sampler=neg_sampler,
    items=train_dataset.items, batch_size=batch_size)

  kwargs = {'num_workers': cfg.num_workers, 'pin_memory': cfg.pin_memory} if cfg.use_gpu else {'num_workers': cfg.num_workers}
  train_data = torch.utils.data.DataLoader(train_dataset,
                                           batch_sampler=batch_sampler,
                                           collate_fn=data.collate_with_neg_fn,
                                           **kwargs)

  model_class = getattr(models, cfg.model)
  # increment n_attributes for padding_idx at index zero
  kwargs = dict(n_users=train_dataset.n_users,
                n_items=train_dataset.n_items,
                n_attributes=train_dataset.n_attributes + 1, 
                emb_size=cfg.emb_size,
                padding_idx=cfg.padding_idx, 
                sparse=cfg.sparse,
                dropout=cfg.dropout)
  if model_class == models.ResidualInnerProduct or model_class == models.Deep:
    kwargs['hidden_size'] = cfg.hidden_size
    kwargs['resnet'] = cfg.resnet

  model = model_class(**kwargs)
  model.reset_parameters()
  model.reset_padding_idx()
  model.to(device)

  # possibly warm start from linear model
  if cfg.inner_product_checkpoint is not None:
    checkpoint = torch.load(cfg.inner_product_checkpoint)
    model.inner_product_model.load_state_dict(checkpoint['model'])

  # binary cross entropy loss (negative Bernoulli log-likelihood)
  loss = torch.nn.BCEWithLogitsLoss()

  # need separate optimizer because sparse gradients do not support weight decay
  if isinstance(model, models.InnerProduct):
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.learning_rate, momentum=cfg.momentum)
  else:
    optimizer = torch.optim.SGD([
      {'params': model.inner_product_model.parameters()},
      {'params': model.net.parameters(), 'weight_decay': cfg.weight_decay}
      ], lr=cfg.learning_rate, momentum=cfg.momentum)
  if cfg.lr_decay == 'linear':
    # return multiplicative factor with which to decay
    # this schedule decays the learning rate to zero at the max iterations
    lr_lambda = lambda step: 1.0 - step / cfg.max_iterations
  elif cfg.lr_decay == 'plateau':
    # divide learning rate by ten if no improvement
    lr_lambda = lambda step: 0.1
  scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

  # dummy labels with negative samples in the last half of the batch
  labels = torch.Tensor((np.arange(batch_size) < batch_size // 2).astype(np.float32))
  labels = labels.to(device)

  best_valid_recall = -np.inf
  step = 0
  t0 = time.time()
  num_no_improvement = 0
  model.train()
  for batch in iter(util.cycle(train_data)):
    users, items, item_attributes, num_attributes = batch
    users = users.to(device)
    items = items.to(device)
    item_attributes = item_attributes.to(device)
    num_attributes = num_attributes.to(device)
    model.zero_grad()
    logits = model(users, items, item_attributes, num_attributes)
    L = loss(logits, labels)
    # for big batches, empty cache must be after loss and *before* both backward and optimizer
    torch.cuda.empty_cache()
    L.backward()
    optimizer.step()
    model.reset_padding_idx()
    step += 1
    if cfg.lr_decay == 'linear':
      scheduler.step()
    if step % 50 == 0:
      tmp_loss = L.detach().mean().cpu().numpy()
      print(step, tmp_loss)
      if np.isnan(tmp_loss):
        raise ValueError('Loss hit nan')
    if step % cfg.log_interval == 0:
      t1 = time.time()
      current_loss = np.mean(L.detach().cpu().numpy())
      speed = (t1 - t0) / cfg.log_interval
      logger.info(f'step {step}\tloss {current_loss:.2f}\tspeed {speed:.3e} s/iter\ttime: {t1 - t0:.2f}\tlr: {scheduler.get_lr()[0]}')
      average = 0.
      current_loss = 0.
      t0 = time.time()
      valid_recall = eval_hook(step, model, valid_dataset, 'valid')
      model.train()
      if valid_recall > best_valid_recall:
        num_no_improvement = 0
        logger.info('Valid recall improved')
        best_valid_recall = valid_recall
        states = {'model': model.state_dict(),
                  'optimizer': optimizer.state_dict()}
        torch.save(states, cfg.train_dir / 'best_state_dict')
      else:
        num_no_improvement += 1
        if cfg.lr_decay == 'plateau' and valid_recall <= best_valid_recall:
          checkpoint = torch.load(cfg.train_dir / 'best_state_dict')
          model.load_state_dict(checkpoint['model'])
          optimizer.load_state_dict(checkpoint['optimizer'])
          scheduler.step()
      if num_no_improvement >= 5 or step >= cfg.max_iterations:
        checkpoint = torch.load(cfg.train_dir / 'best_state_dict')
        model.load_state_dict(checkpoint['model'])
        eval_hook(step, model, test_dataset, 'test')
        logger.info("Halting - validation performance has not been improving")
        return
