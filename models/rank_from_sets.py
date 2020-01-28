import torch
from torch import nn

from models import networks


class InnerProduct(nn.Module):
  """Predict item consumption from inner product with user and item embeddings."""
  def __init__(self, n_users, n_items, n_attributes, emb_size, padding_idx, sparse, dropout):
    super().__init__()
    self.emb_size = emb_size
    self.user_embeddings = nn.Embedding(n_users, emb_size, sparse=sparse)
    self.attribute_emb_sum = nn.EmbeddingBag(n_attributes, emb_size, mode='sum', sparse=sparse)
    self.item_embeddings = nn.Embedding(n_items, emb_size, sparse=sparse)
    self.item_bias = nn.Embedding(n_items, 1, sparse=sparse)
    self.padding_idx = padding_idx
    self.dropout = nn.Dropout(dropout)

  def reset_parameters(self):
    for module in [self.user_embeddings, self.attribute_emb_sum]:
      scale = 0.07
      nn.init.uniform_(module.weight, -scale, scale)
    for module in [self.item_embeddings, self.item_bias]:
      # initializing item embeddings to non-zero prevents large batch sizes
      nn.init.constant_(module.weight, 0)

  def reset_padding_idx(self):
    with torch.no_grad():
      self.attribute_emb_sum.weight[self.padding_idx] = 0

  def forward(self, users, items, item_attributes, num_attributes, pairwise=False, return_intermediate=False):
    user_emb = self.user_embeddings(users)
    attribute_emb = self.attribute_emb_sum(item_attributes)
    # mean of attribute embeddings
    attribute_emb = attribute_emb / num_attributes.unsqueeze(-1)
    item_emb = self.item_embeddings(items)
    item_and_attr_emb = self.dropout(attribute_emb + item_emb)
    item_bias = self.item_bias(items)
    
    if pairwise:
      # for every user, compute inner product with every item
      # (users, emb_size) x (emb_size, items) -> (users, items)
      logits = user_emb @ item_and_attr_emb.t()
    else:
      # for every user, only compute inner product with corresponding minibatch element
      # (batch_size, 1, emb_size) x (batch_size, emb_size, 1) -> (batch_size, 1)
      logits = torch.bmm(user_emb.view(-1, 1, self.emb_size), 
                         (item_and_attr_emb).view(-1, self.emb_size, 1)).squeeze()

    if return_intermediate:
      return logits + item_bias.squeeze(), user_emb, attribute_emb, item_emb, item_bias
    else:
      return logits + item_bias.squeeze()


class Deep(nn.Module):
  """Use a neural network to predict whether a user consumed an item."""
  def __init__(self, n_users, n_items, n_attributes, emb_size, padding_idx, sparse, hidden_size, dropout, resnet):
    super().__init__()
    self.inner_product_model = InnerProduct(n_users, n_items, n_attributes, emb_size, padding_idx, sparse, dropout=0.0)
    # inputs to neural network: user, attribute, item embeddings, scalar item bias
    # output bias does not improve performance for larger embedding sizes
    if resnet:
      self.net = networks.ResidualNetwork(input_size=emb_size * 3 + 1, 
                                          output_size=1, 
                                          inplanes=emb_size, 
                                          hidden_size=hidden_size, 
                                          output_bias=False, 
                                          num_blocks=2)
    else:
      self.net = networks.NeuralNetwork(input_size=emb_size * 3, 
                             output_size=1, 
                             hidden_size=hidden_size,
                             dropout=dropout,

                             output_bias=False,
                                        batchnorm=False)

  def reset_parameters(self):
    self.inner_product_model.reset_parameters()

  def reset_padding_idx(self):
    self.inner_product_model.reset_padding_idx()

  def forward(self, users, items, item_attributes, num_attributes):
    user_emb = self.inner_product_model.user_embeddings(users)
    attribute_emb = self.inner_product_model.attribute_emb_sum(item_attributes)
    attribute_emb = attribute_emb / num_attributes.unsqueeze(-1)
    item_emb = self.inner_product_model.item_embeddings(items)
    item_bias = self.inner_product_model.item_bias(items).squeeze()
    logits = self.net(torch.cat((user_emb, attribute_emb, item_emb), dim=-1)).squeeze()
    return logits + item_bias


class ResidualInnerProduct(nn.Module):
  """Use a neural network to learn the residual of the inner product model."""
  def __init__(self, n_users, n_items, n_attributes, emb_size, padding_idx, sparse, hidden_size, dropout, resnet):
    super().__init__()
    self.inner_product_model = InnerProduct(n_users, n_items, n_attributes, emb_size, padding_idx, sparse, dropout=0.0)
    # inputs: user, attribute, item embeddings, item bias
    if resnet:
      self.net = networks.ResidualNetwork(input_size=emb_size * 3 + 1, 
                                          output_size=1, 
                                          inplanes=emb_size, 
                                          hidden_size=hidden_size, 
                                          output_bias=True, 
                                          num_blocks=2)
    else:
      self.net = networks.NeuralNetwork(input_size=emb_size * 3 + 1, 
                                        output_size=1, 
                                        hidden_size=hidden_size,
                                        dropout=dropout,
                                        # do not learn a scalar bias; in the inner product model
                                        # a global scalar bias decreases performance
                                        output_bias=True,
                                        batchnorm=False)


  def reset_parameters(self):
    self.inner_product_model.reset_parameters()

  def reset_padding_idx(self):
    self.inner_product_model.reset_padding_idx()
  
  def forward(self, users, items, item_attributes, num_attributes):
    logits, user_emb, attr_emb, item_emb, item_bias = self.inner_product_model(
      users, items, item_attributes, num_attributes, return_intermediate=True)
    residual = self.net(torch.cat((user_emb, attr_emb, item_emb, item_bias), dim=-1)).squeeze()
    return logits + residual
