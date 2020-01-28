import torch
from torch import nn


class NeuralNetwork(nn.Module):
  """Simple MLP."""
  def __init__(self, input_size, output_size, hidden_size, dropout, output_bias, batchnorm):
    super().__init__()
    modules = []
    if batchnorm:
      modules.append(nn.BatchNorm1d(input_size))
    modules.extend([nn.Dropout(dropout),
                    nn.Linear(input_size, hidden_size),
                    nn.Dropout(dropout),
                    nn.ReLU(inplace=True)])
    for _ in range(1):
      modules.extend([nn.Linear(hidden_size, hidden_size),
                      nn.Dropout(dropout),
                      nn.ReLU(inplace=True)])
    modules.append(nn.Linear(hidden_size, output_size, bias=output_bias))
    self.net = nn.Sequential(*modules)
    
  def forward(self, inputs):
    return self.net(inputs)


class Block(nn.Module):
  """Basic ResNet block with fully-connected layers."""
  def __init__(self, inplanes, hidden_size):
    super().__init__()
    self.fc1 = nn.Linear(inplanes, hidden_size)
    self.bn1 = nn.BatchNorm1d(hidden_size)
    self.relu = nn.ReLU(inplace=True)
    self.fc2 = nn.Linear(hidden_size, inplanes)
    self.bn2 = nn.BatchNorm1d(inplanes)

  def forward(self, x):
    identity = x
    out = self.fc1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.fc2(out)
    out = self.bn2(out)

    out += identity
    out = self.relu(out)
    return out


class ResidualNetwork(nn.Module):
  """Following https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py"""
  def __init__(self, input_size, output_size, inplanes, hidden_size, output_bias, num_blocks):
    super().__init__()
    self.fc1 = nn.Linear(input_size, inplanes)
    self.bn1 = nn.BatchNorm1d(inplanes)
    self.relu = nn.ReLU(inplace=True)
    self.layer1 = nn.Sequential(*[Block(inplanes, hidden_size) for _ in range(num_blocks)])
    self.layer2 = nn.Sequential(*[Block(inplanes, hidden_size) for _ in range(num_blocks)])
    self.layer3 = nn.Sequential(*[Block(inplanes, hidden_size) for _ in range(num_blocks)])
    self.layer4 = nn.Sequential(*[Block(inplanes, hidden_size) for _ in range(num_blocks)])
    self.fc_out = nn.Linear(inplanes, output_size, bias=output_bias)

    for m in self.modules():
      if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
      elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    
      # zero-initialize last batchnorm in every block
      if isinstance(m, Block):
        nn.init.constant_(m.bn2.weight, 0)

  def forward(self, x):
    x = self.fc1(x)
    x = self.bn1(x)
    x = self.relu(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    x = self.fc_out(x)
    return x
