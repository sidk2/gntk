import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
os.environ['TORCH'] = torch.__version__
print(torch.__version__)

import matplotlib.pyplot as plt

from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.nn import GCNConv, SSGConv

dataset = Planetoid(root='data/', name='Cora', transform=NormalizeFeatures(), split='public')

data = dataset[0]  # Get the first graph object.



class GCN(torch.nn.Module):
    def __init__(self, hidden_channels, hidden_layers):
        super().__init__()
        layers = [dataset.num_features] + [hidden_channels]*hidden_layers + [dataset.num_classes]
        self.convs = nn.ModuleList([GCNConv(layers[i], layers[i+1]) for i in range(len(layers ) -1)])

    def forward(self, x, edge_index):
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        return x

class SSGC(torch.nn.Module):
    def __init__(self, hidden_channels, hidden_layers):
        super().__init__()
        self.conv = SSGConv(in_channels = dataset.num_features, out_channels = hidden_channels, K=hidden_layers, alpha=0)
        self.lin = nn.Linear(hidden_channels, dataset.num_classes)
    def forward(self, x, edge_index):
        return self.lin(self.conv(x, edge_index))

class GCN(torch.nn.Module):
    def __init__(self, hidden_channels, hidden_layers):
        super().__init__()
        layers = [dataset.num_features] + [hidden_channels]*hidden_layers + [dataset.num_classes]
        self.convs = nn.ModuleList([GCNConv(layers[i], layers[i+1]) for i in range(len(layers ) -1)])

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = F.relu(conv(x, edge_index))
        return self.convs[-1](x, edge_index)

def train(model, optimizer, criterion):
      model.train()
      model = model.cuda()
      optimizer.zero_grad()  # Clear gradients.
      out = model(data.x.cuda(), data.edge_index.cuda())  # Perform a single forward pass.
      loss = criterion(out[data.train_mask], data.y[data.train_mask].cuda())  # Compute the loss solely based on the training nodes.
      loss.backward()  # Derive gradients.
      optimizer.step()  # Update parameters based on gradients.
      return model

def test(model):
      model.eval()
      out = model(data.x.cuda(), data.edge_index.cuda())
      pred = out.argmax(dim=1).cpu()  # Use the class with highest probability.
      test_correct = pred[data.test_mask] == data.y[data.test_mask]  # Check against ground-truth labels.
      test_acc = int(test_correct.sum()) / int(data.test_mask.sum())  # Derive ratio of correct predictions.
      return test_acc
  
hiddens = [1, 2, 4, 8, 16, 32]
num_trials = 20
num_epochs = 200

for h_depth in hiddens:
    accs = []
    for i in range(num_trials):
        model = GCN(hidden_channels=4096, hidden_layers=h_depth-1)
        max_test_acc = 0
        for j in range(num_epochs):
            model = train(model=model, optimizer=torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4), criterion = torch.nn.CrossEntropyLoss())
            test_acc = test(model)
            if test_acc > max_test_acc:
                max_test_acc = test_acc
        accs.append(max_test_acc)
    print(f"GCN with depth {h_depth} -  Mean: {np.mean(np.array(accs))}, Stdev: {np.std(np.array(accs))}")