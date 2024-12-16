import time

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric as pyg
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import to_dense_adj


class GCNLayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, weight_var: float, bias_var: float):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.linear = nn.Linear(in_features=input_dim, out_features=output_dim, bias=True)
        
        nn.init.normal_(self.linear.weight, mean=0.0, std=weight_var ** 0.5)
        if self.linear.bias is not None:
            nn.init.normal_(self.linear.bias, mean=0.0, std=bias_var ** 0.5)
    
    def reset_parameters(self,):
        nn.init.normal_(self.linear.weight, mean=0.0, std=1)
        if self.linear.bias is not None:
            nn.init.normal_(self.linear.bias, mean=0.0, std=1)
        
    def forward(self, features, adj_matr):
           # Shape: (N, out_features)

        # Step 2: Aggregate transformed features from neighbors
        aggregated_features = torch.matmul(adj_matr, features)
        transformed_features = self.linear(aggregated_features)
        return transformed_features

class GNN(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, weight_var, bias_var, activation=F.relu):
        """
        Args:
            input_dim (int): Number of input features per node.
            hidden_dims (list of int): List of hidden layer dimensions.
            output_dim (int): Number of output features per node (e.g., classes for node classification).
            weight_var (float): Variance of the Gaussian distribution for initializing weights.
            bias_var (float): Variance of the Gaussian distribution for initializing biases.
            activation (callable): Activation function applied after each layer (default: ReLU).
        """
        super().__init__()
        self.activation = activation

        # Define layers
        layers = []
        dims = [input_dim] + hidden_dims + [output_dim]
        for i in range(len(dims) - 1):
            layers.append(
                GCNLayer(
                    input_dim=dims[i],
                    output_dim=dims[i + 1],
                    weight_var=weight_var,
                    bias_var=bias_var
                )
            )
        self.layers = nn.ModuleList(layers)
        
    def reset_parameters(self,):
        self.layers[-2].reset_parameters()

    def forward(self, features, adj_matr):
        """
        Forward pass through the stacked GNN layers.

        Args:
            features (torch.Tensor): Node features of shape (N, input_dim).
            adj_matr (torch.Tensor): Adjacency matrix of shape (N, N).

        Returns:
            torch.Tensor: Output node features of shape (N, output_dim).
        """
        x = features
        for layer in self.layers[:-1]:  # Apply activation after all but the last layer
            x = self.activation(layer(x, adj_matr))
        x = self.layers[-1](x, adj_matr)  # No activation on the final layer
        return x

def relu_exp(cov: torch.Tensor):
    assert cov.shape == torch.Size([2,2]), "Assertion error. Wrong shape for covariance computation."
    lam = torch.clamp(cov[0,1]/torch.sqrt(cov[0,0])/torch.sqrt(cov[1,1]), min= -1, max =1)
    ret = (lam*(torch.pi - torch.acos(lam)) + torch.sqrt(1-lam**2))/2/torch.pi*torch.sqrt(cov[0,0])*torch.sqrt(cov[1,1])
    return ret


def compute_metric(x: torch.Tensor, adj: torch.Tensor, weight_variance: int, bias_variance: int) -> torch.Tensor:
    x = adj @ x
    num_samples, num_features = x.shape
    metric = bias_variance + weight_variance*(x @ x.T)
    return metric

def compute_layer_two_metr(cov: torch.Tensor, adj: torch.Tensor, weight_variance: float, bias_variance: float):
    """
    Computes the layer two metric using a 2D covariance matrix.

    Args:
        cov (torch.Tensor): Covariance matrix of shape (num_samples, num_samples).
        adj (torch.Tensor): Adjacency matrix of shape (num_samples, num_samples).
        weight_variance (float): Weight variance for scaling.
        bias_variance (float): Bias variance for scaling.

    Returns:
        torch.Tensor: Output metric matrix of shape (num_samples, num_samples).
    """
    num_samples = cov.shape[0]

    # Extract diagonal elements for normalization
    diag = torch.sqrt(torch.diag(cov))  # Shape: (num_samples,)
    diag_inv = 1 / diag
    diag_inv[torch.isinf(diag_inv)] = 0  # Handle divide-by-zero safely

    # Compute normalized correlation coefficients (lambda)
    outer_diag_inv = diag_inv[:, None] * diag_inv[None, :]  # Outer product for scaling
    lam = cov * outer_diag_inv  # Shape: (num_samples, num_samples)
    lam_clipped = torch.clamp(lam, -1 + 1e-7, 1 - 1e-7)  # Clip for numerical stability

    # Compute the ReLU expectation for all pairs
    relu_exps = (
        (lam_clipped * (torch.pi - torch.acos(lam_clipped)) + torch.sqrt(1 - lam_clipped**2))
        / (2 * torch.pi)
        * diag[:, None]
        * diag[None, :]
    )  # Shape: (num_samples, num_samples)
    # Compute metric matrix
    # return relu_exps
    metr = bias_variance + weight_variance * (adj @ relu_exps @ adj.T)
    return metr
    
    
data = Planetoid(root='data/Planetoid', name='Cora', split='public', transform=NormalizeFeatures())[0]
train_mask = data.train_mask
train_features = data.x[train_mask]

adj_matrix = to_dense_adj(data.edge_index)[0]
degree_matr = torch.diag(1/torch.sqrt(adj_matrix @ torch.ones(len(train_mask))))
norm_adj = (degree_matr @ adj_matrix @ degree_matr)

weight_variance = 1
bias_variance = 0

metr = compute_metric(data.x, norm_adj, weight_variance, bias_variance)
print("Computed Layer 1 Metric")
metr = compute_layer_two_metr(metr, norm_adj, weight_variance, bias_variance)
print("Computed Layer 2 Metric")

hiddens = [1, 10, 50, 100, 200, 500, 800, 1000, 1500, 2000]
pct_errs = []
for i in range(10):
    print(i)
    hidden_width = hiddens[i]
    output_dim = 1
    num_samples = hidden_width * 10
    samples = torch.zeros((num_samples,2708 ))
    inp = data.x.cuda()
    norm_adj = norm_adj.cuda()


    for i in range(num_samples):
        with torch.no_grad():
            model = GNN(input_dim=data.x.shape[-1], hidden_dims=[hidden_width], output_dim=output_dim, weight_var=weight_variance, bias_var=bias_variance).cuda()
            outs = model(inp, norm_adj)
            model.cpu()
            samples[i, :] = outs[:, 0]

    # Step 2: Center the data by subtracting the mean
    centered_samples = samples

    # Step 3: Compute the covariance matrix
    N = samples.size(0)  # Number of samples
    covariance_matrix = torch.mm(centered_samples.T, centered_samples) / (N - 1) / hidden_width
    print(f"Avg. relative error between matrix elements of the analytic and measured covariance, width = {hidden_width}: {torch.mean(torch.abs(covariance_matrix - metr) / torch.abs(metr))}")