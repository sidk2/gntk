import torch
from torch_geometric.datasets import Planetoid
import numpy as np
import scipy.sparse as sp
from scipy.linalg import eigvalsh

import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams.update({'font.size': 22})

def process_dataset(dataset_name, ks=10):
    # Load dataset
    data_path = './data'  # Path to store the dataset
    dataset = Planetoid(root=data_path, name=dataset_name)
    data = dataset[0]

    # Extract the adjacency matrix
    edge_index = data.edge_index
    num_nodes = data.num_nodes

    # Create a sparse adjacency matrix in COO format
    row, col = edge_index
    adj_matrix = sp.coo_matrix((np.ones(row.shape[0]), (row.numpy(), col.numpy())),
                               shape=(num_nodes, num_nodes))

    # Add self-loops
    adj_matrix.setdiag(1)

    # Degree normalization
    degree = np.array(adj_matrix.sum(axis=1)).flatten()
    degree_inv_sqrt = np.power(degree, -0.5, where=degree > 0)
    degree_inv_sqrt[degree == 0] = 0
    D_inv_sqrt = sp.diags(degree_inv_sqrt)
    normalized_adj_matrix = D_inv_sqrt @ adj_matrix @ D_inv_sqrt

    median_diff, q1_diff, q3_diff = [], [], []
    median_power, q1_power, q3_power = [], [], []

    for i in range(1, ks):
        print(f"{dataset_name} - Iteration: {i}")
        normalized_adj_dense = normalized_adj_matrix.toarray()
        sum_matr_pow = np.zeros_like(normalized_adj_dense)
        matr_pow = np.linalg.matrix_power(normalized_adj_dense, i)  # Element-wise power

        # Compute cumulative sum of powers up to i
        for k in range(1, i + 1):
            sum_matr_pow += np.linalg.matrix_power(normalized_adj_dense, k)

        sum_matr_pow /= i  # Average matrix powers

        # Compute eigenvalues for averaged and powered matrices
        eigenvalues = np.sort(np.abs(eigvalsh(sum_matr_pow)))
        eigs_expm = np.sort(np.abs(eigvalsh(matr_pow)))

        # Collect statistics for averaged matrix powers
        median_diff.append(np.median(eigenvalues))
        q1_diff.append(np.percentile(eigenvalues, 25))
        q3_diff.append(np.percentile(eigenvalues, 75))

        # Collect statistics for powered matrices
        median_power.append(np.median(eigs_expm))
        q1_power.append(np.percentile(eigs_expm, 25))
        q3_power.append(np.percentile(eigs_expm, 75))

    return (median_diff, q1_diff, q3_diff, median_power, q1_power, q3_power)

# Process datasets
ks = 32
print("Processing Citeseer")
citeseer_results = process_dataset("Citeseer", ks)
print("Processing Cora")
cora_results = process_dataset("Cora", ks)
print("Processing Pubmed")
pubmed_results = process_dataset("Pubmed", ks)

# Plot line plots with shaded regions for all datasets in separate subplots
fig, axes = plt.subplots(1, 2, figsize=(24,12))

datasets = ["Citeseer", "Cora", "Pubmed"]
results = [citeseer_results, cora_results, pubmed_results]
colors = [("blue", "green"),("blue", "green"),("blue", "green")]

for ax, dataset_name, dataset_results, (diff_color, pow_color) in zip(axes, datasets, results, colors):
    median_diff, q1_diff, q3_diff, median_power, q1_power, q3_power = dataset_results

    # Plot for averaged matrix powers (Diffusion Kernel)
    ax.plot(range(1, ks), median_diff, label=f"{dataset_name} Median (Diffusion Kernel)", color=diff_color)
    ax.fill_between(range(1, ks), q1_diff, q3_diff, color=diff_color, alpha=0.3, label=f"{dataset_name} Q1-Q3 (Diffusion Kernel)")

    # Plot for powered matrices
    ax.plot(range(1, ks), median_power, label=f"{dataset_name} Median (l-th power)", linestyle="--", color=pow_color)
    ax.fill_between(range(1, ks), q1_power, q3_power, color=pow_color, alpha=0.2, label=f"{dataset_name} Q1-Q3 (l-th power)")
    ax.set_yscale("log")
    # Customize subplot
    ax.set_title(f"{dataset_name} - Median Eigenvalue with Quartile Ranges")
    ax.set_xlabel("Iteration (k)")
    ax.set_ylabel("Eigenvalue")
    ax.set_xticks(range(1, ks, 5))
    ax.set_xticklabels([f"k={i}" for i in range(1, ks, 5)])
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.legend()

# Adjust layout and save the figure
plt.tight_layout()
plt.savefig("lineplot_subplots_with_quartiles.png")
plt.show()
