import scipy.sparse
import util
import time
import numpy as np
import scipy
from os.path import join
import argparse
import os
from multiprocessing import Pool
from gntk import GNTK
import torch_geometric.transforms as Transform
from torch_geometric.datasets import Planetoid
from types import SimpleNamespace

import torch
import torch.nn.functional as F

num_classes = {"Cora" : 7, "Citeseer" : 6, "Pubmed" : 3}
def parse_arguments():
    parser = argparse.ArgumentParser(description='GNTK computation')
    parser.add_argument('--dataset', type=str, default="Cora",
                        help='name of dataset (default: Cora)')
    parser.add_argument('--num_mlp_layers', type=int, default=2,
                        help='number of mlp layers')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='number of layers')
    parser.add_argument('--scale', type=str, default='degree',
                        help='scaling methods')
    parser.add_argument('--jk', type=int, default=1,
                        help='whether to add jk')
    parser.add_argument('--out_dir', type=str, default="out",
                        help='output directory')
    parser.add_argument('--type', type=str, default='GCN', help='GCN or SSGC')
    parser.add_argument('--skip', type=int, default=0, help='whether to add skip pc')
    return parser.parse_args()

def one_hot_encode(numbers, num_classes):
    n = len(numbers)
    one_hot = np.zeros((n, num_classes), dtype=int)  # Initialize n x 7 matrix with zeros
    for i, j in enumerate(numbers):
        one_hot[i,j] = 1
    return one_hot

def main():
    args = parse_arguments()

    type = args.type

    # path = os.path.join(os.path.dirname(os.path.abspath('')), '..', 'data', 'Planetoid')
    path = '/home/sid/gntk/data'

    dataset = Planetoid(path, args.dataset, split='public', transform=Transform.NormalizeFeatures())

    all_labels = dataset[0].y.numpy().astype(int)
    one_hot_labs = one_hot_encode(all_labels, num_classes=num_classes[args.dataset])
    n = all_labels.shape[0]

    #Processing adjacency into scipy sparse coo format
    edge_index = dataset[0].edge_index
    row = edge_index[0].numpy()
    col = edge_index[1].numpy()

    #If the dataset has edge weights, this will need to be changed. I don't think Core, CiteSeer, or PubMed does though
    data = np.ones(edge_index.size(1))
    A = scipy.sparse.coo_matrix((data, (row, col)), shape=(n, n))

    #This is adding self loops. I don't think A.T needs to be added like in the original prepare graphs function
    A = A + scipy.sparse.identity(n, format='coo')
    degree_values = 1/np.sqrt(np.array(A.sum(axis=1)).flatten())  # Sum of each row
    degree_matrix = scipy.sparse.coo_matrix((degree_values, (range(n), range(n))), shape=(n, n))
    A = degree_matrix @ A @ degree_matrix
    if type == "SSGC":
        T = scipy.sparse.linalg.matrix_power(A, 0)
        for i in range(1, args.num_layers + 1):
            T = T + scipy.sparse.linalg.matrix_power(A, i)

        A = T / args.num_layers
        gntk = GNTK(num_layers=2, num_mlp_layers=args.num_mlp_layers, jk=args.jk, scale=args.scale, task='node')
    else:
        gntk = GNTK(num_layers=args.num_layers, num_mlp_layers=args.num_mlp_layers, jk=args.jk, scale=args.scale, task='node', skip_pc=args.skip)


    node_features = dataset[0].x.numpy()
    #Wrapper for the node features, so they can be accessed as g.node_features
    graph = SimpleNamespace(node_features=node_features)

    diag = gntk.diag(graph, A)

    ntk = gntk.gntk(graph, graph, diag, diag, A, A)


    train = dataset[0].train_mask
    test = dataset[0].test_mask

    y = one_hot_labs[train, :]

    H = ntk[train][:, train]
    
    if np.linalg.cond(H) > 1e6:
        U, S, Vt = np.linalg.svd(H)
        S_inv = np.diag(1 / S)
        H_inverse = Vt.T @ S_inv @ U.T
    else:
        H_inverse = np.linalg.inv(H)
    
    kdotHinv = np.dot(ntk[test][:,train], H_inverse)

    preds = torch.Tensor(np.dot(kdotHinv, y))
    
    probs = F.softmax(preds, dim=1)
    preds = np.argmax(probs.numpy(), axis=1)
    print(np.mean((preds - all_labels[test]) == 0))

    os.makedirs(args.out_dir, exist_ok=True)
    np.save(join(args.out_dir, 'ntk'), ntk)
    np.save(join(args.out_dir, 'preds'), preds)

if __name__ == "__main__":
    main()