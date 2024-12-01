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
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from types import SimpleNamespace

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
    parser.add_argument('--use_diff_kern', type=bool, default=False)
    parser.add_argument('--diff_kern_k', type=int, default=5, help='diffusion kernel k value')
    return parser.parse_args()

def main():
    args = parse_arguments()
    use_k = args.use_diff_kern
    k = args.diff_kern_k

    gntk = GNTK(num_layers=args.num_layers, num_mlp_layers=args.num_mlp_layers, jk=args.jk, scale=args.scale, task='node')

    path = os.path.join(os.path.dirname(os.path.abspath('')), '..', 'data', 'Planetoid')
    dataset = Planetoid(path, args.dataset, split='full', transform=T.NormalizeFeatures())

    all_labels = dataset[0].y.numpy().astype(int)
    n = all_labels.shape[0]

    #Processing adjacency into scipy sparse coo format
    edge_index = dataset[0].edge_index
    row = edge_index[0].numpy()
    col = edge_index[1].numpy()

    #If the dataset has edge weights, this will need to be changed. I don't think Core, CiteSeer, or PubMed does though
    data = np.ones(edge_index.size(1))

    A = scipy.sparse.coo_matrix((data, (row, col)), shape=(n, n), dtype=np.float32)

    #This is adding self loops. I don't think A.T needs to be added like in the original prepare graphs function
    A = A + scipy.sparse.identity(n)

    node_features = dataset[0].x.numpy()
    #Wrapper for the node features, so they can be accessed as g.node_features
    graph = SimpleNamespace(node_features=node_features)

    diag = gntk.diag(graph, A)
    
    ntk = gntk.gntk(graph, graph, diag, diag, A, A)
    print("NTK: ")
    print(ntk)

    train = dataset[0].train_mask
    test = dataset[0].test_mask

    y = all_labels[train]

    H = ntk[train][:, train]
    H_inverse = np.linalg.inv(H)

    kdotHinv = np.dot(ntk[test][:,train], H_inverse)

    preds = np.dot(kdotHinv, y)

    print("Preds according to kernel regression from Arora paper (this doesn't seem to be working)")
    print(preds)

    os.makedirs(args.out_dir, exist_ok=True)
    np.save(join(args.out_dir, 'ntk'), ntk)
    np.save(join(args.out_dir, 'preds'), preds)

if __name__ == "__main__":
    print("running")
    main()
