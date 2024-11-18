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

def parse_arguments():
    parser = argparse.ArgumentParser(description='GNTK computation')
    parser.add_argument('--dataset', type=str, default="COLLAB",
                        help='name of dataset (default: COLLAB)')
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

def prepare_graphs(graphs, gntk):
    A_list = []
    diag_list = []
    
    for i in range(len(graphs)):
        n = len(graphs[i].neighbors)
        for j in range(n):
            graphs[i].neighbors[j].append(j)
        edges = graphs[i].g.edges
        row = [e[0] for e in edges]
        col = [e[1] for e in edges]

        A = scipy.sparse.coo_matrix(([1] * len(edges), (row, col)), shape=(n, n), dtype=np.float32)
        A = A + A.T + scipy.sparse.identity(n)
        diag = gntk.diag(graphs[i], A)

        A_list.append(A)
        diag_list.append(diag)
    
    return A_list, diag_list

def compute_kern(A, k):
    t = scipy.sparse.coo_matrix(np.zeros(A.shape))
    last_adj = scipy.sparse.identity(t.shape[0])
    
    for i in range(k + 1):
        t += last_adj
        last_adj = last_adj @ A
    
    return t

def calc(T, A_list, diag_list, graphs, gntk, use_k, k):
    adj_0 = A_list[T[0]] if not use_k else compute_kern(A_list[T[0]], k)
    adj_1 = A_list[T[1]] if not use_k else compute_kern(A_list[T[1]], k)
    return gntk.gntk(graphs[T[0]], graphs[T[1]], diag_list[T[0]], diag_list[T[1]], adj_0, adj_1)

def main():
    args = parse_arguments()
    degree_as_tag = args.dataset in ['IMDBBINARY', 'COLLAB', 'IMDBMULTI', 'COLLAB']
    use_k = args.use_diff_kern
    k = args.diff_kern_k

    graphs, _ = util.load_data(args.dataset, degree_as_tag)
    labels = np.array([g.label for g in graphs]).astype(int)

    gntk = GNTK(num_layers=args.num_layers, num_mlp_layers=args.num_mlp_layers, jk=args.jk, scale=args.scale)
    A_list, diag_list = prepare_graphs(graphs, gntk)

    calc_list = [(i, j) for i in range(len(graphs)) for j in range(i, len(graphs))]

    with Pool(12) as pool:
        print("Calculating GNTKs")
        results = pool.starmap(calc, [(T, A_list, diag_list, graphs, gntk, use_k, k) for T in calc_list])

    gram = np.zeros((len(graphs), len(graphs)))
    for t, v in zip(calc_list, results):
        gram[t[0], t[1]] = v
        gram[t[1], t[0]] = v

    os.makedirs(args.out_dir, exist_ok=True)
    np.save(join(args.out_dir, 'gram'), gram)
    np.save(join(args.out_dir, 'labels'), labels)

if __name__ == "__main__":
    print("running")
    main()
