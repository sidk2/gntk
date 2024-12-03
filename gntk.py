import math
import numpy as np
import scipy as sp
import time

class GNTK(object):
    """
    implement the Graph Neural Tangent Kernel
    """
    def __init__(self, num_layers, num_mlp_layers, jk, scale, task='graph', skip_pc = False):
        """
        num_layers: number of layers in the neural networks (including the input layer)
        num_mlp_layers: number of MLP layers
        jk: a bool variable indicating whether to add jumping knowledge
        scale: the scale used aggregate neighbors [uniform, degree]
        """
        self.num_layers = num_layers
        self.num_mlp_layers = num_mlp_layers
        self.jk = jk
        self.scale = scale
        assert(scale in ['uniform', 'degree'])
        self.task = task
        assert(task in ['graph', 'node'])
        self.skip_pc = skip_pc
    
    def __next_diag(self, S):
        """
        go through one normal layer, for diagonal element
        S: covariance of last layer
        """
        diag = np.sqrt(np.diag(S))
        S = S / diag[:, None] / diag[None, :]
        S = np.clip(S, -1, 1)
        # dot sigma
        DS = (math.pi - np.arccos(S)) / math.pi
        S = (S * (math.pi - np.arccos(S)) + np.sqrt(1 - S * S)) / np.pi
        S = S * diag[:, None] * diag[None, :]
        return S, DS, diag

    def __adj_diag(self, S, adj_block, N, scale_mat):
        """
        go through one adj layer
        S: the covariance
        adj_block: the adjacency relation
        N: number of vertices
        scale_mat: scaling matrix
        """
        adj_block = adj_block.tocoo()
        res = np.zeros((N*N, 1))
        for r, c, v in zip(adj_block.row, adj_block.col, adj_block.data):
            kron_entry = v*adj_block @ S[r]
            res[c*N:(c+1)*N] += kron_entry.reshape((N,1))
        return res.reshape(N, N) * scale_mat

    def __next(self, S, diag1, diag2):
        """
        go through one normal layer, for all elements
        """
        S = S / diag1[:, None] / diag2[None, :]
        S = np.clip(S, -1, 1)
        DS = (math.pi - np.arccos(S)) / math.pi
        S = (S * (math.pi - np.arccos(S)) + np.sqrt(1 - S * S)) / np.pi
        S = S * diag1[:, None] * diag2[None, :]
        return S, DS
    
    def __adj(self, S, adj_block, N1, N2, scale_mat):
        """
        go through one adj layer, for all elements
        """
        # For now, assuming N1 = N2
        adj_block = adj_block.tocoo()
        res = np.zeros((N1*N1, 1))
        for r, c, v in zip(adj_block.row, adj_block.col, adj_block.data):
            kron_entry = v*adj_block @ S[r]
            res[c*N1:(c+1)*N1] += kron_entry.reshape((N1,1))
        return res.reshape(N1, N1) * scale_mat
      
    def diag(self, g, A):
        """
        compute the diagonal element of GNTK for graph `g` with adjacency matrix `A`
        g: graph g
        A: adjacency matrix
        """
        N = A.shape[0]
        if self.scale == 'uniform':
            scale_mat = 1.
        else:
            scale_mat = 1. / np.array(np.sum(A, axis=1) * np.sum(A, axis=0))

        diag_list = []

        # input covariance
        sigma = np.matmul(g.node_features, g.node_features.T)
        sigma = self.__adj_diag(sigma, A, N, scale_mat)
        ntk = np.copy(sigma)
		
        
        for layer in range(1, self.num_layers):
            for mlp_layer in range(self.num_mlp_layers):
                sigma, dot_sigma, diag = self.__next_diag(sigma)
                diag_list.append(diag)
                ntk = ntk * dot_sigma + sigma
            # if not last layer
            if layer != self.num_layers - 1:
                sigma = self.__adj_diag(sigma, A, N, scale_mat)
                ntk = self.__adj_diag(ntk, A, N, scale_mat)
        return diag_list

    def gntk(self, g1, g2, diag_list1, diag_list2, A1, A2):
        """
        compute the GNTK value \Theta(g1, g2)
        g1: graph1
        g2: graph2
        diag_list1, diag_list2: g1, g2's the diagonal elements of covariance matrix in all layers
        A1, A2: g1, g2's adjacency matrix
        """

        n1 = A1.shape[0]
        n2 = A2.shape[0]
        
        if self.scale == 'uniform':
            scale_mat = 1.
        else:
            scale_mat = 1. / np.array(np.sum(A1, axis=1) * np.sum(A2, axis=0))
        
        A1 = A1.astype(np.uint8)
        A2 = A2.astype(np.uint8)
        # adj_block = sp.sparse.kron(A1, A2)
        
        jump_ntk = 0
        sigma = np.matmul(g1.node_features, g2.node_features.T)
        jump_ntk += sigma
        sigma = self.__adj(sigma, A1, n1, n2, scale_mat)
        ntk = np.copy(sigma)
        
        for layer in range(1, self.num_layers):
            for mlp_layer in range(self.num_mlp_layers):
                sigma, dot_sigma = self.__next(sigma, 
                                    diag_list1[(layer - 1) * self.num_mlp_layers + mlp_layer],
                                    diag_list2[(layer - 1) * self.num_mlp_layers + mlp_layer])
                ntk = ntk * dot_sigma + sigma
            jump_ntk += ntk
            # if not last layer
            if layer != self.num_layers - 1:
                sigma = self.__adj(sigma, A1, n1, n2, scale_mat)
                #Skip connection is slightly different from paper 
                if(layer == 1): sigma1 = np.copy(sigma)
                if(self.skip_pc and layer != 1): sigma += sigma1
                ntk = self.__adj(ntk, A1, n1, n2, scale_mat)
        if self.task == 'graph':
            if self.jk:
                return np.sum(jump_ntk) * 2
            else:
                return np.sum(ntk) * 2
        elif self.task == 'node':
            #Should this still be multiplied by 2? I don't know
            if self.jk:
                return jump_ntk * 2
            else:
                return ntk * 2
