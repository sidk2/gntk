# Oversmoothing in Graph Neural Networks

This repository contains code for node level GNTK, GNTK with skip connections, SSGC NTK, corresponding neural networks, and code to computer preactivations and plot the spectrum. 

This repository forks https://github.com/KangchengHou/gntk, which implements Graph Neural Tangent Kernel (infinitely wide multi-layer GNNs trained by gradient descent), described in the following paper:

Simon S. Du, Kangcheng Hou, Barnabás Póczos, Ruslan Salakhutdinov, Ruosong Wang, Keyulu Xu. Graph Neural Tangent Kernel: Fusing Graph Neural Networks with Graph Kernels. NeurIPS 2019. [[arXiv]](https://arxiv.org/abs/1905.13192) [[Paper]](https://papers.nips.cc/paper/8809-graph-neural-tangent-kernel-fusing-graph-neural-networks-with-graph-kernels)

## Experiment for all datasets
To reproduce our experiments, run param_sweep_full.py
Then, create plots with Plots.ipynb

Add instructions for other stuff.
