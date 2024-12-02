import numpy as np
import subprocess

if __name__=='__main__':
    mlp_layers = list(range(1, 5))
    block_ops = [1, 2, 4, 8, 12, 16, 20, 26, 32]
    dataset = ["Cora", "Citeseer"]
    
    for d in dataset:
        foo = np.zeros((len(mlp_layers), len(block_ops)))
        for i, n_mlp in enumerate(mlp_layers):
            for j, n_block in enumerate(block_ops):
                print(f"Running on {d} with k={n_block} and {n_mlp} MLP layers")
                cmd = ['python3', 'gram_node.py', '--dataset', f'{d}', '--num_mlp_layers', f'{n_mlp}', '--num_layers', str(n_block), '--scale', 'degree', '--out_dir', 'out/', '--type', 'SSGC']
                output = subprocess.run(cmd, capture_output=True, text=True)
                foo[i, j] = float(output.stdout.strip())
                print(output.stdout.strip())
        np.save(f'{d}_s2gc.npy', foo)