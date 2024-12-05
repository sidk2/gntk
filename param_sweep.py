import numpy as np
import subprocess

if __name__=='__main__':
    mlp_layers = [1]
    block_ops = [1, 2, 4, 8, 12, 16]
    dataset = ["Cora", "Citeseer"]
    
    for d in dataset:
        foo = np.zeros((len(mlp_layers), len(block_ops)))
        for i, n_mlp in enumerate(mlp_layers):
            for j, n_block in enumerate(block_ops):
                print(f"Running on {d} with k={n_block} and {n_mlp} MLP layers")
                cmd = ['python', 'gram_node.py', '--dataset', f'{d}', '--num_mlp_layers', f'{n_mlp}', '--num_layers', str(n_block), '--scale', 'uniform', '--out_dir', 'out/', '--jk', '0', '--type', 'SSGC']
                output = subprocess.run(cmd, capture_output=True, text=True)
                print(output.stdout.strip())
                foo[i, j] = float(output.stdout.strip())
                
        np.save(f'{d}.npy', foo)