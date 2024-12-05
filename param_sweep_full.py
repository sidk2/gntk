import numpy as np
import subprocess
import time

if __name__=='__main__':
    mlp_layers = [1]
    block_ops = [1, 2, 4, 8, 12, 16, 32]
    dataset = ["Cora", "Citeseer"]
    
    for d in dataset:
        foo = np.zeros((len(mlp_layers), len(block_ops)))
        for i, n_mlp in enumerate(mlp_layers):
            for j, n_block in enumerate(block_ops):
                print(f"Running GCN on {d} with k={n_block} and no jk or skip")
                t1 = time.time()
                cmd = ['python', 'gram_node.py', '--dataset', f'{d}', '--num_mlp_layers', f'{n_mlp}', '--num_layers', str(n_block), '--scale', 'uniform', '--out_dir', 'out/', '--jk', '0', '--type', 'GCN']
                output = subprocess.run(cmd, capture_output=True, text=True)
                print("Runtime: ", time.time() - t1)
                print("Accuracy: ", output.stdout.strip())
                foo[i, j] = float(output.stdout.strip())
                
        np.save(f'GCN_{d}.npy', foo)
        foo = np.zeros((len(mlp_layers), len(block_ops)))
        for i, n_mlp in enumerate(mlp_layers):
            for j, n_block in enumerate(block_ops):
                print(f"Running GCN on {d} with k={n_block} and jk")
                t2 = time.time()
                cmd = ['python', 'gram_node.py', '--dataset', f'{d}', '--num_mlp_layers', f'{n_mlp}', '--num_layers', str(n_block), '--scale', 'uniform', '--out_dir', 'out/', '--jk', '1', '--type', 'GCN']
                output = subprocess.run(cmd, capture_output=True, text=True)
                print("Runtime: ", time.time() - t2)
                print("Accuracy: ", output.stdout.strip())
                foo[i, j] = float(output.stdout.strip())
                
        np.save(f'GCN_jk_{d}.npy', foo)
        foo = np.zeros((len(mlp_layers), len(block_ops)))
        for i, n_mlp in enumerate(mlp_layers):
            for j, n_block in enumerate(block_ops):
                print(f"Running GCN on {d} with k={n_block} and skip")
                t3 = time.time()
                cmd = ['python', 'gram_node.py', '--dataset', f'{d}', '--num_mlp_layers', f'{n_mlp}', '--num_layers', str(n_block), '--scale', 'uniform', '--out_dir', 'out/', '--jk', '0', '--type', 'GCN', '--skip', '1'] 
                output = subprocess.run(cmd, capture_output=True, text=True)
                print("Runtime: ", time.time() - t3)
                print("Accuracy: ", output.stdout.strip())
                foo[i, j] = float(output.stdout.strip())
                
        np.save(f'GCN_skip_{d}.npy', foo)
        foo = np.zeros((len(mlp_layers), len(block_ops)))
        for i, n_mlp in enumerate(mlp_layers):
            for j, n_block in enumerate(block_ops):
                print(f"Running SSGC on {d} with k={n_block}")
                t4 = time.time()
                cmd = ['python', 'gram_node.py', '--dataset', f'{d}', '--num_mlp_layers', f'{n_mlp}', '--num_layers', str(n_block), '--scale', 'uniform', '--out_dir', 'out/', '--jk', '0', '--type', 'SSGC']
                output = subprocess.run(cmd, capture_output=True, text=True)
                print("Runtime: ", time.time() - t4)
                print("Accuracy: ", output.stdout.strip())
                foo[i, j] = float(output.stdout.strip())
                
        np.save(f'SSGC_{d}.npy', foo)