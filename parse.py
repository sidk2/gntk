import numpy as np

# Sample data as a string (replace this with reading from a file if needed)
data = """
GCN with depth 1 -  Mean: 0.7404000000000001, Stdev: 0.007844743462982083
GCN with depth 2 -  Mean: 0.7986500000000001, Stdev: 0.0034967842369811753
GCN with depth 4 -  Mean: 0.81745, Stdev: 0.00947879211714232
GCN with depth 8 -  Mean: 0.5802500000000002, Stdev: 0.03821109132176152
GCN with depth 16 -  Mean: 0.19805, Stdev: 0.004852576635149618
GCN with depth 32 -  Mean: 0.20050000000000004, Stdev: 0.006614378277661473

SSGC with depth 1 -  Mean: 0.7838999999999999, Stdev: 0.0017000000000000014
SSGC with depth 2 -  Mean: 0.80755, Stdev: 0.004779905856813469
SSGC with depth 4 -  Mean: 0.8257499999999999, Stdev: 0.0032229644738966667
SSGC with depth 8 -  Mean: 0.8295999999999999, Stdev: 0.00267207784317748
SSGC with depth 16 -  Mean: 0.8163999999999998, Stdev: 0.004454211490264004
SSGC with depth 32 -  Mean: 0.8017, Stdev: 0.004087786687193943
"""

# Initialize lists for GCN and SSGC
gcn_data = []
ssgc_data = []

# Parse the data line by line
for line in data.strip().split("\n"):
    if "GCN" in line:
        depth = int(line.split("with depth")[1].split("-")[0].strip())
        mean = float(line.split("Mean:")[1].split(",")[0].strip())
        stdev = float(line.split("Stdev:")[1].strip())
        gcn_data.append([depth, mean, stdev])
    elif "SSGC" in line:
        depth = int(line.split("with depth")[1].split("-")[0].strip())
        mean = float(line.split("Mean:")[1].split(",")[0].strip())
        stdev = float(line.split("Stdev:")[1].strip())
        ssgc_data.append([depth, mean, stdev])

# Convert lists to numpy arrays
gcn_array = np.array(gcn_data)
ssgc_array = np.array(ssgc_data)

np.save("gcn_network_cora.npy", gcn_data)
np.save("ssgc_network_cora.npy",ssgc_data)

# Output the arrays
print("GCN Array:")
print(gcn_array)

print("\nSSGC Array:")
print(ssgc_array)