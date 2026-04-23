# plot_mpi.py
import pandas as pd
import matplotlib.pyplot as plt
import os

if not os.path.exists("results/times_mpi.csv"):
    print("File results/times_mpi.csv not found.")
    exit()

df = pd.read_csv("results/times_mpi.csv")

print("\n=== MPI Results Table ===\n")
print(df.to_string(index=False))

plt.figure(figsize=(12, 7))
for procs in sorted(df['processes'].unique()):
    subset = df[df['processes'] == procs]
    plt.plot(subset['size'], subset['seconds'], 'o-', label=f'{procs} processes')

plt.xlabel("Matrix size (N)")
plt.ylabel("Time (sec)")
plt.title("MPI Matrix Multiplication: Time vs Size")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("results/graph_mpi.png", dpi=150)
plt.show()

print("\nGraph saved: results/graph_mpi.png")
