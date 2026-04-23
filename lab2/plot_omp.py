# plot_omp.py
import pandas as pd
import matplotlib.pyplot as plt
import os

if not os.path.exists("results/times_omp.csv"):
    print("File results/times_omp.csv not found.")
    exit()

df = pd.read_csv("results/times_omp.csv")

print("\n=== OpenMP Results Table ===\n")
print(df.to_string(index=False))

# График времени
plt.figure(figsize=(12, 7))
for threads in df['threads'].unique():
    subset = df[df['threads'] == threads]
    plt.plot(subset['size'], subset['seconds'], 'o-', label=f'{threads} threads')

plt.xlabel("Matrix size (N)")
plt.ylabel("Time (sec)")
plt.title("OpenMP: Time vs Size for Different Thread Counts")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("results/graph_omp.png", dpi=150)
plt.show()

print("\nGraph saved: results/graph_omp.png")