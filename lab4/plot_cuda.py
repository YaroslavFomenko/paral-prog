import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results_cuda.csv")

# Таблица
print("\n=== Результаты ===\n")
print(df.to_string(index=False))

# График
plt.figure(figsize=(12, 7))
for bs in sorted(df['block'].unique()):
    s = df[df['block'] == bs]
    plt.plot(s['size'], s['time_ms'], 'o-', label=f'Block {bs}x{bs}')

plt.xlabel("Размер матрицы")
plt.ylabel("Время (мс)")
plt.title("CUDA: время умножения матриц")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("graph_cuda.png", dpi=150)
plt.show()
print("\nСохранено: graph_cuda.png")