import pandas as pd
import matplotlib.pyplot as plt
import os

if not os.path.exists("results/times.csv"):
    print("Файл results/times.csv не найден. Сначала запустите main.exe")
    exit()

df = pd.read_csv("results/times.csv")
df = df.sort_values("size")

print("\n=== Таблица результатов ===\n")
print(df.to_string(index=False))

plt.figure(figsize=(10, 6))
plt.plot(df["size"], df["seconds"], 'o-', color='blue', linewidth=2, markersize=8)
plt.xlabel("Размер матрицы")
plt.ylabel("Время (сек)")
plt.title("Зависимость времени умножения от размера матрицы")
plt.grid(True, alpha=0.3)
plt.savefig("results/graph.png", dpi=150)
plt.show()

print("\nГрафик сохранён: results/graph.png")

with open("results/table.md", "w", encoding="utf-8") as f:
    f.write("| Размер | Время (сек) |\n")
    f.write("|--------|-------------|\n")
    for _, row in df.iterrows():
        f.write(f"| {int(row['size'])} | {row['seconds']:.3f} |\n")

print("Markdown таблица сохранена: results/table.md")