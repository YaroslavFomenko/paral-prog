import numpy as np
import os

print("=== Генерация матриц ===")

sizes = [200, 400, 800, 1200, 1600, 2000]

os.makedirs("matrices", exist_ok=True)
print("Папка matrices создана")

for n in sizes:
    print(f"  Размер {n}x{n}...", end=" ", flush=True)

    a = np.random.randint(0, 1001, size=(n, n))
    b = np.random.randint(0, 1001, size=(n, n))

    np.savetxt(f"matrices/a_{n}.csv", a, delimiter=",", fmt="%d")
    np.savetxt(f"matrices/b_{n}.csv", b, delimiter=",", fmt="%d")

    print("OK", flush=True)

print("=== Готово! ===")
