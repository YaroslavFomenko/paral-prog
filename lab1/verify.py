import numpy as np
import sys
import os


def main():
    if len(sys.argv) < 2:
        print("Usage: py verify.py <size>")
        sys.exit(1)

    size = sys.argv[1]

    file_a = f"matrices/a_{size}.csv"
    file_b = f"matrices/b_{size}.csv"
    file_c = f"matrices/c_{size}.csv"

    if not os.path.exists(file_c):
        print(f"ERROR: {file_c} not found")
        sys.exit(1)

    print(f"  Проверка размера {size}...", end=" ")

    a = np.loadtxt(file_a, delimiter=",")
    b = np.loadtxt(file_b, delimiter=",")
    c_cpp = np.loadtxt(file_c, delimiter=",")

    c_numpy = np.dot(a, b)

    if np.array_equal(c_cpp, c_numpy):
        print("OK")
        sys.exit(0)
    else:
        print("FAIL")
        diff = np.abs(c_cpp - c_numpy)
        print(f"  Max diff: {np.max(diff)}")
        sys.exit(1)


if __name__ == "__main__":
    main()