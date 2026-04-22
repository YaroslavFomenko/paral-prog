#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <chrono>
#include <cstdlib>

using namespace std;
using namespace chrono;

vector<vector<int>> readMatrix(const string& filename) {
    vector<vector<int>> matrix;
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Ошибка открытия: " << filename << endl;
        return matrix;
    }
    string line;
    while (getline(file, line)) {
        vector<int> row;
        stringstream ss(line);
        string val;
        while (getline(ss, val, ',')) {
            row.push_back(stoi(val));
        }
        matrix.push_back(row);
    }
    file.close();
    return matrix;
}

void saveMatrix(const string& filename, const vector<vector<long long>>& matrix) {
    ofstream file(filename);
    if (!file.is_open()) {
        cerr << "Ошибка записи: " << filename << endl;
        return;
    }
    for (const auto& row : matrix) {
        for (size_t j = 0; j < row.size(); ++j) {
            file << row[j];
            if (j < row.size() - 1) file << ",";
        }
        file << "\n";
    }
    file.close();
}

void appendResult(const string& filename, int n, double sec) {
    bool exists = false;
    ifstream check(filename);
    if (check.good()) exists = true;
    check.close();

    ofstream out(filename, ios::app);
    if (!exists) {
        out << "size,seconds\n";
    }
    out << n << "," << sec << "\n";
    out.close();
}

vector<vector<long long>> multiply(const vector<vector<int>>& A, const vector<vector<int>>& B) {
    int n = A.size();
    vector<vector<long long>> C(n, vector<long long>(n, 0));

    for (int i = 0; i < n; i++) {
        for (int k = 0; k < n; k++) {
            int aik = A[i][k];
            for (int j = 0; j < n; j++) {
                C[i][j] += (long long)aik * B[k][j];
            }
        }
    }
    return C;
}

bool verify(int n) {
    string cmd = "py verify.py " + to_string(n);
    return system(cmd.c_str()) == 0;
}

int main() {
    int sizes[] = {200, 400, 800, 1200, 1600, 2000};

    cout << "=== Умножение матриц ===\n";
    cout << "=== С верификацией через NumPy ===\n\n";

    for (int n : sizes) {
        cout << "Размер: " << n << "x" << n << "\n";

        string sn = to_string(n);
        string pathA = "matrices/a_" + sn + ".csv";
        string pathB = "matrices/b_" + sn + ".csv";
        string pathC = "matrices/c_" + sn + ".csv";

        auto A = readMatrix(pathA);
        auto B = readMatrix(pathB);

        if (A.empty() || B.empty()) {
            cerr << "Ошибка загрузки матриц. Запустите gen.py\n";
            return 1;
        }

        cout << "  Вычисление... ";
        auto start = high_resolution_clock::now();
        auto C = multiply(A, B);
        auto end = high_resolution_clock::now();
        double elapsed = duration_cast<milliseconds>(end - start).count() / 1000.0;

        saveMatrix(pathC, C);
        appendResult("results/times.csv", n, elapsed);

        cout << elapsed << " сек.\n";

        cout << "  Верификация... ";
        if (verify(n)) {
            cout << "ПРОЙДЕНА\n";
        } else {
            cout << "ОШИБКА!\n";
        }
        cout << endl;
    }

    cout << "=== Завершено. Результаты в results/times.csv ===\n";
    return 0;
}