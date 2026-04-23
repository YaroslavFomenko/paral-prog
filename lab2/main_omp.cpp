// main_omp.cpp
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <chrono>
#include <omp.h>

using namespace std;
using namespace chrono;

vector<vector<int>> readMatrix(const string& filename) {
    vector<vector<int>> matrix;
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error opening: " << filename << endl;
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

void appendResult(const string& filename, int n, int threads, double sec, double data_mb) {
    bool exists = false;
    ifstream check(filename);
    if (check.good()) exists = true;
    check.close();

    ofstream out(filename, ios::app);
    if (!exists) {
        out << "size,threads,seconds,data_mb\n";
    }
    out << n << "," << threads << "," << sec << "," << data_mb << "\n";
    out.close();
}

vector<vector<long long>> multiplyOMP(const vector<vector<int>>& A, const vector<vector<int>>& B, int num_threads) {
    int n = A.size();
    vector<vector<long long>> C(n, vector<long long>(n, 0));

    omp_set_num_threads(num_threads);

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            long long sum = 0;
            for (int k = 0; k < n; k++) {
                sum += (long long)A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }
    return C;
}

int main() {
    int sizes[] = {200, 400, 800, 1200, 1600, 2000};
    int thread_counts[] = {1, 2, 4, 8};

    cout << "=== OpenMP Matrix Multiplication ===\n\n";

    for (int n : sizes) {
        string sn = to_string(n);
        string pathA = "matrices/a_" + sn + ".csv";
        string pathB = "matrices/b_" + sn + ".csv";

        auto A = readMatrix(pathA);
        auto B = readMatrix(pathB);

        if (A.empty() || B.empty()) {
            cerr << "Error loading matrices. Run gen.py first.\n";
            return 1;
        }

        double data_mb = (3.0 * n * n * sizeof(int)) / (1024.0 * 1024.0);

        for (int threads : thread_counts) {
            cout << "Size: " << n << "x" << n << ", Threads: " << threads << "\n";

            auto start = high_resolution_clock::now();
            auto C = multiplyOMP(A, B, threads);
            auto end = high_resolution_clock::now();
            double elapsed = duration_cast<milliseconds>(end - start).count() / 1000.0;

            appendResult("results/times_omp.csv", n, threads, elapsed, data_mb);

            cout << "  Time: " << elapsed << " sec, Data: " << data_mb << " MB\n";
        }
        cout << endl;
    }

    cout << "=== Done. Results in results/times_omp.csv ===\n";
    return 0;
}