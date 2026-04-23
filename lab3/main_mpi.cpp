// main_mpi.cpp
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <chrono>
#include <mpi.h>

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

void appendResult(const string& filename, int n, int procs, double sec, double data_mb) {
    bool exists = false;
    ifstream check(filename);
    if (check.good()) exists = true;
    check.close();
    
    ofstream out(filename, ios::app);
    if (!exists) {
        out << "size,processes,seconds,data_mb\n";
    }
    out << n << "," << procs << "," << sec << "," << data_mb << "\n";
    out.close();
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (rank == 0) {
        cout << "=== MPI Matrix Multiplication ===\n";
        cout << "Processes: " << size << "\n\n";
    }
    
    int sizes[] = {200, 400, 800, 1200, 1600, 2000};
    
    for (int n : sizes) {
        vector<int> flatA, flatB;
        vector<long long> flatC;
        
        if (rank == 0) {
            string sn = to_string(n);
            auto A = readMatrix("matrices/a_" + sn + ".csv");
            auto B = readMatrix("matrices/b_" + sn + ".csv");
            
            if (A.empty() || B.empty()) {
                cerr << "Error loading matrices.\n";
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            
            flatA.resize(n * n);
            flatB.resize(n * n);
            flatC.resize(n * n, 0);
            
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    flatA[i * n + j] = A[i][j];
                    flatB[i * n + j] = B[i][j];
                }
            }
        }
        
        MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        if (rank != 0) {
            flatA.resize(n * n);
            flatB.resize(n * n);
            flatC.resize(n * n);
        }
        
        MPI_Bcast(flatA.data(), n * n, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(flatB.data(), n * n, MPI_INT, 0, MPI_COMM_WORLD);
        
        int rows_per_proc = n / size;
        int start_row = rank * rows_per_proc;
        int end_row = (rank == size - 1) ? n : start_row + rows_per_proc;
        
        MPI_Barrier(MPI_COMM_WORLD);
        auto start = high_resolution_clock::now();
        
        for (int i = start_row; i < end_row; i++) {
            for (int k = 0; k < n; k++) {
                int aik = flatA[i * n + k];
                for (int j = 0; j < n; j++) {
                    flatC[i * n + j] += (long long)aik * flatB[k * n + j];
                }
            }
        }
        
        auto end = high_resolution_clock::now();
        double elapsed = duration_cast<milliseconds>(end - start).count() / 1000.0;
        
        // Собираем результаты в процессе 0
        if (rank == 0) {
            MPI_Gather(MPI_IN_PLACE, rows_per_proc * n, MPI_LONG_LONG,
                       flatC.data(), rows_per_proc * n, MPI_LONG_LONG,
                       0, MPI_COMM_WORLD);
        } else {
            MPI_Gather(flatC.data() + start_row * n, rows_per_proc * n, MPI_LONG_LONG,
                       nullptr, 0, MPI_LONG_LONG,
                       0, MPI_COMM_WORLD);
        }
        
        if (rank == 0) {
            double data_mb = (3.0 * n * n * sizeof(int)) / (1024.0 * 1024.0);
            appendResult("results/times_mpi.csv", n, size, elapsed, data_mb);
            cout << "Size: " << n << "x" << n 
                 << ", Procs: " << size 
                 << ", Time: " << elapsed << " sec\n";
        }
    }
    
    if (rank == 0) {
        cout << "\n=== Done ===\n";
    }
    
    MPI_Finalize();
    return 0;
}
