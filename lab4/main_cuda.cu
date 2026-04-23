// main_cuda.cu
#include <iostream>
#include <fstream>
#include <vector>
#include <cuda_runtime.h>

using namespace std;

__global__ void multiply(const int* A, const int* B, long long* C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        long long sum = 0;
        for (int k = 0; k < n; k++) {
            sum += (long long)A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

int main() {
    int sizes[] = {200, 400, 800, 1200, 1600, 2000};
    int blocks[] = {8, 16, 32};

    ofstream f("results_cuda.csv");
    f << "size,block,time_ms\n";

    cout << "=== CUDA Start ===\n\n";

    // ПРОГРЕВ GPU
    int warm_n = 200;
    int *d_a, *d_b;
    long long *d_c;
    cudaMalloc(&d_a, warm_n * warm_n * 4);
    cudaMalloc(&d_b, warm_n * warm_n * 4);
    cudaMalloc(&d_c, warm_n * warm_n * 8);
    multiply<<<1,1>>>(d_a, d_b, d_c, warm_n);
    cudaDeviceSynchronize();
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    cout << "GPU warmup done\n\n";

    for (int n : sizes) {
        vector<int> a(n * n), b(n * n);
        for (int i = 0; i < n * n; i++) {
            a[i] = rand() % 1000;
            b[i] = rand() % 1000;
        }

        int *d_a, *d_b;
        long long *d_c;
        cudaMalloc(&d_a, n * n * 4);
        cudaMalloc(&d_b, n * n * 4);
        cudaMalloc(&d_c, n * n * 8);
        cudaMemcpy(d_a, a.data(), n * n * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, b.data(), n * n * 4, cudaMemcpyHostToDevice);

        for (int bs : blocks) {
            dim3 thr(bs, bs);
            dim3 grd((n + bs - 1) / bs, (n + bs - 1) / bs);

            // Прогоняем один раз без замера
            multiply<<<grd, thr>>>(d_a, d_b, d_c, n);
            cudaDeviceSynchronize();

            // Теперь замеряем
            float t;
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);

            cudaEventRecord(start);
            multiply<<<grd, thr>>>(d_a, d_b, d_c, n);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&t, start, stop);

            cudaEventDestroy(start);
            cudaEventDestroy(stop);

            cout << "  " << n << "x" << n << " | " << bs << "x" << bs << " | " << t << " ms\n";
            f << n << "," << bs << "," << t << "\n";
        }

        cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    }

    f.close();
    cout << "\n=== Done ===\n";
    return 0;
}