#include <stdio.h>

// CUDA kernel for vector addition
__global__ void vectorAdd(int *a, int *b, int *c, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        c[index] = a[index] + b[index];
    }
}

int main() {
    int *a, *b, *c; // host vectors
    int *d_a, *d_b, *d_c; // device vectors
    int N; // Size of vectors

    // Input size of vectors from user
    printf("Enter the size of vectors: ");
    scanf("%d", &N);

    int size = N * sizeof(int);

    // Allocate memory for host vectors
    a = (int*)malloc(size);
    b = (int*)malloc(size);
    c = (int*)malloc(size);

    // Initialize host vectors
    printf("Enter elements for Vector A:\n");
    for (int i = 0; i < N; i++) {
        scanf("%d", &a[i]);
    }

    printf("Enter elements for Vector B:\n");
    for (int i = 0; i < N; i++) {
        scanf("%d", &b[i]);
    }

    // Allocate memory for device vectors
    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_c, size);

    // Copy host vectors to device
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch kernel
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);

    // Copy result back to host
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    // Print result vector
    printf("Result Vector:\n");
    for (int i = 0; i < N; i++) {
        printf("%d ", c[i]);
    }
    printf("\n");

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    // Free host memory
    free(a);
    free(b);
    free(c);

    return 0;
}
