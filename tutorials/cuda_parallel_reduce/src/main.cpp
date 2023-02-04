#include <bits/stdc++.h>
#include <iostream>
#include "/usr/local/cuda/include/cuda_runtime_api.h"
#include "/usr/local/cuda/include/cuda.h"
#include <stdio.h>

#include "cuda_function.cu"

void fill_in_int(* int target, int N){
    for (i=0;i <= N; i++){
        target[i]=1;
    }
}

int main(){
    int *a, *b, *c; // host copies of a, b, c
    int *d_a, *d_b, *d_c; // device copies of a, b, c
    int size = N * sizeof(int);

    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    a = (int *)malloc(size); fill_in_int(a, N);
    b = (int *)malloc(size); fill_in_int(b, N);
    c = (int *)malloc(size);

    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    int dimGrid = 256;
    int dimBlock = 64;
    unsigned long smemSize = 512*sizeof(int);

    cudaMemcpy(d_c, c, sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << c[0] << std::endl;

    free(a); free(b); free(c);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    
}

