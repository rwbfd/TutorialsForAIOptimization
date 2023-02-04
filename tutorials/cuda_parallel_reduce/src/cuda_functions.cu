template <unsigned int blockSize>
__device__ void warp_reduce(volatile int *sdata, unsigned int tid) {
    if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
    if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
    if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
    if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
    if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
    if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
}

template <unsigned int blockSize>
__global__ void reduce(int *g_idata, int *g_odata, unsigned int n) {
    extern __shared__ int sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockSize*2) + tid;
    unsigned int gridSize = blockSize*2*gridDim.x;
    sdata[tid] = 0;
    while (i < n) { sdata[tid] += g_idata[i] + g_idata[i+blockSize]; i += gridSize; }
    __syncthreads();
    if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
    if (blockSize >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }
    if (tid < 32) warp_reduce(sdata, tid);
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

__host__ __device__ void launch_kernel(int dimGrid, int dimBlock, unsigned  long smemSize, *int a, *int c){
    switch (threads)
    {
        case 512:
            reduce<<<dimGrid, dimBlock, smemSize>>>(a, c); break;
        case 256:
            reduce<256><<<dimGrid, dimBlock, smemSize>>>(a, c); break;
        case 128:
            reduce<128><<<dimGrid, dimBlock, smemSize>>>(a, c); break;
        case 64:
            reduce<64><<<dimGrid, dimBlock, smemSize>>>(a, c); break;
        case 32:
            reduce<32><<<dimGrid, dimBlock, smemSize>>>(a, c); break;
        case 16:
            reduce<16><<<dimGrid, dimBlock, smemSize>>>(a, c); break;
        case 8:
            reduce<8><<<dimGrid, dimBlock, smemSize>>>(a, c); break;
        case 4:
            reduce<4><<<dimGrid, dimBlock, smemSize>>>(a, c); break;
        case 2:
            reduce<2><<<dimGrid, dimBlock, smemSize>>>(a, c); break;
        case 1:
            reduce<1><<<dimGrid, dimBlock, smemSize>>>(a, c); break;
    }
}