// Matrix multiplication using shared memory.
//
// Each thread computes multiple elements in the output matrix, allowing for better shared memory
// usage within the limit of 1024 threads per block. No Tensor Cores are used.
//
// It is assumed that a dynamic allocation in shared memory aligns at least at the 256 bit boundary,
// which appears not to be stated by NVIDIA, but needs to be true for loading and storing fragments
// from/to shared memory (https://forums.developer.nvidia.com/t/alignment-requirements-shared-memory/305244).
//
// Tested: A100, L4.

#include <stdio.h>
#include <assert.h>
#include "cuda_fp16.h"

#include "matmul_test.h"

const int NUM_THREAD_VALS_MAX = 32;

inline
cudaError_t checkCuda(cudaError_t res) {
#if defined(DEBUG) || defined(_DEBUG)
    if (res != cudaSuccess) {
        fprintf(stderr, "CUDA error %s.\n", cudaGetErrorString(res));
        assert(res == cudaSuccess);
    }
#endif
    return res;
}

// Computes matrix multiplication (C = A * B) without Tensor Cores and with the following arguments:
// DT: half, float, or int,
// DT_ACC: half, float, or int,
// M, N, K: multiples of tile_dim,
// tile_dim: multiple of block_rows,
// block_rows: tile_dim / block_rows <= NUM_THREAD_VALS_MAX.
// The requested shared memory must be at least (2 * tile_dim * tile_dim + tile_dim) * sizeof(DT).
template<typename DT, typename DT_ACC>
__global__
void matmul_shmem_3_kernel(
    const DT *A,
    const DT *B,
    DT *C,
    int M,
    int N,
    int K,
    int tile_dim,
    int block_rows) {
    int offset_a = blockIdx.x * tile_dim * K;
    int offset_b = blockIdx.y * tile_dim;
    const int offset_c = blockIdx.x * tile_dim * N + blockIdx.y * tile_dim;
    DT_ACC vals_c[NUM_THREAD_VALS_MAX];

#pragma unroll
    for (int i = 0; i < tile_dim / block_rows; ++i) {
        vals_c[i] = static_cast<DT_ACC>(0);
    }

    extern __shared__ char shmem_[];
    DT* shmem = reinterpret_cast<DT *>(&shmem_[0]);

    const int shmem_offset_a = 0;
    const int shmem_offset_b = tile_dim * tile_dim;
    const int shmem_stride_a = tile_dim;
    const int shmem_stride_b = tile_dim + 1;

#pragma unroll
    for (int i = 0; i < K; i += tile_dim) {
#pragma unroll
        for (int j = 0; j < tile_dim; j += block_rows) {
            const int shmem_idx_a = shmem_offset_a + (j + threadIdx.y) * shmem_stride_a + threadIdx.x;
            const int shmem_idx_b = shmem_offset_b + threadIdx.x * shmem_stride_b + (j + threadIdx.y);
            const int idx_a = offset_a + (j + threadIdx.y) * K + threadIdx.x;
            const int idx_b = offset_b + (j + threadIdx.y) * N + threadIdx.x;
            shmem[shmem_idx_a] = A[idx_a];
            shmem[shmem_idx_b] = B[idx_b];
        }

        __syncthreads();

#pragma unroll
        for (int j = 0; j < tile_dim / block_rows; ++j) {
#pragma unroll
            for (int k = 0; k < tile_dim; ++k) {
                const int shmem_idx_a = shmem_offset_a + (j * block_rows + threadIdx.y) * shmem_stride_a + k;
                const int shmem_idx_b = shmem_offset_b + threadIdx.x * shmem_stride_b + k;
                vals_c[j] += static_cast<DT_ACC>(shmem[shmem_idx_a]) * static_cast<DT_ACC>(shmem[shmem_idx_b]);
            }
        }

        __syncthreads();

        offset_a += tile_dim;
        offset_b += tile_dim * N;
    }

#pragma unroll
    for (int i = 0; i < tile_dim / block_rows; ++i) {
        const int idx_c = offset_c + (i * block_rows + threadIdx.y) * N + threadIdx.x;
        C[idx_c] = static_cast<DT>(vals_c[i]);
    }
}

template
__global__
void matmul_shmem_3_kernel<half, float>(
    const half *A,
    const half *B,
    half *C,
    int M,
    int N,
    int K,
    int tile_dim,
    int block_rows);
template
__global__
void matmul_shmem_3_kernel<float, float>(
    const float *A,
    const float *B,
    float *C,
    int M,
    int N,
    int K,
    int tile_dim,
    int block_rows);
template
__global__
void matmul_shmem_3_kernel<int, int>(
    const int *A,
    const int *B,
    int *C,
    int M,
    int N,
    int K,
    int tile_dim,
    int block_rows);

template<typename DT>
int get_shmem_req(int tile_dim) {
    return (2 * tile_dim * tile_dim + tile_dim) * sizeof(DT);
}

template<typename DT, typename DT_ACC>
void matmul_shmem(
    const DT *A,
    const DT *B,
    DT *C,
    int M,
    int N,
    int K,
    int tile_dim,
    int block_rows) {
    assert(!(M % tile_dim || N % tile_dim || K % tile_dim));
    assert(!(tile_dim % block_rows));
    assert(tile_dim / block_rows <= NUM_THREAD_VALS_MAX);

    //Ampere: 164 * 1024, Ada: 100 * 1024;
    const int SHMEM_REQ = get_shmem_req<DT>(tile_dim);
    dim3 gridDim(M / tile_dim, N / tile_dim, 1);
    dim3 blockDim(tile_dim, block_rows, 1);
    if (SHMEM_REQ > 48 * 1024) {
        checkCuda(cudaFuncSetAttribute(matmul_shmem_3_kernel<DT, DT_ACC>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, SHMEM_REQ));
    }
    matmul_shmem_3_kernel<DT, DT_ACC><<<gridDim, blockDim, SHMEM_REQ>>>(
        A,
        B,
        C,
        M,
        N,
        K,
        tile_dim,
        block_rows);
}

template<typename DT, typename DT_ACC>
void RunCorrectnessTestSquare(
    const cudaDeviceProp *prop,
    int M,
    int N,
    int K,
    int tile_dim_start,
    int tile_dim_end,
    int num_reps,
    DT val_min,
    DT val_max,
    DT atol) {
    int mem_size_a = M * K * sizeof(DT);
    int mem_size_b = K * N * sizeof(DT);
    int mem_size_c = M * N * sizeof(DT);

    DT *A, *B, *C;

    checkCuda(cudaMalloc(&A, mem_size_a));
    checkCuda(cudaMalloc(&B, mem_size_b));
    checkCuda(cudaMalloc(&C, mem_size_c));

    for (int tile_dim = tile_dim_start; tile_dim <= tile_dim_end; tile_dim *= 2) {
        if (get_shmem_req<DT>(tile_dim) <= prop->sharedMemPerMultiprocessor) {
            matmul_test::MatmulTestSquare<DT, DT_ACC> mt(M, tile_dim, val_min, val_max);
            checkCuda(cudaMemcpy(A, mt.GetA(), mem_size_a, cudaMemcpyHostToDevice));
            checkCuda(cudaMemcpy(B, mt.GetB(), mem_size_b, cudaMemcpyHostToDevice));
            for (int block_rows = 1; block_rows <= tile_dim; block_rows *= 2) {
                if (tile_dim * block_rows <= prop->maxThreadsPerBlock &&
                    tile_dim / block_rows <= NUM_THREAD_VALS_MAX &&
                    !(tile_dim % block_rows)) {
                    bool res = true;
                    for (int i = 0; i < num_reps; ++i) {
                        checkCuda(cudaMemset(C, 0, mem_size_c));
                        matmul_test::MatDim dim = mt.GetMatDim();
                        matmul_shmem<DT, DT_ACC>(A, B, C, dim.m, dim.n, dim.k, tile_dim, block_rows);
                        checkCuda(cudaMemcpy(mt.GetRes(), C, mem_size_c, cudaMemcpyDeviceToHost));
                        res = res && mt.IsCorrect(mt.GetRes(), dim, atol);
                    }
                    printf("(%d, %d): %s\n", tile_dim, block_rows, res ? "passed" : "failed");
                }
            }
        }
    }

    checkCuda(cudaFree(A));
    checkCuda(cudaFree(B));
    checkCuda(cudaFree(C));
}

template<typename DT, typename DT_ACC>
void RunPerformanceTestSquare(
    const cudaDeviceProp *prop,
    int M,
    int N,
    int K,
    int tile_dim_start,
    int tile_dim_end,
    int num_reps,
    DT val_min,
    DT val_max,
    DT atol) {
    int mem_size_a = M * K * sizeof(DT);
    int mem_size_b = K * N * sizeof(DT);
    int mem_size_c = M * N * sizeof(DT);

    DT *A, *B, *C;

    checkCuda(cudaMalloc(&A, mem_size_a));
    checkCuda(cudaMalloc(&B, mem_size_b));
    checkCuda(cudaMalloc(&C, mem_size_c));

    cudaEvent_t start, stop;
    checkCuda(cudaEventCreate(&start));
    checkCuda(cudaEventCreate(&stop));

    for (int tile_dim = tile_dim_start; tile_dim <= tile_dim_end; tile_dim *= 2) {
        if (get_shmem_req<DT>(tile_dim) <= prop->sharedMemPerMultiprocessor) {
            matmul_test::MatmulTestSquare<DT, DT_ACC> mt(M, tile_dim, val_min, val_max);
            checkCuda(cudaMemcpy(A, mt.GetA(), mem_size_a, cudaMemcpyHostToDevice));
            checkCuda(cudaMemcpy(B, mt.GetB(), mem_size_b, cudaMemcpyHostToDevice));
            for (int block_rows = 1; block_rows <= tile_dim; block_rows *= 2) {
                if (tile_dim * block_rows <= prop->maxThreadsPerBlock &&
                    tile_dim / block_rows <= NUM_THREAD_VALS_MAX &&
                    !(tile_dim % block_rows)) {
                    float ms = 0.0f;
                    matmul_test::MatDim dim = mt.GetMatDimMax();
                    matmul_shmem<DT, DT_ACC>(A, B, C, dim.m, dim.n, dim.k, tile_dim, block_rows);
                    checkCuda(cudaEventRecord(start, 0));
                    for (int i = 0; i < num_reps; ++i) {
                        matmul_shmem<DT, DT_ACC>(A, B, C, dim.m, dim.n, dim.k, tile_dim, block_rows);
                    }
                    checkCuda(cudaEventRecord(stop, 0));
                    checkCuda(cudaEventSynchronize(stop));
                    checkCuda(cudaEventElapsedTime(&ms, start, stop));
                    printf("(%d, %d): %.2f\n", tile_dim, block_rows, 2 * M * N * 1e-9 * K * num_reps / ms);
                }
            }
        }
    }

    checkCuda(cudaFree(A));
    checkCuda(cudaFree(B));
    checkCuda(cudaFree(C));

    checkCuda(cudaEventDestroy(start));
    checkCuda(cudaEventDestroy(stop));
}

int main(void) {

    const int devId = 0;
    cudaDeviceProp prop;
    checkCuda(cudaGetDeviceProperties(&prop, devId));
    printf("\nDevice: %s\n", prop.name);
    printf("\nsharedMemPerMultiprocessor: %lu\n", prop.sharedMemPerMultiprocessor);
    checkCuda(cudaSetDevice(devId));

    // Correctness tests.

    int M = 256; //1024;
    int N = 256; //1024;
    int K = 256; //1024;

    printf("\n\n%-30s", "matmul_shmem_kernel, <half, float>, correctness:\n");
    RunCorrectnessTestSquare<half, float>(&prop, M, N, K, 32, 256, 2, -1.0f, 1.0f, 0.1f);

    printf("\n\n%-30s", "matmul_shmem_kernel, <float, float>, correctness:\n");
    RunCorrectnessTestSquare<float, float>(&prop, M, N, K, 32, 256, 2, -1.0f, 1.0f, 0.1f);

    printf("\n\n%-30s", "matmul_shmem_kernel, <int, int>, correctness:\n");
    RunCorrectnessTestSquare<int, int>(&prop, M, N, K, 32, 256, 2, -10, 10, 0);

    // Performance tests.

    M = 16384;
    N = 16384;
    K = 16384;

    printf("\n\n%-30s", "matmul_shmem_kernel, <half, float>, [TFLOPS]:\n");
    RunPerformanceTestSquare<half, float>(&prop, M, N, K, 32, 256, 2, -1.0f, 1.0f, 0.1f);

    printf("\n\n%-30s", "matmul_shmem_kernel, <float, float>, [TFLOPS]:\n");
    RunPerformanceTestSquare<float, float>(&prop, M, N, K, 32, 256, 2, -1.0f, 1.0f, 0.1f);

    printf("\n\n%-30s", "matmul_shmem_kernel, <int, int>, [TFLOPS]:\n");
    RunPerformanceTestSquare<int, int>(&prop, M, N, K, 32, 256, 2, -10, 10, 0);

    return 0;
}
