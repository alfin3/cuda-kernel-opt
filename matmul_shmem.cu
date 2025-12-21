// Matrix multiplication using shared memory.
//
// Each thread computes multiple elements in the output matrix, allowing for
// better shared memory usage within the limit of 1024 threads per block.
// No Tensor Cores are used.
//
// Tested: A100, L4.

#include <stdio.h>
#include <assert.h>
#include "matmul_test.h"

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

template<int TILE_DIM, int BLOCK_ROWS, typename DT, typename DT_ACC>
__global__
void matmul_shmem_2_kernel(
    const DT *A,
    const DT *B,
    DT *C,
    int M) {
    int offset_a = blockIdx.y * TILE_DIM * M;
    int offset_b = blockIdx.x * TILE_DIM;
    DT_ACC vals_c[TILE_DIM / BLOCK_ROWS];

#pragma unroll
    for (int i = 0; i < TILE_DIM / BLOCK_ROWS; ++i) {
        vals_c[i] = static_cast<DT_ACC>(0);
    }

    __shared__ DT shmem_tile_a[TILE_DIM][TILE_DIM];
    __shared__ DT shmem_tile_b[TILE_DIM][TILE_DIM + 1];

#pragma unroll
    for (int i = 0; i < M; i += TILE_DIM) {
#pragma unroll
        for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
            const int shmem_x = threadIdx.x;
            const int shmem_y = j + threadIdx.y;
            shmem_tile_a[shmem_y][shmem_x] = A[offset_a + shmem_y * M + shmem_x];
            shmem_tile_b[shmem_x][shmem_y] = B[offset_b + shmem_y * M + shmem_x];
        }

        __syncthreads();

#pragma unroll
        for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
            const int shmem_x = threadIdx.x;
            const int shmem_y = j + threadIdx.y;
            const int val_idx = j / BLOCK_ROWS;
#pragma unroll
            for (int k = 0; k < TILE_DIM; ++k) {
                vals_c[val_idx] += static_cast<DT_ACC>(shmem_tile_a[shmem_y][k]) * static_cast<DT_ACC>(shmem_tile_b[shmem_x][k]);
            }
        }

        __syncthreads();

        offset_a += TILE_DIM;
        offset_b += TILE_DIM * M;
    }

#pragma unroll
    for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
        const int x = blockIdx.x * TILE_DIM + threadIdx.x;
        const int y = blockIdx.y * TILE_DIM + threadIdx.y + i;
        const int val_idx = i / BLOCK_ROWS;
        C[y * M + x] = static_cast<DT>(vals_c[val_idx]);
    }
}

template<int TILE_DIM, int BLOCK_ROWS, typename DT, typename DT_ACC>
void matmul_shmem_2(
    const DT *A,
    const DT *B,
    DT *C,
    int M,
    int N,
    int K) {
    assert(!(M % TILE_DIM || N % TILE_DIM || K % TILE_DIM));
    assert(!(TILE_DIM % BLOCK_ROWS));
    dim3 gridDim(M / TILE_DIM, N / TILE_DIM, 1);
    dim3 blockDim(TILE_DIM, BLOCK_ROWS, 1);
    matmul_shmem_2_kernel<TILE_DIM, BLOCK_ROWS, DT, DT_ACC><<<gridDim, blockDim>>>(
        A,
        B,
        C,
        M); // Currently square matrices.
}

// Includes dynamic allocation of shared memory with resolved bank conflicts
// in the transpose.
template<int TILE_DIM, int BLOCK_ROWS, typename DT, typename DT_ACC>
__global__
void matmul_shmem_3_kernel(
    const DT *A,
    const DT *B,
    DT *C,
    int M) {
    int offset_a = blockIdx.y * TILE_DIM * M;
    int offset_b = blockIdx.x * TILE_DIM;
    DT_ACC vals_c[TILE_DIM / BLOCK_ROWS];

#pragma unroll
    for (int i = 0; i < TILE_DIM / BLOCK_ROWS; ++i) {
        vals_c[i] = static_cast<DT_ACC>(0);
    }

    extern __shared__ DT shmem[];
    DT *shmem_tile_a = &shmem[0];
    DT *shmem_tile_b = &shmem[TILE_DIM * TILE_DIM];
    const int shmem_tile_width_a = TILE_DIM;
    const int shmem_tile_width_b = TILE_DIM + 1;

#pragma unroll
    for (int i = 0; i < M; i += TILE_DIM) {
#pragma unroll
        for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
            const int shmem_x = threadIdx.x;
            const int shmem_y = j + threadIdx.y;
            shmem_tile_a[shmem_y * shmem_tile_width_a + shmem_x] = A[offset_a + shmem_y * M + shmem_x];
            shmem_tile_b[shmem_x * shmem_tile_width_b + shmem_y] = B[offset_b + shmem_y * M + shmem_x];
        }

        __syncthreads();

#pragma unroll
        for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
            const int shmem_x = threadIdx.x;
            const int shmem_y = j + threadIdx.y;
            const int val_idx = j / BLOCK_ROWS;
#pragma unroll
            for (int k = 0; k < TILE_DIM; ++k) {
                vals_c[val_idx] += static_cast<DT_ACC>(shmem_tile_a[shmem_y * shmem_tile_width_a + k]) *
                    static_cast<DT_ACC>(shmem_tile_b[shmem_x * shmem_tile_width_b + k]);
            }
        }

        __syncthreads();

        offset_a += TILE_DIM;
        offset_b += TILE_DIM * M;
    }

#pragma unroll
    for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
        const int x = blockIdx.x * TILE_DIM + threadIdx.x;
        const int y = blockIdx.y * TILE_DIM + threadIdx.y + i;
        const int val_idx = i / BLOCK_ROWS;
        C[y * M + x] = static_cast<DT>(vals_c[val_idx]);
    }
}

template<int TILE_DIM, int BLOCK_ROWS, typename DT, typename DT_ACC>
void matmul_shmem_3(
    const DT *A,
    const DT *B,
    DT *C,
    int M,
    int N,
    int K) {
    assert(!(M % TILE_DIM || N % TILE_DIM || K % TILE_DIM));
    assert(!(TILE_DIM % BLOCK_ROWS));

    //Ampere: 164 * 1024, Ada: 100 * 1024;
    const int SHMEM_REQ = (TILE_DIM * TILE_DIM + TILE_DIM * (TILE_DIM + 1)) * sizeof(DT);
    dim3 gridDim(M / TILE_DIM, N / TILE_DIM, 1);
    dim3 blockDim(TILE_DIM, BLOCK_ROWS, 1);
    matmul_shmem_3_kernel<TILE_DIM, BLOCK_ROWS, DT, DT_ACC><<<gridDim, blockDim, SHMEM_REQ>>>(
        A,
        B,
        C,
        M); // Currently square matrices.
}

int main(void) {
    const int M = 16384; // Is multiple of TILE_DIM; currently square matrices.
    const int TILE_DIM = 64; // 64 is optimal for A100 and likely L4.
    const int BLOCK_ROWS = 2; // 2 is optimal for A100 and likely L4.
    const int num_reps_corr = 10;
    const int num_reps_perf = 10;
    float *A, *B, *C;

    matmul_test::MatmulTestSquare<float> mt_32(1024, 32, -1.0f, 1.0f);
    matmul_test::MatmulTestSquare<float> mt_64(1024, 64, -1.0f, 1.0f);
    matmul_test::MatmulTestSquare<float> mt(M, TILE_DIM, -1.0f, 1.0f);

    // mem_size_corr must be less or equal to mem_size_perf.
    const int mem_size_corr = 1024 * 1024 * sizeof(float);
    const int mem_size_perf = mt.GetMatDimMax().m * mt.GetMatDimMax().m * sizeof(float);
    checkCuda(cudaMalloc(&A, mem_size_perf));
    checkCuda(cudaMalloc(&B, mem_size_perf));
    checkCuda(cudaMalloc(&C, mem_size_perf));

    float ms = 0.0f;
    cudaEvent_t start, stop;
    checkCuda(cudaEventCreate(&start));
    checkCuda(cudaEventCreate(&stop));

    const int devId = 0;
    cudaDeviceProp prop;
    checkCuda(cudaGetDeviceProperties(&prop, devId));
    printf("\nDevice: %s\n", prop.name);
    printf("\nsharedMemPerBlock: %lu\n", prop.sharedMemPerBlock);
    printf("\nsharedMemPerMultiprocessor: %lu\n", prop.sharedMemPerMultiprocessor);
    printf("\nmaxThreadsPerBlock: %d\n", prop.maxThreadsPerBlock);
    printf("\nmaxThreadsDim: %d, %d, %d\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1],
           prop.maxThreadsDim[2]);
    printf("\nRequired sharedMemPerBlock: %lu\n",
           (TILE_DIM * TILE_DIM + TILE_DIM * (TILE_DIM + 1)) * sizeof(float));
    checkCuda(cudaSetDevice(devId));

    printf("\n\n%-30s", "matmul_shmem_2_kernel, correctness");
    bool res = true;
    checkCuda(cudaMemcpy(A, mt_32.GetA(), mem_size_corr, cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(B, mt_32.GetB(), mem_size_corr, cudaMemcpyHostToDevice));
    for (int i = 0; i < num_reps_corr; ++i) {
        checkCuda(cudaMemset(C, 0, mem_size_corr));
        matmul_test::MatDim dim = mt_32.GetMatDim();
        matmul_shmem_2<32, BLOCK_ROWS, float, float>(A, B, C, dim.m, dim.n, dim.k);
        checkCuda(cudaMemcpy(mt_32.GetC(), C, mem_size_corr, cudaMemcpyDeviceToHost));
        res = res && mt_32.IsCorrect(mt_32.GetC(), dim, 0.001f);
    }
    checkCuda(cudaMemcpy(A, mt_64.GetA(), mem_size_corr, cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(B, mt_64.GetB(), mem_size_corr, cudaMemcpyHostToDevice));
    for (int i = 0; i < num_reps_corr; ++i) {
        checkCuda(cudaMemset(C, 0, mem_size_corr));
        matmul_test::MatDim dim = mt_64.GetMatDim();
        matmul_shmem_2<64, BLOCK_ROWS, float, float>(A, B, C, dim.m, dim.n, dim.k);
        checkCuda(cudaMemcpy(mt_64.GetC(), C, mem_size_corr, cudaMemcpyDeviceToHost));
        res = res && mt_64.IsCorrect(mt_64.GetC(), dim, 0.001f);
    }
    if (res) {
        printf("\npassed\n");
    } else {
        printf("\nfailed\n");
    }

    printf("\n\n%-30s", "matmul_shmem_2_kernel, [TFLOPS]");
    matmul_test::MatDim dim = mt.GetMatDimMax();
    matmul_shmem_2<TILE_DIM, BLOCK_ROWS, float, float>(A, B, C, dim.m, dim.n, dim.k);
    checkCuda(cudaEventRecord(start, 0));
    for (int i = 0; i < num_reps_perf; ++i) {
        matmul_shmem_2<TILE_DIM, BLOCK_ROWS, float, float>(A, B, C, dim.m, dim.n, dim.k);
    }
    checkCuda(cudaEventRecord(stop, 0));
    checkCuda(cudaEventSynchronize(stop));
    checkCuda(cudaEventElapsedTime(&ms, start, stop));
    printf("\n%.2f\n", 2 * dim.m * dim.n * 1e-9 * dim.k * num_reps_perf / ms);

    printf("\n\n%-30s", "matmul_shmem_3_kernel, correctness");
    res = true;
    checkCuda(cudaMemcpy(A, mt_32.GetA(), mem_size_corr, cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(B, mt_32.GetB(), mem_size_corr, cudaMemcpyHostToDevice));
    for (int i = 0; i < num_reps_corr; ++i) {
        checkCuda(cudaMemset(C, 0, mem_size_corr));
        matmul_test::MatDim dim = mt_32.GetMatDim();
        matmul_shmem_3<32, BLOCK_ROWS, float, float>(A, B, C, dim.m, dim.n, dim.k);
        checkCuda(cudaMemcpy(mt_32.GetC(), C, mem_size_corr, cudaMemcpyDeviceToHost));
        res = res && mt_32.IsCorrect(mt_32.GetC(), dim, 0.001f);
    }
    checkCuda(cudaMemcpy(A, mt_64.GetA(), mem_size_corr, cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(B, mt_64.GetB(), mem_size_corr, cudaMemcpyHostToDevice));
    for (int i = 0; i < num_reps_corr; ++i) {
        checkCuda(cudaMemset(C, 0, mem_size_corr));
        matmul_test::MatDim dim = mt_64.GetMatDim();
        matmul_shmem_3<64, BLOCK_ROWS, float, float>(A, B, C, dim.m, dim.n, dim.k);
        checkCuda(cudaMemcpy(mt_64.GetC(), C, mem_size_corr, cudaMemcpyDeviceToHost));
        res = res && mt_64.IsCorrect(mt_64.GetC(), dim, 0.001f);
    }
    if (res) {
        printf("\npassed\n");
    } else {
        printf("\nfailed\n");
    }

    printf("\n\n%-30s", "matmul_shmem_3_kernel, [TFLOPS]");
    dim = mt.GetMatDimMax();
    matmul_shmem_3<TILE_DIM, BLOCK_ROWS, float, float>(A, B, C, dim.m, dim.n, dim.k);
    checkCuda(cudaEventRecord(start, 0));
    for (int i = 0; i < num_reps_perf; ++i) {
        matmul_shmem_3<TILE_DIM, BLOCK_ROWS, float, float>(A, B, C, dim.m, dim.n, dim.k);
    }
    checkCuda(cudaEventRecord(stop, 0));
    checkCuda(cudaEventSynchronize(stop));
    checkCuda(cudaEventElapsedTime(&ms, start, stop));
    printf("\n%.2f\n", 2 * dim.m * dim.n * 1e-9 * dim.k * num_reps_perf / ms);

    checkCuda(cudaFree(A));
    checkCuda(cudaFree(B));
    checkCuda(cudaFree(C));
    checkCuda(cudaEventDestroy(start));
    checkCuda(cudaEventDestroy(stop));

    return 0;
}
