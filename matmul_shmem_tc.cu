// GEMM (D = alpha * A * B + beta * C) using shared memory and Tensor Cores.
//
// The kernel enables changing i) the square tile dimension from 64 to 128, ii) the segment of the K
// dimension loaded into the shared memory, and iii) the number of warps from 4 to 8, while providing
// vectorized and coalesced (to the extent possible) load and store accesses for copying data from/to
// shared memory.
//
// This design enables tuning the Tensor Cores workload that each warp executes. Doubling the tile
// dimension increases the warp tile computed by a warp by a factor of 4, and doubling the
// number of warps decreases the warp tile by a factor of 2, thereby enabling to gradually tune the
// Tensor Cores workload that each warp executes.
//
// The input matrices can be of float and half types. It is assumed that a dynamic allocation
// in shared memory aligns at the 256 bit boundary, which appears not to be stated by NVIDIA, but
// needs to be true for loading and storing fragments from/to shared memory
// (https://forums.developer.nvidia.com/t/alignment-requirements-shared-memory/305244).
//
// To test: A100, L4.

#include <stdio.h>
#include <assert.h>
#include <cuda.h>
#include <mma.h>

#include "matmul_test.h"

using namespace nvcuda;

const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;

const int WARP_TILE_ROWS_MAX = 4;
const int WARP_TILE_COLS_MAX = 4;

const int SKEW_HALF = 16;

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

template<typename DT>
__device__
void load_tile(
    DT *shmem_ptr,
    const DT *a_ptr,
    int rows,
    int cols,
    int offset_a,
    int stride_am,
    int shmem_skew, // SKEW_HALF for A and B tiles, and 0 for C tiles.
    int num_warps,
    int warp_id,
    int lane_id) {
    const int len_vec = sizeof(int4) / sizeof(DT);
#pragma unroll
    for (int shmem_offset = 0; shmem_offset < rows * cols; shmem_offset += num_warps * warpSize * len_vec) {
        const int temp = shmem_offset + (warp_id * warpSize + lane_id) * len_vec;
        const int shmem_idx = temp + (temp / cols) * shmem_skew;
        const int idx = offset_a + (temp / cols) * stride_am + temp % cols;
        *reinterpret_cast<int4 *>(shmem_ptr + shmem_idx) = *reinterpret_cast<const int4 *>(a_ptr + idx);
    }
}

template<typename DT>
__device__
void store_tile(
    DT *a_ptr,
    const DT *shmem_ptr,
    int rows,
    int cols,
    int offset_a,
    int stride_am,
    int shmem_skew, // SKEW_HALF for A and B tiles, and 0 for C tiles.
    int num_warps,
    int warp_id,
    int lane_id) {
    const int len_vec = sizeof(int4) / sizeof(DT);
#pragma unroll
    for (int shmem_offset = 0; shmem_offset < rows * cols; shmem_offset += num_warps * warpSize * len_vec) {
        const int temp = shmem_offset + (warp_id * warpSize + lane_id) * len_vec;
        const int shmem_idx = temp + (temp / cols) * shmem_skew;
        const int idx = offset_a + (temp / cols) * stride_am + temp % cols;
        *reinterpret_cast<int4 *>(a_ptr + idx) = *reinterpret_cast<const int4 *>(shmem_ptr + shmem_idx);
    }
}

// Computes GEMM (D = alpha * A * B + beta * C) with Tensor Cores and the following arguments:
// TILE_DIM: 64 or 128,
// SEGMENT_K_DIM: (TILE_DIM / 2) increments,
// DT: half,
// DT_ACC: half or float,
// M, N, K: multiples of TILE_DIM,
// num_warps: 4 or 8.
// The requested shared memory must be at least the greater of (TILE_DIM * TILE_DIM * sizeof(DT_ACC))
// and (TILE_DIM * (SEGMENT_K_DIM + SKEW_HALF) * sizeof(DT)).
template<int TILE_DIM, int SEGMENT_K_DIM, typename DT, typename DT_ACC>
__global__
void gemm_tensor_core_0_kernel(
    const DT *A,
    const DT *B, // Transpose.
    const DT_ACC *C,
    DT_ACC *D,
    DT_ACC alpha,
    DT_ACC beta,
    int M,
    int N,
    int K,
    int num_warps) {
    const int warp_id = threadIdx.x / warpSize;
    const int lane_id = threadIdx.x % warpSize;

    // The dimensions of a warp tile computed by a warp are as follows:
    // 4 warps, TILE_DIM 64 -> 2x2 in terms of fragment tiles,
    // 8 warps, TILE_DIM 64 -> 1x2 in terms of fragment tiles,
    // 4 warps, TILE_DIM 128 -> 4x4 in terms of fragment tiles,
    // 8 warps, TILE_DIM 128 -> 2x4 in terms of fragment tiles.
    const int warp_tile_cols = 4 >> (TILE_DIM == 64);
    const int warp_tile_rows = warp_tile_cols >> (num_warps == 8);

    extern __shared__ DT shmem[][SEGMENT_K_DIM + SKEW_HALF];

    // Assign the computation of C tiles to SMs.
    for (int num_tiles_prev = 0;; num_tiles_prev += gridDim.x) {
        if ((num_tiles_prev + blockIdx.x) * (TILE_DIM * TILE_DIM) >= (M * N)) break;

        const int tile_offset_cd =
            ((num_tiles_prev + blockIdx.x) * TILE_DIM) / N * TILE_DIM  + ((num_tiles_prev + blockIdx.x) * TILE_DIM) % N;
        load_tile<DT_ACC>(reinterpret_cast<DT_ACC *>(&shmem[0][0]), C,
            TILE_DIM, TILE_DIM, tile_offset_cd, N, 0, num_warps, warp_id, lane_id);

        __syncthreads();

        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, DT_ACC> warp_tile_c[WARP_TILE_ROWS_MAX][WARP_TILE_COLS_MAX];

        // There are two columns of warp tiles in a tile.
        const int shmem_offset_warp =
            (warp_id / 2) * (TILE_DIM * warp_tile_rows * WMMA_M) + (warp_id % 2) * (warp_tile_cols * WMMA_N);
#pragma unroll
        for (int i = 0; i < warp_tile_rows; ++i) {
#pragma unroll
            for (int j = 0; j < warp_tile_cols; ++j) {
                const int shmem_idx = shmem_offset_warp + i * WMMA_M * TILE_DIM + j * WMMA_N;
                wmma::load_matrix_sync(warp_tile_c[i][j], reinterpret_cast<const DT_ACC *>(&shmem[0][0]) + shmem_idx,
                    TILE_DIM, wmma::mem_row_major);
            }
        }

        __syncthreads();

#pragma unroll
        for (int i = 0; i < warp_tile_rows; ++i) {
#pragma unroll
            for (int j = 0; j < warp_tile_cols; ++j) {
#pragma unroll
                for (int k = 0; k < warp_tile_c[i][j].num_elements; ++k) {
                    warp_tile_c[i][j].x[k] *= beta;
                }
            }
        }

        // Copy the tiles of A and B into the shared memory, iterating in the K dimension.
        const int shmem_stride = SEGMENT_K_DIM + SKEW_HALF;
#pragma unroll
        for (int segment_offset = 0; segment_offset < K; segment_offset += SEGMENT_K_DIM) {
            const int segment_offset_a = (tile_offset_cd / N) * K + segment_offset;
            const int segment_offset_b = (tile_offset_cd % N) * K + segment_offset;
            load_tile<DT>(&shmem[0][0], A,
                TILE_DIM, SEGMENT_K_DIM, segment_offset_a, K, SKEW_HALF, num_warps, warp_id, lane_id);
            load_tile<DT>(&shmem[0][0] + TILE_DIM * shmem_stride, B,
                TILE_DIM, SEGMENT_K_DIM, segment_offset_b, K, SKEW_HALF, num_warps, warp_id, lane_id);

            __syncthreads();

            // Compute and accumulate a warp tile.
#pragma unroll
            for (int frag_tile_offset = 0; frag_tile_offset < SEGMENT_K_DIM; frag_tile_offset += WMMA_K) {
                wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, DT, wmma::row_major> warp_tile_a[WARP_TILE_ROWS_MAX];
                wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, DT, wmma::col_major> warp_tile_b[WARP_TILE_COLS_MAX];
#pragma unroll
                for (int i = 0; i < warp_tile_rows; ++i) {
                    const int shmem_idx_a = (warp_id / 2) * warp_tile_rows * WMMA_M + i * WMMA_M;
                    wmma::load_matrix_sync(warp_tile_a[i], &shmem[shmem_idx_a][frag_tile_offset], shmem_stride);
#pragma unroll
                    for (int j = 0; j < warp_tile_cols; ++j) {
                        if (i == 0) {
                            const int shmem_idx_b = (warp_id % 2) * warp_tile_cols * WMMA_N + j * WMMA_N + TILE_DIM;
                            wmma::load_matrix_sync(warp_tile_b[j], &shmem[shmem_idx_b][frag_tile_offset], shmem_stride);
                        }
                        wmma::mma_sync(warp_tile_c[i][j], warp_tile_a[i], warp_tile_b[j], warp_tile_c[i][j]);
                    }
                }
            }

            __syncthreads();
        }

#pragma unroll
        for (int i = 0; i < warp_tile_rows; ++i) {
#pragma unroll
            for (int j = 0; j < warp_tile_cols; ++j) {
#pragma unroll
                for (int k = 0; k < warp_tile_c[i][j].num_elements; ++k) {
                    warp_tile_c[i][j].x[k] *= alpha;
                }
                const int shmem_idx = shmem_offset_warp + i * WMMA_M * TILE_DIM + j * WMMA_N;
                wmma::store_matrix_sync(reinterpret_cast<DT_ACC *>(shmem) + shmem_idx, warp_tile_c[i][j],
                    TILE_DIM, wmma::mem_row_major);
            }
        }

        __syncthreads();

        // Store the tile, consisting of warp tiles, from shared memory to D.
        store_tile<DT_ACC>(D, reinterpret_cast<const DT_ACC *>(shmem),
            TILE_DIM, TILE_DIM, tile_offset_cd, N, 0, num_warps, warp_id, lane_id);

        __syncthreads();
    }
}

template<int TILE_DIM, int SEGMENT_K_DIM, typename DT, typename DT_ACC>
void gemm_tensor_core_0(
    const DT *A,
    const DT *B, // Transpose.
    const DT_ACC *C,
    DT_ACC *D,
    DT_ACC alpha,
    DT_ACC beta,
    int M,
    int N,
    int K,
    int num_warps,
    int num_sms) {
    assert(TILE_DIM == 64 || TILE_DIM == 128);
    assert(!(SEGMENT_K_DIM % (TILE_DIM / 2)));
    assert(!(M % TILE_DIM || N % TILE_DIM));
    assert(!(K % SEGMENT_K_DIM));
    assert(num_warps == 4 || num_warps == 8);

    //Ampere: 164 * 1024, Ada: 100 * 1024.
    const int SHMEM_REQ =
        (TILE_DIM * TILE_DIM * sizeof(DT_ACC)) > (2 * TILE_DIM * (SEGMENT_K_DIM + SKEW_HALF) * sizeof(DT)) ?
        (TILE_DIM * TILE_DIM * sizeof(DT_ACC)) : (2 * TILE_DIM * (SEGMENT_K_DIM + SKEW_HALF) * sizeof(DT));
    dim3 gridDim(num_sms, 1, 1);
    dim3 blockDim(num_warps * warpSize, 1, 1);
    gemm_tensor_core_0_kernel<TILE_DIM, SEGMENT_K_DIM, DT, DT_ACC><<<gridDim, blockDim, SHMEM_REQ>>>(
        A,
        B,
        C,
        D,
        alpha,
        beta,
        M,
        N,
        K,
        num_warps);
}

int main(void) {

    const int devId = 0;
    cudaDeviceProp prop;
    checkCuda(cudaGetDeviceProperties(&prop, devId));
    printf("\nDevice: %s\n", prop.name);
    printf("\nsharedMemPerBlock: %lu\n", prop.sharedMemPerBlock);
    printf("\nsharedMemPerMultiprocessor: %lu\n", prop.sharedMemPerMultiprocessor);
    printf("\nRequired sharedMemPerBlock: %lu\n",
        (128 * 128 * sizeof(float)) > (2 * 128 * (64 + SKEW_HALF) * sizeof(half)) ?
        (128 * 128 * sizeof(float)) : (2 * 128 * (64 + SKEW_HALF) * sizeof(half)));
    checkCuda(cudaSetDevice(devId));

    // Correctness tests.

    int M = 1024;
    int N = 1024;
    int K = 1024;

    const int num_reps_corr = 10;
    const int mem_size_corr_ab = M * N * sizeof(half);
    const int mem_size_corr_cd = M * N * sizeof(float);
    half *A, *B;
    float *C, *D;
    checkCuda(cudaMalloc(&A, mem_size_corr_ab));
    checkCuda(cudaMalloc(&B, mem_size_corr_ab));
    checkCuda(cudaMalloc(&C, mem_size_corr_cd));
    checkCuda(cudaMalloc(&D, mem_size_corr_cd));

    printf("\n\n%-30s", "gemm_tensor_core_0_kernel, <half, float>, correctness");

    bool res[4];
    for (int i = 0; i < 4; ++i) {
        res[i] = true;
    }

    matmul_test::MatmulTestGemmSquare<half, float> *mt;
    for (int num_warps = 4, res_idx = 0; num_warps <= 8; num_warps += 4, res_idx += 2) {
        mt = new matmul_test::MatmulTestGemmSquare<half, float>(M, 64, -1.0f, 1.0f);
        checkCuda(cudaMemcpy(A, mt->GetA(), mem_size_corr_ab, cudaMemcpyHostToDevice));
        checkCuda(cudaMemcpy(B, mt->GetB(), mem_size_corr_ab, cudaMemcpyHostToDevice));
        checkCuda(cudaMemcpy(C, mt->GetC(), mem_size_corr_cd, cudaMemcpyHostToDevice));
        for (int i = 0; i < num_reps_corr; ++i) {
            checkCuda(cudaMemset(D, 0, mem_size_corr_cd));
            matmul_test::MatDim dim = mt->GetMatDim();
            gemm_tensor_core_0<64, 32, half, float>(
                A, B, C, D, 1.0f, 1.0f, dim.m, dim.n, dim.k, num_warps, prop.multiProcessorCount);
            checkCuda(cudaMemcpy(mt->GetRes(), D, mem_size_corr_cd, cudaMemcpyDeviceToHost));
            res[res_idx] = res[res_idx] && mt->IsCorrect(mt->GetRes(), dim, 1.0f, 1.0f, 0.001f);
        }

        delete mt; // Assignment and copy operators currently not defined.
        mt = new matmul_test::MatmulTestGemmSquare<half, float>(M, 128, -1.0f, 1.0f);
        checkCuda(cudaMemcpy(A, mt->GetA(), mem_size_corr_ab, cudaMemcpyHostToDevice));
        checkCuda(cudaMemcpy(B, mt->GetB(), mem_size_corr_ab, cudaMemcpyHostToDevice));
        checkCuda(cudaMemcpy(C, mt->GetC(), mem_size_corr_cd, cudaMemcpyHostToDevice));
        for (int i = 0; i < num_reps_corr; ++i) {
            checkCuda(cudaMemset(D, 0, mem_size_corr_cd));
            matmul_test::MatDim dim = mt->GetMatDim();
            gemm_tensor_core_0<128, 64, half, float>(
                A, B, C, D, 1.0f, 1.0f, dim.m, dim.n, dim.k, num_warps, prop.multiProcessorCount);
            checkCuda(cudaMemcpy(mt->GetRes(), D, mem_size_corr_cd, cudaMemcpyDeviceToHost));
            res[res_idx + 1] = res[res_idx + 1] && mt->IsCorrect(mt->GetRes(), dim, 1.0f, 1.0f, 0.001f);
        }
    }

    for (int i = 0; i < 4; ++i) {
        if (res[i]) {
            printf("\npassed\n");
        } else {
            printf("\nfailed\n");
        }
    }

    checkCuda(cudaFree(A));
    checkCuda(cudaFree(B));
    checkCuda(cudaFree(C));
    checkCuda(cudaFree(D));

    // Performance tests.

    M = 16384;
    N = 16384;
    K = 16384;

    const int num_reps_perf = 3;
    const int mem_size_perf_ab = M * N * sizeof(half);
    const int mem_size_perf_cd = M * N * sizeof(float);
    checkCuda(cudaMalloc(&A, mem_size_perf_ab));
    checkCuda(cudaMalloc(&B, mem_size_perf_ab));
    checkCuda(cudaMalloc(&C, mem_size_perf_cd));
    checkCuda(cudaMalloc(&D, mem_size_perf_cd));

    printf("\n\n%-30s", "gemm_tensor_core_0_kernel, <half, float>, [TFLOPS]");

    float ms[4];
    cudaEvent_t start[4];
    cudaEvent_t stop[4];
    for (int i = 0; i < 4; ++i) {
        ms[i] = 0.0f;
        checkCuda(cudaEventCreate(&start[i]));
        checkCuda(cudaEventCreate(&stop[i]));
    }
 
    for (int num_warps = 4, event_idx = 0; num_warps <= 8; num_warps += 4, event_idx += 2) {
        mt = new matmul_test::MatmulTestGemmSquare<half, float>(M, 64, -1.0f, 1.0f);
        checkCuda(cudaMemcpy(A, mt->GetA(), mem_size_perf_ab, cudaMemcpyHostToDevice));
        checkCuda(cudaMemcpy(B, mt->GetB(), mem_size_perf_ab, cudaMemcpyHostToDevice));
        checkCuda(cudaMemcpy(C, mt->GetC(), mem_size_perf_cd, cudaMemcpyHostToDevice));
        gemm_tensor_core_0<64, 32, half, float>(
            A, B, C, D, 1.0f, 1.0f, M, N, K, num_warps, prop.multiProcessorCount);
        checkCuda(cudaEventRecord(start[event_idx], 0));
        for (int i = 0; i < num_reps_perf; ++i) {
            gemm_tensor_core_0<64, 32, half, float>(
                A, B, C, D, 1.0f, 1.0f, M, N, K, num_warps, prop.multiProcessorCount);
        }
        checkCuda(cudaEventRecord(stop[event_idx], 0));
        checkCuda(cudaEventSynchronize(stop[event_idx]));
        checkCuda(cudaEventElapsedTime(&ms[event_idx], start[event_idx], stop[event_idx]));

        delete mt;
        mt = new matmul_test::MatmulTestGemmSquare<half, float>(M, 128, -1.0f, 1.0f);
        checkCuda(cudaMemcpy(A, mt->GetA(), mem_size_perf_ab, cudaMemcpyHostToDevice));
        checkCuda(cudaMemcpy(B, mt->GetB(), mem_size_perf_ab, cudaMemcpyHostToDevice));
        checkCuda(cudaMemcpy(C, mt->GetC(), mem_size_perf_cd, cudaMemcpyHostToDevice));
        gemm_tensor_core_0<128, 64, half, float>(
            A, B, C, D, 1.0f, 1.0f, M, N, K, num_warps, prop.multiProcessorCount);
        checkCuda(cudaEventRecord(start[event_idx + 1], 0));
        for (int i = 0; i < num_reps_perf; ++i) {
            gemm_tensor_core_0<128, 64, half, float>(
                A, B, C, D, 1.0f, 1.0f, M, N, K, num_warps, prop.multiProcessorCount);
        }
        checkCuda(cudaEventRecord(stop[event_idx + 1], 0));
        checkCuda(cudaEventSynchronize(stop[event_idx + 1]));
        checkCuda(cudaEventElapsedTime(&ms[event_idx + 1], start[event_idx + 1], stop[event_idx + 1]));
    }

    for (int i = 0; i < 4; ++i) {
        printf("\n%.2f\n", 2 * M * N * 1e-9 * K * num_reps_perf / ms[i]);
        checkCuda(cudaEventDestroy(start[i]));
        checkCuda(cudaEventDestroy(stop[i]));
    }

    checkCuda(cudaFree(A));
    checkCuda(cudaFree(B));
    checkCuda(cudaFree(C));
    checkCuda(cudaFree(D));

    return 0;
}
