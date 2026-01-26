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
// Tested: A100, L4.

#include <assert.h>
#include <stdio.h>

#include <cooperative_groups.h>
#include <cuda.h>
#include <cuda/pipeline>
#include <mma.h>

#include "matmul_test.h"

#pragma nv_diag_suppress static_var_with_dynamic_init
using namespace nvcuda;

const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;

const int WARP_SIZE = 32;
const int WARP_TILE_ROWS_MAX = 4;
const int WARP_TILE_COLS_MAX = 4;

const int NUM_STAGES_MAX = 4;
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
void load_tile_async(
    auto& pipeline,
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
    for (int shmem_offset = 0; shmem_offset < rows * cols; shmem_offset += num_warps * WARP_SIZE * len_vec) {
        const int temp = shmem_offset + (warp_id * WARP_SIZE + lane_id) * len_vec;
        const int shmem_idx = temp + (temp / cols) * shmem_skew;
        const int idx = offset_a + (temp / cols) * stride_am + temp % cols;
        cuda::memcpy_async(reinterpret_cast<int4 *>(shmem_ptr + shmem_idx), reinterpret_cast<const int4 *>(a_ptr + idx),
            cuda::aligned_size_t<alignof(int4)>(sizeof(int4)), pipeline);
    }
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
    for (int shmem_offset = 0; shmem_offset < rows * cols; shmem_offset += num_warps * WARP_SIZE * len_vec) {
        const int temp = shmem_offset + (warp_id * WARP_SIZE + lane_id) * len_vec;
        const int shmem_idx = temp + (temp / cols) * shmem_skew;
        const int idx = offset_a + (temp / cols) * stride_am + temp % cols;
        *reinterpret_cast<int4 *>(shmem_ptr + shmem_idx) = *reinterpret_cast<const int4 *>(a_ptr + idx);
    }
}

template<typename DT>
__device__ inline
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
    for (int shmem_offset = 0; shmem_offset < rows * cols; shmem_offset += num_warps * WARP_SIZE * len_vec) {
        const int temp = shmem_offset + (warp_id * WARP_SIZE + lane_id) * len_vec;
        const int shmem_idx = temp + (temp / cols) * shmem_skew;
        const int idx = offset_a + (temp / cols) * stride_am + temp % cols;
        *reinterpret_cast<int4 *>(a_ptr + idx) = *reinterpret_cast<const int4 *>(shmem_ptr + shmem_idx);
    }
}

// Computes GEMM (D = alpha * A * B + beta * C) with Tensor Cores and the following arguments:
// DT: half,
// DT_ACC: half or float,
// M, N, K: multiples of tile_dim,
// tile_dim: 64 or 128,
// segment_k_dim: (tile_dim / 2) increments,
// num_stages,
// num_producers: 1, 2, 4, or 8.
// num_consumers: 4 or 8.
// The requested shared memory must be at least the greater of (tile_dim * tile_dim * sizeof(DT_ACC))
// and (2 * num_stages * tile_dim * (segment_k_dim + SKEW_HALF) * sizeof(DT)).
template<typename DT, typename DT_ACC>
__global__
void gemm_tensor_core_async_0_kernel(
    const DT *A,
    const DT *B, // Transpose.
    const DT_ACC *C,
    DT_ACC *D,
    DT_ACC alpha,
    DT_ACC beta,
    int M,
    int N,
    int K,
    int tile_dim,
    int segment_k_dim,
    int num_stages,
    int num_producers,
    int num_consumers) {
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;

    // The dimensions of a warp tile computed by a warp are as follows:
    // 4 warps, tile_dim 64 -> 2x2 in terms of fragment tiles,
    // 8 warps, tile_dim 64 -> 1x2 in terms of fragment tiles,
    // 4 warps, tile_dim 128 -> 4x4 in terms of fragment tiles,
    // 8 warps, tile_dim 128 -> 2x4 in terms of fragment tiles.
    const int warp_tile_cols = 4 >> (tile_dim == 64);
    const int warp_tile_rows = warp_tile_cols >> (num_consumers == 8);

    extern __shared__ char shmem_[];
    DT* shmem = reinterpret_cast<DT *>(&shmem_[0]);

    auto thread_block = cooperative_groups::this_thread_block();
    auto thread_role = (warp_id < num_consumers) ? cuda::pipeline_role::consumer : cuda::pipeline_role::producer;
    __shared__ cuda::pipeline_shared_state<cuda::thread_scope_block, NUM_STAGES_MAX> shared_state;
    auto pipeline = cuda::make_pipeline(thread_block, &shared_state, thread_role);

    // Assign the computation of C tiles to SMs.
    for (int num_tiles_prev = 0;; num_tiles_prev += gridDim.x) {
        if ((num_tiles_prev + blockIdx.x) * (tile_dim * tile_dim) >= (M * N)) break;

        const int tile_offset_cd =
            ((num_tiles_prev + blockIdx.x) * tile_dim) / N * tile_dim * N  +
            ((num_tiles_prev + blockIdx.x) * tile_dim) % N;

        // Only used by consumers, but need to be declared/defined here.
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, DT_ACC> warp_tile_c[WARP_TILE_ROWS_MAX][WARP_TILE_COLS_MAX];
        const int shmem_offset_warp =
            (warp_id / 2) * (tile_dim * warp_tile_rows * WMMA_M) + (warp_id % 2) * (warp_tile_cols * WMMA_N);

        // Load fragments of C by consumers (4 or 8 warps).
        if (warp_id < num_consumers) {
            load_tile<DT_ACC>(reinterpret_cast<DT_ACC *>(&shmem[0]), C,
                tile_dim, tile_dim, tile_offset_cd, N, 0, num_consumers, warp_id, lane_id);

            __syncthreads();

#pragma unroll
            for (int i = 0; i < warp_tile_rows; ++i) {
#pragma unroll
                for (int j = 0; j < warp_tile_cols; ++j) {
                    const int shmem_idx = shmem_offset_warp + i * WMMA_M * tile_dim + j * WMMA_N;
                    wmma::load_matrix_sync(warp_tile_c[i][j], reinterpret_cast<const DT_ACC *>(&shmem[0]) + shmem_idx,
                        tile_dim, wmma::mem_row_major);
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
        }

       __syncthreads();

        const int shmem_stride = segment_k_dim + SKEW_HALF;
#pragma unroll
        for (int segment_offset = 0, stage_count = 0; segment_offset < K; segment_offset += segment_k_dim, ++stage_count) {
            const int segment_offset_a = (tile_offset_cd / N) * K + segment_offset;
            const int segment_offset_b = (tile_offset_cd % N) * K + segment_offset;
            const int shmem_stage_offset_a = stage_count % num_stages * tile_dim * shmem_stride;
            const int shmem_stage_offset_b = shmem_stage_offset_a + num_stages * tile_dim * shmem_stride;

            if (warp_id >= num_consumers) {

                // Copy the tiles of A and B tiles into the shared memory, iterating in the K dimension.
                pipeline.producer_acquire();

                load_tile_async<DT>(pipeline, &shmem[shmem_stage_offset_a], A, tile_dim, segment_k_dim, segment_offset_a, K,
                    SKEW_HALF, num_producers, warp_id - num_consumers, lane_id);
                load_tile_async<DT>(pipeline, &shmem[shmem_stage_offset_b], B, tile_dim, segment_k_dim, segment_offset_b, K,
                    SKEW_HALF, num_producers, warp_id - num_consumers, lane_id);

                pipeline.producer_commit();

            } else {

                // Compute and accumulate a warp tile.
                pipeline.consumer_wait();

#pragma unroll
                for (int frag_tile_offset = 0; frag_tile_offset < segment_k_dim; frag_tile_offset += WMMA_K) {
                    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, DT, wmma::row_major>
                    warp_tile_a[WARP_TILE_ROWS_MAX];
                    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, DT, wmma::col_major>
                    warp_tile_b[WARP_TILE_COLS_MAX];
#pragma unroll
                    for (int i = 0; i < warp_tile_rows; ++i) {
                        const int shmem_row_a = (warp_id / 2) * warp_tile_rows * WMMA_M + i * WMMA_M;
                        wmma::load_matrix_sync(warp_tile_a[i],
                            &shmem[shmem_stage_offset_a + shmem_row_a * shmem_stride + frag_tile_offset], shmem_stride);
#pragma unroll
                        for (int j = 0; j < warp_tile_cols; ++j) {
                            if (i == 0) {
                                const int shmem_row_b = (warp_id % 2) * warp_tile_cols * WMMA_N + j * WMMA_N + tile_dim;
                                wmma::load_matrix_sync(warp_tile_b[j],
                                    &shmem[shmem_stage_offset_b + shmem_row_b * shmem_stride + frag_tile_offset], shmem_stride);
                            }
                            wmma::mma_sync(warp_tile_c[i][j], warp_tile_a[i], warp_tile_b[j], warp_tile_c[i][j]);
                        }
                    }
                }

                pipeline.consumer_release();
            }
        }

        __syncthreads();

        // Store the tile, consisting of warp tiles, by consumers.
        if (warp_id < num_consumers) {
#pragma unroll
            for (int i = 0; i < warp_tile_rows; ++i) {
#pragma unroll
                for (int j = 0; j < warp_tile_cols; ++j) {
#pragma unroll
                    for (int k = 0; k < warp_tile_c[i][j].num_elements; ++k) {
                        warp_tile_c[i][j].x[k] *= alpha;
                    }
                    const int shmem_idx = shmem_offset_warp + i * WMMA_M * tile_dim + j * WMMA_N;
                    wmma::store_matrix_sync(reinterpret_cast<DT_ACC *>(&shmem[0]) + shmem_idx, warp_tile_c[i][j],
                        tile_dim, wmma::mem_row_major);
                }
            }

            __syncthreads();

            store_tile<DT_ACC>(D, reinterpret_cast<const DT_ACC *>(&shmem[0]),
                tile_dim, tile_dim, tile_offset_cd, N, 0, num_consumers, warp_id, lane_id);

            __syncthreads();
        }
    }
}

template
__global__
void gemm_tensor_core_async_0_kernel<half, float>(
    const half *A,
    const half *B,
    const float *C,
    float *D,
    float alpha,
    float beta,
    int M,
    int N,
    int K,
    int tile_dim,
    int segment_k_dim,
    int num_stages,
    int num_producers,
    int num_consumers);
template
__global__
void gemm_tensor_core_async_0_kernel<half, half>(
    const half *A,
    const half *B,
    const half *C,
    half *D,
    half alpha,
    half beta,
    int M,
    int N,
    int K,
    int tile_dim,
    int segment_k_dim,
    int num_stages,
    int num_producers,
    int num_consumers);

template<typename DT, typename DT_ACC>
int get_shmem_req(int tile_dim, int segment_k_dim, int num_stages) {
    const int req_ab = 2 * num_stages * tile_dim * (segment_k_dim + SKEW_HALF) * sizeof(DT);
    const int req_cd = tile_dim * tile_dim * sizeof(DT_ACC);
    return req_cd > req_ab ? req_cd : req_ab;
}

template<typename DT, typename DT_ACC>
void gemm_tensor_core_async_0(
    const DT *A,
    const DT *B, // Transpose.
    const DT_ACC *C,
    DT_ACC *D,
    DT_ACC alpha,
    DT_ACC beta,
    int M,
    int N,
    int K,
    int tile_dim,
    int segment_k_dim,
    int num_stages,
    int num_producers,
    int num_consumers,
    int num_sms) {
    assert(tile_dim == 64 || tile_dim == 128);
    assert(!(segment_k_dim % (tile_dim / 2)));
    assert(!(M % tile_dim || N % tile_dim));
    assert(!(K % segment_k_dim));
    assert(num_producers == 1 || num_producers == 2 || num_producers == 4 || num_producers == 8);
    assert(num_consumers == 4 || num_consumers == 8);

    //Ampere: 164 * 1024, Ada: 100 * 1024.
    const int SHMEM_REQ = get_shmem_req<DT, DT_ACC>(tile_dim, segment_k_dim, num_stages);
    dim3 gridDim(num_sms, 1, 1);
    dim3 blockDim((num_producers + num_consumers) * WARP_SIZE, 1, 1);
    if (SHMEM_REQ > 48 * 1024) {
        checkCuda(cudaFuncSetAttribute(gemm_tensor_core_async_0_kernel<DT, DT_ACC>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, SHMEM_REQ));
    }
    gemm_tensor_core_async_0_kernel<DT, DT_ACC><<<gridDim, blockDim, SHMEM_REQ>>>(
        A,
        B,
        C,
        D,
        alpha,
        beta,
        M,
        N,
        K,
        tile_dim,
        segment_k_dim,
        num_stages,
        num_producers,
        num_consumers);
}

template<typename DT, typename DT_ACC>
void RunCorrectnessTestSquare(
    const cudaDeviceProp *prop,
    DT_ACC alpha,
    DT_ACC beta,
    int M,
    int N,
    int K,
    int tile_dim_start, // 64 or 128.
    int tile_dim_end, // 64 or 128.
    int segment_k_dim_start, // multiple of (tile_dim / 2).
    int segment_k_dim_end, // multiple of (tile_dim / 2).
    int num_stages_start,
    int num_stages_end,
    int num_producers_start,
    int num_producers_end,
    int num_consumers_start,
    int num_consumers_end,
    int num_reps,
    DT val_min,
    DT val_max,
    DT_ACC atol) {
    int mem_size_a = M * K * sizeof(DT);
    int mem_size_b = K * N * sizeof(DT);
    int mem_size_c = M * N * sizeof(DT_ACC);
    int mem_size_d = M * N * sizeof(DT_ACC);

    DT *A, *B;
    DT_ACC *C, *D;

    checkCuda(cudaMalloc(&A, mem_size_a));
    checkCuda(cudaMalloc(&B, mem_size_b));
    checkCuda(cudaMalloc(&C, mem_size_c));
    checkCuda(cudaMalloc(&D, mem_size_d));

    for (int tile_dim = tile_dim_start; tile_dim <= tile_dim_end; tile_dim *= 2) {
        matmul_test::MatmulTestGemmSquare<DT, DT_ACC> mt(M, tile_dim, val_min, val_max);
        checkCuda(cudaMemcpy(A, mt.GetA(), mem_size_a, cudaMemcpyHostToDevice));
        checkCuda(cudaMemcpy(B, mt.GetB(), mem_size_b, cudaMemcpyHostToDevice));
        checkCuda(cudaMemcpy(C, mt.GetC(), mem_size_c, cudaMemcpyHostToDevice));
        for (int segment_k_dim = segment_k_dim_start; segment_k_dim <= segment_k_dim_end; segment_k_dim *= 2) {
            for (int num_stages = num_stages_start; num_stages <= num_stages_end; ++num_stages) {
                if (get_shmem_req<DT, DT_ACC>(tile_dim, segment_k_dim, num_stages) <= prop->sharedMemPerMultiprocessor &&
                    !(segment_k_dim % (tile_dim / 2)) &&
                    !(K % segment_k_dim)) {
                    for (int num_producers = num_producers_start; num_producers <= num_producers_end; num_producers *= 2) {
                        for (int num_consumers = num_consumers_start; num_consumers <= num_consumers_end; num_consumers *= 2) {
                            bool res = true;
                            if (segment_k_dim <= tile_dim) {
                                for (int i = 0; i < num_reps; ++i) {
                                    checkCuda(cudaMemset(D, 0, mem_size_d));
                                    matmul_test::MatDim dim = mt.GetMatDim();
                                    gemm_tensor_core_async_0<DT, DT_ACC>(A, B, C, D, alpha, beta, dim.m, dim.n, dim.k,
                                        tile_dim, segment_k_dim, num_stages, num_producers, num_consumers,
                                        prop->multiProcessorCount);
                                    checkCuda(cudaMemcpy(mt.GetRes(), D, mem_size_d, cudaMemcpyDeviceToHost));
                                    res = res && mt.IsCorrect(mt.GetRes(), dim, alpha, beta, atol);
                                }
                            } else {
                                checkCuda(cudaMemset(D, 0, mem_size_d));
                                matmul_test::MatDim dim = mt.GetMatDimMax();
                                gemm_tensor_core_async_0<DT, DT_ACC>(A, B, C, D, alpha, beta, dim.m, dim.n, dim.k,
                                    tile_dim, segment_k_dim, num_stages, num_producers, num_consumers,
                                    prop->multiProcessorCount);
                                checkCuda(cudaMemcpy(mt.GetRes(), D, mem_size_d, cudaMemcpyDeviceToHost));
                                res = mt.IsCorrect(mt.GetRes(), dim, alpha, beta, atol);
                            }
                            printf("(%d, %d, %d, %d, %d): %s\n",
                                tile_dim, segment_k_dim, num_stages, num_producers, num_consumers,
                                res ? "passed" : "failed");
                        }
                    }
                }
            }
        }
    }

    checkCuda(cudaFree(A));
    checkCuda(cudaFree(B));
    checkCuda(cudaFree(C));
    checkCuda(cudaFree(D));
}

template<typename DT, typename DT_ACC>
void RunPerformanceTestSquare(
    const cudaDeviceProp *prop,
    DT_ACC alpha,
    DT_ACC beta,
    int M,
    int N,
    int K,
    int tile_dim_start, // 64 or 128.
    int tile_dim_end, // 64 or 128.
    int segment_k_dim_start, // multiple of (tile_dim / 2).
    int segment_k_dim_end, // multiple of (tile_dim / 2).
    int num_stages_start,
    int num_stages_end,
    int num_producers_start,
    int num_producers_end,
    int num_consumers_start,
    int num_consumers_end,
    int num_reps,
    DT val_min,
    DT val_max,
    DT_ACC atol) {
    int mem_size_a = M * K * sizeof(DT);
    int mem_size_b = K * N * sizeof(DT);
    int mem_size_c = M * N * sizeof(DT_ACC);
    int mem_size_d = M * N * sizeof(DT_ACC);

    DT *A, *B;
    DT_ACC *C, *D;

    checkCuda(cudaMalloc(&A, mem_size_a));
    checkCuda(cudaMalloc(&B, mem_size_b));
    checkCuda(cudaMalloc(&C, mem_size_c));
    checkCuda(cudaMalloc(&D, mem_size_d));

    cudaEvent_t start, stop;
    checkCuda(cudaEventCreate(&start));
    checkCuda(cudaEventCreate(&stop));

    for (int tile_dim = tile_dim_start; tile_dim <= tile_dim_end; tile_dim *= 2) {
        matmul_test::MatmulTestGemmSquare<DT, DT_ACC> mt(M, tile_dim, val_min, val_max);
        checkCuda(cudaMemcpy(A, mt.GetA(), mem_size_a, cudaMemcpyHostToDevice));
        checkCuda(cudaMemcpy(B, mt.GetB(), mem_size_b, cudaMemcpyHostToDevice));
        checkCuda(cudaMemcpy(C, mt.GetC(), mem_size_c, cudaMemcpyHostToDevice));
        for (int segment_k_dim = segment_k_dim_start; segment_k_dim <= segment_k_dim_end; segment_k_dim *= 2) {
            for (int num_stages = num_stages_start; num_stages <= num_stages_end; ++num_stages) {
                if (get_shmem_req<DT, DT_ACC>(tile_dim, segment_k_dim, num_stages) <= prop->sharedMemPerMultiprocessor &&
                    !(segment_k_dim % (tile_dim / 2)) &&
                    !(K % segment_k_dim)) {
                    for (int num_producers = num_producers_start; num_producers <= num_producers_end; num_producers *= 2) {
                        for (int num_consumers = num_consumers_start; num_consumers <= num_consumers_end; num_consumers *= 2) {
                            float ms = 0.0f;
                            matmul_test::MatDim dim = mt.GetMatDimMax();
                            gemm_tensor_core_async_0<DT, DT_ACC>(A, B, C, D, alpha, beta, dim.m, dim.n, dim.k,
                                tile_dim, segment_k_dim, num_stages, num_producers, num_consumers,
                                prop->multiProcessorCount);
                            checkCuda(cudaEventRecord(start, 0));
                            for (int i = 0; i < num_reps; ++i) {
                                gemm_tensor_core_async_0<DT, DT_ACC>(A, B, C, D, alpha, beta, dim.m, dim.n, dim.k,
                                    tile_dim, segment_k_dim, num_stages, num_producers, num_consumers,
                                    prop->multiProcessorCount);
                            }
                            checkCuda(cudaEventRecord(stop, 0));
                            checkCuda(cudaEventSynchronize(stop));
                            checkCuda(cudaEventElapsedTime(&ms, start, stop));
                            printf("(%d, %d, %d, %d, %d):%.2f\n",
                                tile_dim, segment_k_dim, num_stages, num_producers, num_consumers,
                                2 * M * N * 1e-9 * K * num_reps / ms);
                        }
                    }
                }
            }
        }
    }

    checkCuda(cudaFree(A));
    checkCuda(cudaFree(B));
    checkCuda(cudaFree(C));
    checkCuda(cudaFree(D));

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

    printf("\n\n%-30s", "gemm_tensor_core_async_0_kernel, <half, half>, correctness:\n");
    RunCorrectnessTestSquare<half, half>(
        &prop, 1.0f, 1.0f, M, N, K, 64, 128, 32, 512, 4, 4, 1, 8, 4, 8, 2, -1.0f, 1.0f, 0.2f);

    printf("\n\n%-30s", "gemm_tensor_core_async_0_kernel, <half, float>, correctness:\n");
    RunCorrectnessTestSquare<half, float>(
        &prop, 1.0f, 1.0f, M, N, K, 64, 128, 32, 512, 4, 4, 1, 8, 4, 8, 2, -1.0f, 1.0f, 0.1f);

    // Performance tests.

    M = 16384;
    N = 16384;
    K = 16384;

    printf("\n\n%-30s", "gemm_tensor_core_async_0_kernel, <half, half>, [TFLOPS]:\n");
    RunPerformanceTestSquare<half, half>(
        &prop, 1.0f, 1.0f, M, N, K, 64, 128, 32, 512, 4, 4, 1, 8, 4, 8, 2, -1.0f, 1.0f, 0.2f);

    printf("\n\n%-30s", "gemm_tensor_core_async_0_kernel, <half, float>, [TFLOPS]:\n");
    RunPerformanceTestSquare<half, float>(
        &prop, 1.0f, 1.0f, M, N, K, 64, 128, 32, 512, 4, 4, 1, 8, 4, 8, 2, -1.0f, 1.0f, 0.1f);

    return 0;
}
