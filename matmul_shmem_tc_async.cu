// GEMM (D = alpha * A * B + beta * C) using shared memory and Tensor Cores, as well as various methods
// to overlap computation with data movement.
//
// The kernel enables changing the following parameter values:
//     -  the square tile dimension from 64 to 128,
//     -  the segment of the K dimension loaded into the shared memory in (tile dimension / 2) increments,
//     -  the number of pipeline stages in [1, 16],
//     -  the number of producer warps in [1, 8], and
//     -  the number of the consumer warps from 4 to 8,
// while providing vectorized and coalesced (to the extent possible) accesses for data movement to/from
// shared memory.
//
// This design enables tuning the Tensor Cores workload that each warp executes within the available
// number of registers. Doubling the tile dimension increases the array of accumulators that a warp
// computes by a factor of 4, and doubling the number of warps decreases the array of accumulators
// that a warp computes by a factor of 2, thereby enabling to gradually tune the Tensor Cores workload
// that each warp executes.
//
// The input matrices can be of float and half types.
//
// Tested: A100, L4.

#include <assert.h>
#include <stdio.h>

#include <cuda.h>
#include <cuda/barrier>
#include <mma.h>

#include "matmul_test.h"

#pragma nv_diag_suppress static_var_with_dynamic_init
using namespace nvcuda;
using CudaBarrier = cuda::barrier<cuda::thread_scope_block>;

const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;

const int NUM_STAGES_MAX = 16;
const int SHMEM_ALIGNMENT = 32;
const int SKEW_HALF = 16;
const int WARP_SIZE = 32;

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
    CudaBarrier& bar,
    DT * __restrict__ shmem_ptr,
    const DT * __restrict__ a_ptr,
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
            cuda::aligned_size_t<alignof(int4)>(sizeof(int4)), bar);
    }
}

template<typename DT>
__device__
void store_tile(
    DT * __restrict__ a_ptr,
    const DT * __restrict__ shmem_ptr,
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

template<typename DT>
__device__
void produce(
    CudaBarrier *empty,
    CudaBarrier *full,
    DT *shmem_ptr,
    const DT *a_ptr,
    const DT *b_ptr,
    int K,
    int rows,
    int cols,
    int offset_a,
    int offset_b,
    int shmem_stride,
    int shmem_skew,
    int num_stages,
    int num_warps,
    int warp_id,
    int lane_id){
    for (int segment_count = 0; segment_count < K / cols; ++segment_count) {

        empty[segment_count % num_stages].arrive_and_wait();

        const int segment_offset_a = offset_a + segment_count * cols;
        const int segment_offset_b = offset_b + segment_count * cols;
        const int shmem_stage_offset_a = (segment_count % num_stages) * rows * shmem_stride;
        const int shmem_stage_offset_b =
            (segment_count % num_stages) * rows * shmem_stride + num_stages * rows * shmem_stride;
        load_tile_async<DT>(full[segment_count % num_stages], shmem_ptr + shmem_stage_offset_a, a_ptr,
            rows, cols, segment_offset_a, K, shmem_skew, num_warps, warp_id, lane_id);
        load_tile_async<DT>(full[segment_count % num_stages], shmem_ptr + shmem_stage_offset_b, b_ptr,
            rows, cols, segment_offset_b, K, shmem_skew, num_warps, warp_id, lane_id);

        auto token = full[segment_count % num_stages].arrive();
    }

    // If a producer thread arrived here, then all threads already arrived at the last instance of
    // empty barrier and the empty barriers can be reset for the next tile.
    for (int i = 0; i < num_stages; ++i) {
        auto token = empty[i].arrive();
    }
}

template<typename DT, typename DT_ACC, int NUM_ACC>
__device__
void consume(
    CudaBarrier *empty,
    CudaBarrier *full,
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, DT_ACC> warp_tile_c[NUM_ACC],
    const DT *shmem_ptr,
    int K,
    int rows,
    int cols,
    int shmem_stride,
    int num_stages,
    int warp_id) {
    const int warp_tile_rows = NUM_ACC < 8 ? NUM_ACC / 2 : NUM_ACC / 4;
    const int warp_tile_cols = NUM_ACC / warp_tile_rows;

    for (int i = 0; i < num_stages; ++i) {
        auto token = empty[i].arrive();
    }

    for (int segment_count = 0; segment_count < K / cols; ++segment_count) {

        full[segment_count % num_stages].arrive_and_wait();

        const int shmem_stage_offset_a = (segment_count % num_stages) * rows * shmem_stride;
        const int shmem_stage_offset_b = shmem_stage_offset_a + num_stages * rows * shmem_stride;
#pragma unroll
        for (int frag_tile_offset = 0; frag_tile_offset < cols; frag_tile_offset += WMMA_K) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, DT, wmma::row_major> warp_tile_a[warp_tile_rows];
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, DT, wmma::col_major> warp_tile_b[warp_tile_cols];
#pragma unroll
            for (int i = 0; i < warp_tile_rows; ++i) {
                const int shmem_row_a = (warp_id / 2) * warp_tile_rows * WMMA_M + i * WMMA_M;
                wmma::load_matrix_sync(warp_tile_a[i],
                    shmem_ptr + shmem_stage_offset_a + (shmem_row_a * shmem_stride + frag_tile_offset), shmem_stride);
#pragma unroll
                for (int j = 0; j < warp_tile_cols; ++j) {
                    if (i == 0) {
                        const int shmem_row_b = (warp_id % 2) * warp_tile_cols * WMMA_N + j * WMMA_N;
                        wmma::load_matrix_sync(warp_tile_b[j],
                            shmem_ptr + shmem_stage_offset_b + (shmem_row_b * shmem_stride + frag_tile_offset), shmem_stride);
                    }
                    wmma::mma_sync(warp_tile_c[i * warp_tile_cols + j], warp_tile_a[i], warp_tile_b[j],
                        warp_tile_c[i * warp_tile_cols + j]);
                }
            }
        }

        auto token = empty[segment_count % num_stages].arrive();
    }
}

// Computes GEMM (D = alpha * A * B + beta * C) with Tensor Cores. Enables tuning the matrix multiply
// and accumulate workload of a warp, and changing the number of pipeline stages, the number of
// producer warps, and the number of consumer warps, according to the following arguments:
// DT: half,
// DT_ACC: half or float,
// NUM_ACC: 2, 4, 8, or 16 according to get_num_acc,
// M, N, K: multiples of tile_dim,
// tile_dim: 64 or 128,
// segment_k_dim: (tile_dim / 2) increments,
// num_stages: [1, 16],
// num_producer_warps: 1, 2, 4, or 8,
// num_consumer_warps: 4 or 8,
// The requested shared memory requirements are defined in get_shmem_req.
template<typename DT, typename DT_ACC, int NUM_ACC>
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
    int num_producer_warps,
    int num_consumer_warps) {

    // 2 -> 1 x 2, 4 -> 2 x 2, 8 -> 2 x 4, 16 -> 4 x 4.
    const int warp_tile_rows = NUM_ACC < 8 ? NUM_ACC / 2 : NUM_ACC / 4;
    const int warp_tile_cols = NUM_ACC / warp_tile_rows;

    // __align__(32) due to Tensor Cores and memcpy_async.
    // 2 * 16 (NUM_STAGES_MAX) * sizeof(CudaBarrier) maintains alignment regardless of CudaBarrier size.
    extern __shared__ __align__(SHMEM_ALIGNMENT) char shmem_[];
    CudaBarrier *bar_consumer = reinterpret_cast<CudaBarrier *>(&shmem_[0]);
    CudaBarrier *empty = reinterpret_cast<CudaBarrier *>(&shmem_[1 * sizeof(CudaBarrier)]);
    CudaBarrier *full = reinterpret_cast<CudaBarrier *>(&shmem_[(1 + NUM_STAGES_MAX) * sizeof(CudaBarrier)]);
    const int bar_alloc_size = (1 + 2 * NUM_STAGES_MAX) * sizeof(CudaBarrier);
    DT *shmem = reinterpret_cast<DT *>(&shmem_[bar_alloc_size + bar_alloc_size % SHMEM_ALIGNMENT]);

    if (threadIdx.x < 2 * NUM_STAGES_MAX) {
        init(&empty[threadIdx.x], (num_producer_warps + num_consumer_warps) * WARP_SIZE);
    }
    if (threadIdx.x == 2 * NUM_STAGES_MAX) {
        init(bar_consumer, (num_consumer_warps) * WARP_SIZE);
    }

    __syncthreads();

    // Assign the computation of C tiles to SMs.
    for (int num_tiles_prev = 0;; num_tiles_prev += gridDim.x) {
        if ((num_tiles_prev + blockIdx.x) * (tile_dim * tile_dim) >= (M * N)) break;

        const int tile_offset_cd =
            ((num_tiles_prev + blockIdx.x) * tile_dim) / N * tile_dim * N  + ((num_tiles_prev + blockIdx.x) * tile_dim) % N;

        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, DT_ACC> warp_tile_c[NUM_ACC];

        if (threadIdx.x >= num_producer_warps * WARP_SIZE) {
            const int warp_id = (threadIdx.x - num_producer_warps * WARP_SIZE) / WARP_SIZE;
            const int lane_id = threadIdx.x % WARP_SIZE;

            load_tile_async<DT_ACC>(*bar_consumer, reinterpret_cast<DT_ACC *>(shmem), C,
                tile_dim, tile_dim, tile_offset_cd, N, 0, num_consumer_warps, warp_id, lane_id);

            bar_consumer->arrive_and_wait();

            // There are two columns of warp tiles in a tile.
            const int shmem_offset_warp =
                (warp_id / 2) * (tile_dim * warp_tile_rows * WMMA_M) + (warp_id % 2) * (warp_tile_cols * WMMA_N);
#pragma unroll
            for (int i = 0; i < warp_tile_rows; ++i) {
#pragma unroll
                for (int j = 0; j < warp_tile_cols; ++j) {
                    const int shmem_idx = shmem_offset_warp + i * WMMA_M * tile_dim + j * WMMA_N;
                    wmma::load_matrix_sync(warp_tile_c[i * warp_tile_cols + j],
                        reinterpret_cast<const DT_ACC *>(shmem) + shmem_idx, tile_dim, wmma::mem_row_major);
                }
            }

            bar_consumer->arrive_and_wait();

#pragma unroll
            for (int i = 0; i < warp_tile_rows; ++i) {
#pragma unroll
                for (int j = 0; j < warp_tile_cols; ++j) {
#pragma unroll
                    for (int k = 0; k < warp_tile_c[i * warp_tile_cols + j].num_elements; ++k) {
                        warp_tile_c[i * warp_tile_cols + j].x[k] *= beta;
                    }
                }
            }
        }
        // Empty barriers block the producer threads until all consumer threads arrive.

        // Copy the tiles of A and B into the shared memory, iterating in the K dimension.
        const int shmem_stride = segment_k_dim + SKEW_HALF;
        if (threadIdx.x < num_producer_warps * WARP_SIZE) {
            const int warp_id = threadIdx.x / WARP_SIZE;
            const int lane_id = threadIdx.x % WARP_SIZE;
            produce<DT>(empty, full, shmem, A, B, K, tile_dim, segment_k_dim, (tile_offset_cd / N) * K,
                (tile_offset_cd % N) * K, shmem_stride, SKEW_HALF, num_stages, num_producer_warps, warp_id, lane_id);
        } else {
            const int warp_id = (threadIdx.x - num_producer_warps * WARP_SIZE) / WARP_SIZE;
            consume<DT, DT_ACC, NUM_ACC>(empty, full, warp_tile_c, shmem, K, tile_dim, segment_k_dim, shmem_stride,
                num_stages, warp_id);
        }

        if (threadIdx.x >= num_producer_warps * WARP_SIZE) {
            const int warp_id = (threadIdx.x - num_producer_warps * WARP_SIZE) / WARP_SIZE;
            const int lane_id = threadIdx.x % WARP_SIZE;
            const int shmem_offset_warp =
                (warp_id / 2) * (tile_dim * warp_tile_rows * WMMA_M) + (warp_id % 2) * (warp_tile_cols * WMMA_N);

            bar_consumer->arrive_and_wait();
            // All consumer threads completed matrix multiply and accumulate.

#pragma unroll
            for (int i = 0; i < warp_tile_rows; ++i) {
#pragma unroll
                for (int j = 0; j < warp_tile_cols; ++j) {
#pragma unroll
                    for (int k = 0; k < warp_tile_c[i * warp_tile_cols + j].num_elements; ++k) {
                        warp_tile_c[i * warp_tile_cols + j].x[k] *= alpha;
                    }
                    const int shmem_idx = shmem_offset_warp + i * WMMA_M * tile_dim + j * WMMA_N;
                    wmma::store_matrix_sync(reinterpret_cast<DT_ACC *>(shmem) + shmem_idx,
                        warp_tile_c[i * warp_tile_cols + j], tile_dim, wmma::mem_row_major);
                }
            }

            bar_consumer->arrive_and_wait();

            // Store the tile, consisting of warp tiles, from shared memory to D.
            store_tile<DT_ACC>(D, reinterpret_cast<const DT_ACC *>(shmem),
                tile_dim, tile_dim, tile_offset_cd, N, 0, num_consumer_warps, warp_id, lane_id);
        }

        __syncthreads();
    }
}

template
__global__
void gemm_tensor_core_async_0_kernel<half, float, 2>(
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
    int num_producer_warps,
    int num_consumer_warps);
template
__global__
void gemm_tensor_core_async_0_kernel<half, float, 4>(
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
    int num_producer_warps,
    int num_consumer_warps);
template
__global__
void gemm_tensor_core_async_0_kernel<half, float, 8>(
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
    int num_producer_warps,
    int num_consumer_warps);
template
__global__
void gemm_tensor_core_async_0_kernel<half, float, 16>(
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
    int num_producer_warps,
    int num_consumer_warps);
template
__global__
void gemm_tensor_core_async_0_kernel<half, half, 2>(
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
    int num_producer_warps,
    int num_consumer_warps);
template
__global__
void gemm_tensor_core_async_0_kernel<half, half, 4>(
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
    int num_producer_warps,
    int num_consumer_warps);
template
__global__
void gemm_tensor_core_async_0_kernel<half, half, 8>(
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
    int num_producer_warps,
    int num_consumer_warps);
template
__global__
void gemm_tensor_core_async_0_kernel<half, half, 16>(
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
    int num_producer_warps,
    int num_consumer_warps);

template<typename DT, typename DT_ACC>
int get_shmem_req(int tile_dim, int segment_k_dim, int num_stages) {
    const int bar_alloc_size = (1 + 2 * NUM_STAGES_MAX) * sizeof(CudaBarrier);
    const int bar_alloc_size_aligned = bar_alloc_size + bar_alloc_size % SHMEM_ALIGNMENT;
    const int size_ab =
        num_stages * (2 * tile_dim * (segment_k_dim + SKEW_HALF) * sizeof(DT)) + bar_alloc_size_aligned;
    const int size_cd = tile_dim * tile_dim * sizeof(DT_ACC) + bar_alloc_size_aligned;
    return size_ab > size_cd ? size_ab : size_cd;
}

int get_num_acc(int tile_dim, int num_consumer_warps) {
    return (4 >> (tile_dim == 64)) * ((4 >> (tile_dim == 64)) >> (num_consumer_warps == 8));
}

template<typename DT, typename DT_ACC>
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
    int tile_dim,
    int segment_k_dim,
    int num_stages,
    int num_producer_warps,
    int num_consumer_warps,
    int num_sms) {
    assert(tile_dim == 64 || tile_dim == 128);
    assert(!(segment_k_dim % (tile_dim / 2)));
    assert(!(M % tile_dim || N % tile_dim));
    assert(!(K % segment_k_dim));
    assert(num_stages > 0 || num_stages <= NUM_STAGES_MAX);
    assert(num_producer_warps == 1 || num_producer_warps == 2 || num_producer_warps == 4 ||num_producer_warps == 8);
    assert(num_consumer_warps == 4 || num_consumer_warps == 8);

    const int shmem_req = get_shmem_req<DT, DT_ACC>(tile_dim, segment_k_dim, num_stages);
    const int num_acc = get_num_acc(tile_dim, num_consumer_warps);
    dim3 gridDim(num_sms, 1, 1);
    dim3 blockDim((num_producer_warps + num_consumer_warps) * WARP_SIZE, 1, 1);
    if (shmem_req > 48 * 1024) {
        if (num_acc == 2) {
            checkCuda(cudaFuncSetAttribute(gemm_tensor_core_async_0_kernel<DT, DT_ACC, 2>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_req));
        }
        if (num_acc == 4) {
            checkCuda(cudaFuncSetAttribute(gemm_tensor_core_async_0_kernel<DT, DT_ACC, 4>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_req));
        }
        if (num_acc == 8) {
            checkCuda(cudaFuncSetAttribute(gemm_tensor_core_async_0_kernel<DT, DT_ACC, 8>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_req));
        }
        if (num_acc == 16) {
            checkCuda(cudaFuncSetAttribute(gemm_tensor_core_async_0_kernel<DT, DT_ACC, 16>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_req));
        }
    }
    if (num_acc == 2) {
        gemm_tensor_core_async_0_kernel<DT, DT_ACC, 2><<<gridDim, blockDim, shmem_req>>>(
            A, B, C, D, alpha, beta, M, N, K, tile_dim, segment_k_dim, num_stages, num_producer_warps, num_consumer_warps);
    }
    if (num_acc == 4) {
        gemm_tensor_core_async_0_kernel<DT, DT_ACC, 4><<<gridDim, blockDim, shmem_req>>>(
            A, B, C, D, alpha, beta, M, N, K, tile_dim, segment_k_dim, num_stages, num_producer_warps, num_consumer_warps);
    }
    if (num_acc == 8) {
        gemm_tensor_core_async_0_kernel<DT, DT_ACC, 8><<<gridDim, blockDim, shmem_req>>>(
            A, B, C, D, alpha, beta, M, N, K, tile_dim, segment_k_dim, num_stages, num_producer_warps, num_consumer_warps);
    }
    if (num_acc == 16) {
        gemm_tensor_core_async_0_kernel<DT, DT_ACC, 16><<<gridDim, blockDim, shmem_req>>>(
            A, B, C, D, alpha, beta, M, N, K, tile_dim, segment_k_dim, num_stages, num_producer_warps, num_consumer_warps);
    }
}

template<typename DT, typename DT_ACC>
void RunCorrectnessTestSquare(
    const cudaDeviceProp *prop,
    DT_ACC alpha,
    DT_ACC beta,
    int M,
    int N,
    int K,
    int tile_dim_start,
    int tile_dim_end,
    int segment_k_dim_start,
    int segment_k_dim_end,
    int num_stages_start,
    int num_stages_end,
    int num_producer_warps_start,
    int num_producer_warps_end,
    int num_consumer_warps_start,
    int num_consumer_warps_end,
    int num_reps,
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
        matmul_test::MatmulTestGemmSquare<DT, DT_ACC> mt(M, tile_dim);
        checkCuda(cudaMemcpy(A, mt.GetA(), mem_size_a, cudaMemcpyHostToDevice));
        checkCuda(cudaMemcpy(B, mt.GetB(), mem_size_b, cudaMemcpyHostToDevice));
        checkCuda(cudaMemcpy(C, mt.GetC(), mem_size_c, cudaMemcpyHostToDevice));
        for (int segment_k_dim = segment_k_dim_start; segment_k_dim <= segment_k_dim_end; segment_k_dim *= 2) {
            for (int num_stages = num_stages_start; num_stages <= num_stages_end; ++num_stages) {
                if (get_shmem_req<DT, DT_ACC>(tile_dim, segment_k_dim, num_stages) <= prop->sharedMemPerMultiprocessor &&
                    !(segment_k_dim % (tile_dim / 2)) &&
                    !(K % segment_k_dim)) {
                    for (int num_producer_warps = num_producer_warps_start; num_producer_warps <= num_producer_warps_end;
                        num_producer_warps *= 2) {
                        for (int num_consumer_warps = num_consumer_warps_start; num_consumer_warps <= num_consumer_warps_end;
                            num_consumer_warps *= 2) {
                            bool res = true;
                            if (segment_k_dim <= tile_dim) {
                                for (int i = 0; i < num_reps; ++i) {
                                    checkCuda(cudaMemset(D, 0, mem_size_d));
                                    matmul_test::MatDim dim = mt.GetMatDim();
                                    gemm_tensor_core_0<DT, DT_ACC>(A, B, C, D, alpha, beta, dim.m, dim.n, dim.k, tile_dim,
                                        segment_k_dim, num_stages, num_producer_warps, num_consumer_warps,
                                        prop->multiProcessorCount);
                                    checkCuda(cudaMemcpy(mt.GetRes(), D, mem_size_d, cudaMemcpyDeviceToHost));
                                    res = res && mt.IsCorrect(mt.GetRes(), dim, alpha, beta, atol);
                                }
                                printf("(%d, %d, %d, %d, %d): %s\n", tile_dim, segment_k_dim, num_stages, num_producer_warps,
                                    num_consumer_warps, res ? "passed" : "failed");
                            } else {
                                checkCuda(cudaMemset(D, 0, mem_size_d));
                                matmul_test::MatDim dim = mt.GetMatDimMax();
                                gemm_tensor_core_0<DT, DT_ACC>(A, B, C, D, alpha, beta, dim.m, dim.n, dim.k, tile_dim,
                                    segment_k_dim, num_stages, num_producer_warps, num_consumer_warps,
                                    prop->multiProcessorCount);
                                checkCuda(cudaMemcpy(mt.GetRes(), D, mem_size_d, cudaMemcpyDeviceToHost));
                                res = res && mt.IsCorrect(mt.GetRes(), dim, alpha, beta, atol);
                                printf("(%d, %d, %d, %d, %d): %s\n", tile_dim, segment_k_dim, num_stages, num_producer_warps,
                                    num_consumer_warps, res ? "passed" : "failed");
                            }
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
void RunAccuracyTestSquare(
    const cudaDeviceProp *prop,
    DT_ACC alpha,
    DT_ACC beta,
    int M,
    int N,
    int K,
    int tile_dim_start,
    int tile_dim_end,
    int segment_k_dim_start,
    int segment_k_dim_end,
    int num_stages_start,
    int num_stages_end,
    int num_producer_warps_start,
    int num_producer_warps_end,
    int num_consumer_warps_start,
    int num_consumer_warps_end,
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
                    for (int num_producer_warps = num_producer_warps_start; num_producer_warps <= num_producer_warps_end;
                        num_producer_warps *= 2) {
                        for (int num_consumer_warps = num_consumer_warps_start; num_consumer_warps <= num_consumer_warps_end;
                            num_consumer_warps *= 2) {
                            int count = 0;
                            if (segment_k_dim <= tile_dim) {
                                for (int i = 0; i < num_reps; ++i) {
                                    checkCuda(cudaMemset(D, 0, mem_size_d));
                                    matmul_test::MatDim dim = mt.GetMatDim();
                                    gemm_tensor_core_0<DT, DT_ACC>(A, B, C, D, alpha, beta, dim.m, dim.n, dim.k, tile_dim,
                                        segment_k_dim, num_stages, num_producer_warps, num_consumer_warps,
                                        prop->multiProcessorCount);
                                    checkCuda(cudaMemcpy(mt.GetRes(), D, mem_size_d, cudaMemcpyDeviceToHost));
                                    count += mt.GetNumIncorrect(mt.GetRes(), dim, alpha, beta, atol);
                                }
                                printf("(%d, %d, %d, %d, %d): %.2f / %d\n", tile_dim, segment_k_dim, num_stages,
                                    num_producer_warps, num_consumer_warps,  static_cast<float>(count) / num_reps, M * N);
                            } else {
                                checkCuda(cudaMemset(D, 0, mem_size_d));
                                matmul_test::MatDim dim = mt.GetMatDimMax();
                                gemm_tensor_core_0<DT, DT_ACC>(A, B, C, D, alpha, beta, dim.m, dim.n, dim.k, tile_dim,
                                    segment_k_dim, num_stages, num_producer_warps, num_consumer_warps,
                                    prop->multiProcessorCount);
                                checkCuda(cudaMemcpy(mt.GetRes(), D, mem_size_d, cudaMemcpyDeviceToHost));
                                count = mt.GetNumIncorrect(mt.GetRes(), dim, alpha, beta, atol);
                                printf("(%d, %d, %d, %d, %d): %.2f / %d\n", tile_dim, segment_k_dim, num_stages,
                                    num_producer_warps, num_consumer_warps,  static_cast<float>(count), M * N);
                            }
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
    int tile_dim_start,
    int tile_dim_end,
    int segment_k_dim_start,
    int segment_k_dim_end,
    int num_stages_start,
    int num_stages_end,
    int num_producer_warps_start,
    int num_producer_warps_end,
    int num_consumer_warps_start,
    int num_consumer_warps_end,
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
                    for (int num_producer_warps = num_producer_warps_start; num_producer_warps <= num_producer_warps_end;
                        num_producer_warps *= 2) {
                        for (int num_consumer_warps = num_consumer_warps_start; num_consumer_warps <= num_consumer_warps_end;
                            num_consumer_warps *= 2) {
                            float ms = 0.0f;
                            matmul_test::MatDim dim = mt.GetMatDimMax();
                            gemm_tensor_core_0<DT, DT_ACC>(A, B, C, D, alpha, beta, dim.m, dim.n, dim.k, tile_dim,
                                segment_k_dim, num_stages, num_producer_warps, num_consumer_warps,
                                prop->multiProcessorCount);
                            checkCuda(cudaEventRecord(start, 0));
                            for (int i = 0; i < num_reps; ++i) {
                                gemm_tensor_core_0<DT, DT_ACC>(A, B, C, D, alpha, beta, dim.m, dim.n, dim.k, tile_dim,
                                    segment_k_dim, num_stages, num_producer_warps, num_consumer_warps,
                                    prop->multiProcessorCount);
                            }
                            checkCuda(cudaEventRecord(stop, 0));
                            checkCuda(cudaEventSynchronize(stop));
                            checkCuda(cudaEventElapsedTime(&ms, start, stop));
                            printf("(%d, %d, %d, %d, %d): %.2f\n", tile_dim, segment_k_dim, num_stages, num_producer_warps,
                                num_consumer_warps,  2 * M * N * 1e-9 * K * num_reps / ms);
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
    printf("\nregsPerBlock: %d\n", prop.regsPerBlock);
    printf("\nmultiProcessorCount: %d\n", prop.multiProcessorCount);
    checkCuda(cudaSetDevice(devId));

    // Correctness tests.

    int M = 1024;
    int N = 1024;
    int K = 1024;

    printf("\n\n%-30s", "gemm_tensor_core_0_kernel, <half, half>, (1024, 1024, 1024), correctness:\n");
    RunCorrectnessTestSquare<half, half>(
        &prop, 1.0f, 1.0f, M, N, K, 64, 128, 32, 256, 1, 10, 1, 8, 4, 8, 2, 0.001f);

    printf("\n\n%-30s", "gemm_tensor_core_0_kernel, <half, float>, (1024, 1024, 1024), correctness:\n");
    RunCorrectnessTestSquare<half, float>(
        &prop, 1.0f, 1.0f, M, N, K, 64, 128, 32, 256, 1, 10, 1, 8, 4, 8, 2, 0.001f);

    // Accuracy tests.

    M = 256;
    N = 256;
    K = 256;

    printf("\n\n%-30s", "gemm_tensor_core_0_kernel, <half, half>, (256, 256, 256), accuracy:\n");
    RunAccuracyTestSquare<half, half>(
        &prop, 1.0f, 1.0f, M, N, K, 64, 128, 32, 256, 1, 10, 1, 8, 4, 8, 2, -1.0f, 1.0f, 0.1f);

    printf("\n\n%-30s", "gemm_tensor_core_0_kernel, <half, float>, (256, 256, 256), accuracy:\n");
    RunAccuracyTestSquare<half, float>(
        &prop, 1.0f, 1.0f, M, N, K, 64, 128, 32, 256, 1, 10, 1, 8, 4, 8, 2, -1.0f, 1.0f, 0.1f);

    M = 512;
    N = 512;
    K = 512;

    printf("\n\n%-30s", "gemm_tensor_core_0_kernel, <half, half>, (512, 512, 512), accuracy:\n");
    RunAccuracyTestSquare<half, half>(
        &prop, 1.0f, 1.0f, M, N, K, 64, 128, 32, 256, 1, 10, 1, 8, 4, 8, 2, -1.0f, 1.0f, 0.1f);

    printf("\n\n%-30s", "gemm_tensor_core_0_kernel, <half, float>, (512, 512, 512), accuracy:\n");
    RunAccuracyTestSquare<half, float>(
        &prop, 1.0f, 1.0f, M, N, K, 64, 128, 32, 256, 1, 10, 1, 8, 4, 8, 2, -1.0f, 1.0f, 0.1f);

    // Performance tests.

    M = 16384;
    N = 16384;
    K = 16384;

    printf("\n\n%-30s", "gemm_tensor_core_0_kernel, <half, half>, (16384, 16384, 16384), [TFLOPS]:\n");
    RunPerformanceTestSquare<half, half>(
        &prop, 1.0f, 1.0f, M, N, K, 64, 128, 32, 256, 1, 10, 1, 8, 4, 8, 2, -1.0f, 1.0f, 0.1f);

    printf("\n\n%-30s", "gemm_tensor_core_0_kernel, <half, float>, (16384, 16384, 16384), [TFLOPS]:\n");
    RunPerformanceTestSquare<half, float>(
        &prop, 1.0f, 1.0f, M, N, K, 64, 128, 32, 256, 1, 10, 1, 8, 4, 8, 2, -1.0f, 1.0f, 0.1f);

    return 0;
}
