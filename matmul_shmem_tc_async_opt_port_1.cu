// GEMM (D = alpha * A * B + beta * C) using shared memory and Tensor Cores, as well as various methods
// to overlap computation with data movement.
//
// The kernel enables changing the following parameter values:
//     -  the square tile dimension from 64 to 128,
//     -  the number of columns of the A and transposed B segments in multiples of 64,
//     -  the number of rows of the A and transposed B segments that are bounded by barriers,
//     -  the number of rows of tiles of the A matrix, iterated in a group for L2 cache efficiency,
//     -  the number of pipeline stages,
//     -  the number of producer warps from 2 to 4 to 8, and
//     -  the number of consumer warps from 4 to 8,
// while providing vectorized and coalesced (to the extent possible) accesses for data movement to/from
// shared memory.
//
// This design enables tuning the Tensor Cores workload, mapped to registers by a compiler, that
// each consumer warp executes. Doubling the tile dimension increases the array of accumulators
// that a consumer warp computes by a factor of 4, and doubling the number of consumer warps decreases
// the array of accumulators that a consumer warp computes by a factor of 2, thereby enabling to
// gradually tune the Tensor Cores workload that each consumer warp executes.
//
// The provided synchronization scheme was designed to i) decouple the consumer warps from each
// other, including at the level of accumulators, and ii) shift the start of the execution by each
// consumer warp according to the order of the load instructions for the A and transposed B segments
// of the K dimension. The order of the load instructions was from top to bottom. The earlier load
// instructions should result in an earlier start of matrix multiply and accumulate. The later load
// instructions should result in a later start of matrix multiply and accumulate. This shift should
// be preserved across the pipeline stages and enable better utilization of Tensor Cores.
//
// The highly portable implementation may be used across industry-grade GPUs and provide an informed
// selection of configurations for further optimization with features that may limit portability.
//
// The A and B matrices are of half type, and the C and D matrices can be of float or half types.
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

const int SHMEM_ALIGNMENT = 32;
const int SKEW_HALF = 16;
const int WARP_SIZE = 32;

#define WARP_TILE_ROWS(NUM_ACC) (NUM_ACC < 8 ? NUM_ACC / 2 : NUM_ACC / 4)

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
        *reinterpret_cast<int4 *>(shmem_ptr + shmem_idx) = *reinterpret_cast<const int4 *>(a_ptr + idx);
    }
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

__host__
__device__
int get_num_acc(int tile_dim, int num_consumer_warps) {
    return (4 >> (tile_dim == 64)) * ((4 >> (tile_dim == 64)) >> (num_consumer_warps == 8));
}

__host__
__device__
int get_warp_tile_rows(int tile_dim, int num_consumer_warps) {
    const int num_acc = get_num_acc(tile_dim, num_consumer_warps);
    return num_acc < 8 ? num_acc / 2 : num_acc / 4;
}

// Produces data across stages and updates the barriers as data are copied to shared memory at the
// granularity specified by the bar_step_rows_a and bar_step_rows_b parameter values.
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
    int bar_step_rows_a,
    int bar_step_rows_b,
    int shmem_stride,
    int shmem_skew,
    int num_stages,
    int num_producer_warps,
    int num_consumer_warps,
    int producer_warp_id,
    int producer_lane_id){
    const int warp_tile_rows = get_warp_tile_rows(rows, num_consumer_warps);
    const int warp_tile_cols = get_num_acc(rows, num_consumer_warps) / warp_tile_rows;

    for (int segment_count = 0; segment_count < K / cols; ++segment_count) {

        const int segment_offset_a = offset_a + segment_count * cols;
        const int segment_offset_b = offset_b + segment_count * cols;
        const int shmem_stage_offset_a = (segment_count % num_stages) * rows * shmem_stride;
        const int shmem_stage_offset_b = shmem_stage_offset_a + num_stages * rows * shmem_stride;

        // Producer warps are split between the A and B segments.
        if (producer_warp_id < num_producer_warps / 2) {
#pragma unroll
            for (int i = 0; i < rows / bar_step_rows_a; ++i) {

                if (bar_step_rows_a == rows) {
                    empty[(segment_count % num_stages)].arrive_and_wait();
                } else if (bar_step_rows_a == warp_tile_rows * WMMA_M) {
                    for (int j = 0; j < 2; ++j) { // rows / (warp_tile_cols * WMMA_N) == 2.
                        const int consumer_warp_id = i * 2 + j;
                        empty[(segment_count % num_stages) * num_consumer_warps +
                            consumer_warp_id].arrive_and_wait();
                    }
                } else if (bar_step_rows_a == WMMA_M) {
                    for (int j = 0; j < 2; ++j) {
                        const int consumer_warp_id = (i / warp_tile_rows) * 2 + j;
                        empty[(segment_count % num_stages) * num_consumer_warps * warp_tile_rows +
                            consumer_warp_id * warp_tile_rows + (i % warp_tile_rows)].arrive_and_wait();
                    }
                }

                // Use non-async load_tile for now. Each producer thread loads the same number of
                // 128-bit values for a segment of A, if the number of rows is equal to a multiple
                // of WMMA_M and the number of columns is equal to a multiple of 64.
                load_tile<DT>(shmem_ptr + shmem_stage_offset_a + i * bar_step_rows_a * shmem_stride, a_ptr,
                    bar_step_rows_a, cols, segment_offset_a + i * bar_step_rows_a * K, K, shmem_skew,
                    num_producer_warps / 2, producer_warp_id, producer_lane_id);

                if (bar_step_rows_a == rows) {
                    auto token = full[(segment_count % num_stages)].arrive();
                } else if (bar_step_rows_a == warp_tile_rows * WMMA_M) {
                    for (int j = 0; j < 2; ++j) {
                        const int consumer_warp_id = i * 2 + j;
                        auto token = full[(segment_count % num_stages) * num_consumer_warps +
                            consumer_warp_id].arrive();
                    }
                } else if (bar_step_rows_a == WMMA_M) {
                    for (int j = 0; j < 2; ++j) {
                        const int consumer_warp_id = (i / warp_tile_rows) * 2 + j;
                        auto token = full[(segment_count % num_stages) * num_consumer_warps * warp_tile_rows +
                            consumer_warp_id * warp_tile_rows + (i % warp_tile_rows)].arrive();
                    }
                }
            }
        } else {
#pragma unroll
            for (int i = 0; i < rows / bar_step_rows_b; ++i) {

                if (bar_step_rows_b == rows) {
                    const int bar_offset_b = num_stages;
                    empty[bar_offset_b + (segment_count % num_stages)].arrive_and_wait();
                } else if (bar_step_rows_b == warp_tile_cols * WMMA_N) {
                    const int bar_offset_b = num_stages * num_consumer_warps;
                    for (int j = 0; j < num_consumer_warps; j += 2) {
                        const int consumer_warp_id = i + j; // i <= 1.
                        empty[bar_offset_b + (segment_count % num_stages) * num_consumer_warps +
                            consumer_warp_id].arrive_and_wait();
                    }
                } else if (bar_step_rows_b == WMMA_N) {
                    const int bar_offset_b = num_stages * num_consumer_warps * warp_tile_rows;
                    for (int j = 0; j < num_consumer_warps; j += 2) {
                        const int consumer_warp_id = i / warp_tile_cols + j; // i / warp_tile_cols <= 1.
                        empty[bar_offset_b + (segment_count % num_stages) * num_consumer_warps * warp_tile_cols +
                            consumer_warp_id * warp_tile_cols + (i % warp_tile_cols)].arrive_and_wait();
                    }
                }

                // Use non-async load_tile for now. Each producer thread loads the same number of
                // 128-bit values for a segment of transposed B, if the number of rows is equal to
                // a multiple of WMMA_N and the number of columns is equal to a multiple of 64.
                load_tile<DT>(shmem_ptr + shmem_stage_offset_b + i * bar_step_rows_b * shmem_stride, b_ptr,
                    bar_step_rows_b, cols, segment_offset_b + i * bar_step_rows_b * K, K, shmem_skew,
                    num_producer_warps / 2, producer_warp_id - num_producer_warps / 2, producer_lane_id);

                if (bar_step_rows_b == rows) {
                    const int bar_offset_b = num_stages;
                    auto token = full[bar_offset_b + (segment_count % num_stages)].arrive();
                } else if (bar_step_rows_b == warp_tile_cols * WMMA_N) {
                    const int bar_offset_b = num_stages * num_consumer_warps;
                    for (int j = 0; j < num_consumer_warps; j += 2) {
                        const int consumer_warp_id = i + j;
                        auto token = full[bar_offset_b + (segment_count % num_stages) * num_consumer_warps +
                            consumer_warp_id].arrive();
                    }
                } else if (bar_step_rows_b == WMMA_N) {
                    const int bar_offset_b = num_stages * num_consumer_warps * warp_tile_rows;
                    for (int j = 0; j < num_consumer_warps; j += 2) {
                        const int consumer_warp_id = i / warp_tile_cols + j;
                        auto token = full[bar_offset_b + (segment_count % num_stages) * num_consumer_warps * warp_tile_cols +
                            consumer_warp_id * warp_tile_cols + (i % warp_tile_cols)].arrive();
                    }
                }
            }
        }
    }

    // If a producer thread arrived here, then all threads already arrived at the last instance of
    // empty barrier and the empty barriers can be reset for the next tile. Half of the producer
    // threads are mapped to each empty barrier of A, and the other half are mapped to each empty
    // barrier of B.
    if (producer_warp_id < num_producer_warps / 2) {
        if (bar_step_rows_a == rows) {
            for (int i = 0; i < num_stages; ++i) {
                auto token = empty[i].arrive();
            }
        } else if (bar_step_rows_a == warp_tile_rows * WMMA_M) {
            for (int i = 0; i < num_stages * num_consumer_warps; ++i) {
                auto token = empty[i].arrive();
            }
        } else if (bar_step_rows_a == WMMA_M) {
            for (int i = 0; i < num_stages * num_consumer_warps * warp_tile_rows; ++i) {
                auto token = empty[i].arrive();
            }
        }
    } else {
        if (bar_step_rows_b == rows) {
            const int bar_offset_b = num_stages;
            for (int i = 0; i < num_stages; ++i) {
                auto token = empty[bar_offset_b + i].arrive();
            }
        } else if (bar_step_rows_b == warp_tile_cols * WMMA_N) {
            const int bar_offset_b = num_stages * num_consumer_warps;
            for (int i = 0; i < num_stages * num_consumer_warps; ++i) {
                auto token = empty[bar_offset_b + i].arrive();
            }
        } else if (bar_step_rows_b == WMMA_N) {
            const int bar_offset_b = num_stages * num_consumer_warps * warp_tile_rows;
            for (int i = 0; i < num_stages * num_consumer_warps * warp_tile_cols; ++i) {
                auto token = empty[bar_offset_b + i].arrive();
            }
        }
    }
}

// Consumes data in the K dimension for an accumulator in a warp tile for earlier data consumption.
template<typename DT, typename DT_ACC>
__device__
void consume_acc(
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, DT_ACC> *frag_tile_c,
    const DT *shmem_ptr,
    int K,
    int rows,
    int cols,
    int row_a,
    int row_b,
    int shmem_stride,
    int num_stages,
    int stage_id) {
    const int shmem_stage_offset_a = stage_id * rows * shmem_stride;
    const int shmem_stage_offset_b = shmem_stage_offset_a + num_stages * rows * shmem_stride;
#pragma unroll
    for (int frag_tile_offset = 0; frag_tile_offset < cols; frag_tile_offset += WMMA_K) {
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, DT, wmma::row_major> frag_tile_a;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, DT, wmma::col_major> frag_tile_b;
        wmma::load_matrix_sync(frag_tile_a,
            shmem_ptr + shmem_stage_offset_a + (row_a * shmem_stride + frag_tile_offset), shmem_stride);
        wmma::load_matrix_sync(frag_tile_b,
            shmem_ptr + shmem_stage_offset_b + (row_b * shmem_stride + frag_tile_offset), shmem_stride);
        wmma::mma_sync(*frag_tile_c, frag_tile_a, frag_tile_b, *frag_tile_c);
    }
}

// Consumes data across stages and updates the barriers at the granularity specified by the
// bar_step_rows_a and bar_step_rows_b parameter values.
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
    int bar_step_rows_a,
    int bar_step_rows_b,
    int shmem_stride,
    int num_stages,
    int num_consumer_warps,
    int consumer_warp_id) {
    const int warp_tile_rows = get_warp_tile_rows(rows, num_consumer_warps);
    const int warp_tile_cols = get_num_acc(rows, num_consumer_warps) / warp_tile_rows;

    for (int i = 0; i < num_stages; ++i) {
        if (bar_step_rows_a == rows && bar_step_rows_b == rows) {
            const int bar_offset_b = num_stages;
            auto token_a = empty[i].arrive();
            auto token_b = empty[bar_offset_b + i].arrive();
        } else if (bar_step_rows_a == warp_tile_rows * WMMA_M && bar_step_rows_b == warp_tile_cols * WMMA_N) {
            const int bar_offset_b = num_stages * num_consumer_warps;
            auto token_a = empty[i * num_consumer_warps + consumer_warp_id].arrive();
            auto token_b = empty[bar_offset_b + i * num_consumer_warps + consumer_warp_id].arrive();
        } else if (bar_step_rows_a == WMMA_M && bar_step_rows_b == WMMA_N) {
            for (int j = 0; j < warp_tile_rows; ++j) {
                auto token_a = empty[i * num_consumer_warps * warp_tile_rows +
                    consumer_warp_id * warp_tile_rows + j].arrive();
            }
            const int bar_offset_b = num_stages * num_consumer_warps * warp_tile_rows;
            for (int j = 0; j < warp_tile_cols; ++j) {
                auto token_b = empty[bar_offset_b + i * num_consumer_warps * warp_tile_cols +
                    consumer_warp_id * warp_tile_cols + j].arrive();
            }
        }
    }

    for (int segment_count = 0; segment_count < K / cols; ++segment_count) {

        const int shmem_stage_offset_a = (segment_count % num_stages) * rows * shmem_stride;
        const int shmem_stage_offset_b = shmem_stage_offset_a + num_stages * rows * shmem_stride;

        if (bar_step_rows_a == WMMA_M && bar_step_rows_b == WMMA_N){

            // TODO: Move warp_tile_cols to the outer loop, because warp_tile_rows <= warp_tile_cols.
#pragma unroll
            for (int i = 0; i < warp_tile_rows; ++i) {
                const int shmem_row_a = (consumer_warp_id / 2) * warp_tile_rows * WMMA_M + i * WMMA_M;
#pragma unroll
                for (int j = 0; j < warp_tile_cols; ++j) {
                    const int shmem_row_b = (consumer_warp_id % 2) * warp_tile_cols * WMMA_N + j * WMMA_N;
                    const int bar_offset_b = num_stages * num_consumer_warps * warp_tile_rows;

                    full[(segment_count % num_stages) * num_consumer_warps * warp_tile_rows +
                         consumer_warp_id * warp_tile_rows + i].arrive_and_wait();
                    full[bar_offset_b + (segment_count % num_stages) * num_consumer_warps * warp_tile_cols +
                         consumer_warp_id * warp_tile_cols + j].arrive_and_wait();

                    consume_acc<DT, DT_ACC>(&warp_tile_c[i * warp_tile_cols + j], shmem_ptr, K, rows, cols,
                        shmem_row_a, shmem_row_b, shmem_stride, num_stages, (segment_count % num_stages));

                    auto token_a = empty[(segment_count % num_stages) * num_consumer_warps * warp_tile_rows +
                        consumer_warp_id * warp_tile_rows + i].arrive();
                    auto token_b = empty[bar_offset_b + (segment_count % num_stages) * num_consumer_warps * warp_tile_cols +
                        consumer_warp_id * warp_tile_cols + j].arrive();
                }
            }
        } else {

            if (bar_step_rows_a == rows && bar_step_rows_b == rows) {
                const int bar_offset_b = num_stages;
                full[segment_count % num_stages].arrive_and_wait();
                full[bar_offset_b + (segment_count % num_stages)].arrive_and_wait();
            } else if (bar_step_rows_a == warp_tile_rows * WMMA_M && bar_step_rows_b == warp_tile_cols * WMMA_N) {
                const int bar_offset_b = num_stages * num_consumer_warps;
                full[(segment_count % num_stages) * num_consumer_warps +
                    consumer_warp_id].arrive_and_wait();
                full[bar_offset_b + (segment_count % num_stages) * num_consumer_warps +
                    consumer_warp_id].arrive_and_wait();
            }

#pragma unroll
            for (int frag_tile_offset = 0; frag_tile_offset < cols; frag_tile_offset += WMMA_K) {
                wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, DT, wmma::row_major> warp_tile_a[WARP_TILE_ROWS(NUM_ACC)];
                wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, DT, wmma::col_major> warp_tile_b[NUM_ACC / WARP_TILE_ROWS(NUM_ACC)];
#pragma unroll
                for (int i = 0; i < warp_tile_rows; ++i) {
                    const int shmem_row_a = (consumer_warp_id / 2) * warp_tile_rows * WMMA_M + i * WMMA_M;
                    wmma::load_matrix_sync(warp_tile_a[i],
                        shmem_ptr + shmem_stage_offset_a + (shmem_row_a * shmem_stride + frag_tile_offset), shmem_stride);
#pragma unroll
                    for (int j = 0; j < warp_tile_cols; ++j) {
                        if (i == 0) {
                            const int shmem_row_b = (consumer_warp_id % 2) * warp_tile_cols * WMMA_N + j * WMMA_N;
                            wmma::load_matrix_sync(warp_tile_b[j],
                                shmem_ptr + shmem_stage_offset_b + (shmem_row_b * shmem_stride + frag_tile_offset),
                                shmem_stride);
                        }
                        wmma::mma_sync(warp_tile_c[i * warp_tile_cols + j], warp_tile_a[i], warp_tile_b[j],
                            warp_tile_c[i * warp_tile_cols + j]);
                    }
                }
            }

            if (bar_step_rows_a == rows && bar_step_rows_b == rows) {
                const int bar_offset_b = num_stages;
                auto token_a = empty[segment_count % num_stages].arrive();
                auto token_b = empty[bar_offset_b + (segment_count % num_stages)].arrive();
            } else if (bar_step_rows_a == warp_tile_rows * WMMA_M && bar_step_rows_b == warp_tile_cols * WMMA_N) {
                const int bar_offset_b = num_stages * num_consumer_warps;
                auto token_a = empty[(segment_count % num_stages) * num_consumer_warps +
                    consumer_warp_id].arrive();
                auto token_b = empty[bar_offset_b + (segment_count % num_stages) * num_consumer_warps +
                    consumer_warp_id].arrive();
            }
        }
    }
}

// Computes the offset to C and D tiles that is L2 cache-optimized across thread blocks.
__device__
int get_tile_offset_cd(int M, int N, int tile_dim, int tile_group_m, int tile_count) {
    const int group_tile_count = tile_group_m * N / tile_dim;
    const int group_first_row = (tile_count / group_tile_count) * tile_group_m;
    const int group_rows = min(M / tile_dim - group_first_row, tile_group_m);
    const int tile_row = group_first_row + (tile_count % group_tile_count) % group_rows;
    const int tile_col = (tile_count % group_tile_count) / group_rows;
    return tile_row * tile_dim * N + tile_col * tile_dim;
}

// Computes GEMM (D = alpha * A * B + beta * C) with Tensor Cores. Provides portability. Enables
// i) tuning the matrix multiply and accumulate workload of a consumer warp, and ii) synchronizing
// the producer and consumer warps at the level of consumer warp tiles and individual accumulators
// for early data consumption in the order of load instructions. The parameters can be set to the
// following values:
// DT: half,
// DT_ACC: half or float,
// NUM_ACC: 2, 4, 8, or 16 according to get_num_acc,
// M, N, K: multiples of tile_dim,
// tile_dim: 64 or 128,
// segment_dim_k: multiple of 64,
// bar_step_rows_a: WMMA_M, get_warp_tile_rows(...) * WMMA_M, or tile_dim,
// bar_step_rows_b: WMMA_N, NUM_ACC / get_warp_tile_rows(...) * WMMA_N, or tile_dim,
// tile_group_m: > 0,
// num_stages: [1, 16],
// num_producer_warps: 2, 4, or 8,
// num_consumer_warps: 4 or 8.
// The requested shared memory requirements are defined in get_shmem_req.
template<typename DT, typename DT_ACC, int NUM_ACC>
__global__
void gemm_shmem_tc_async_opt_port_kernel(
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
    int segment_dim_k,
    int bar_step_rows_a,
    int bar_step_rows_b,
    int tile_group_m,
    int num_stages,
    int num_producer_warps,
    int num_consumer_warps) {

    // 2 -> 1 x 2, 4 -> 2 x 2, 8 -> 2 x 4, 16 -> 4 x 4.
    const int warp_tile_rows = get_warp_tile_rows(tile_dim, num_consumer_warps);
    const int warp_tile_cols = get_num_acc(tile_dim, num_consumer_warps) / warp_tile_rows;

    // __align__(32) due to Tensor Cores and memcpy_async.
    extern __shared__ __align__(SHMEM_ALIGNMENT) char shmem_[];

    // The layout of barriers corresponds to consumer warps. The empty barriers for A across all
    // stages are followed by the empty barriers for B across all stages. The empty barriers are
    // followed by full barriers. The bar_step_rows_a and bar_step_rows_b parameter values result
    // in the following mapping:
    //     bar_step_rows_a == rows && bar_step_rows_b == rows:
    //         all consumer warps are mapped to one empty barrier per stage for A and one empty barrier
    //         per stage for B,
    //     bar_step_rows_a == warp_tile_rows * WMMA_M && bar_step_rows_b == warp_tile_cols * WMMA_N:
    //         a consumer warp is mapped to one empty barrier per stage for A and one empty barrier per
    //         stage for B,
    //     bar_step_rows_a == WMMA_M && bar_step_rows_b == WMMA_N:
    //         a consumer warp is mapped to warp_tile_rows empty barriers per stage for A and
    //         warp_tile_cols empty barriers per stage for B.
    // Half of the producer warps are mapped to each empty barrier of A, and the other half are mapped
    // to each empty barrier of B. The same mapping is applied to the full barriers.
    CudaBarrier *bar_consumer = reinterpret_cast<CudaBarrier *>(&shmem_[0]);
    CudaBarrier *empty = reinterpret_cast<CudaBarrier *>(&shmem_[1 * sizeof(CudaBarrier)]);
    int num_bars_empty_a = num_stages;
    int num_bars_empty_b = num_stages;
    if (bar_step_rows_a == warp_tile_rows * WMMA_M && bar_step_rows_b == warp_tile_cols * WMMA_N) {
        num_bars_empty_a *= num_consumer_warps;
        num_bars_empty_b *= num_consumer_warps;
    } else if (bar_step_rows_a == WMMA_M && bar_step_rows_b == WMMA_N) {
        num_bars_empty_a *= num_consumer_warps * warp_tile_rows;
        num_bars_empty_b *= num_consumer_warps * warp_tile_cols;
    }
    CudaBarrier *full =
        reinterpret_cast<CudaBarrier *>(&shmem_[(1 + num_bars_empty_a + num_bars_empty_b) * sizeof(CudaBarrier)]);
    const int bar_alloc_size = (1 + 2 * (num_bars_empty_a + num_bars_empty_b)) * sizeof(CudaBarrier);
    DT *shmem = reinterpret_cast<DT *>(&shmem_[bar_alloc_size + bar_alloc_size % SHMEM_ALIGNMENT]);

    if (threadIdx.x == 0) {
        init(bar_consumer, (num_consumer_warps) * WARP_SIZE);
    }

    // The number of consumer threads is greater than the number of barriers per stage.
    // TODO: optimize further by using more threads for initialization.
    const int num_bars_stage_a = num_bars_empty_a / num_stages;
    if (threadIdx.x >= num_producer_warps * WARP_SIZE &&
        threadIdx.x < num_producer_warps * WARP_SIZE + num_bars_stage_a) {
        int bar_thread_count_a = (num_producer_warps / 2 + 1) * WARP_SIZE;
        if (bar_step_rows_a == tile_dim) {
            bar_thread_count_a = ((num_producer_warps / 2) + num_consumer_warps) * WARP_SIZE;
        }
        const int idx = threadIdx.x % num_bars_stage_a;
        for (int i = 0; i < num_stages; ++i) {
            init(&empty[i * num_bars_stage_a + idx], bar_thread_count_a);
            init(&full[i * num_bars_stage_a + idx], bar_thread_count_a);
        }
    }
    const int num_bars_stage_b = num_bars_empty_b / num_stages;
    if (threadIdx.x >= num_producer_warps * WARP_SIZE &&
        threadIdx.x < num_producer_warps * WARP_SIZE + num_bars_stage_b) {
        int bar_thread_count_b = (num_producer_warps / 2 + 1) * WARP_SIZE;
        if (bar_step_rows_b == tile_dim) {
            bar_thread_count_b = ((num_producer_warps / 2) + num_consumer_warps) * WARP_SIZE;
        }
        const int idx = threadIdx.x % num_bars_stage_b;
        for (int i = 0; i < num_stages; ++i) {
            init(&empty[num_bars_empty_a + i * num_bars_stage_b + idx], bar_thread_count_b);
            init(&full[num_bars_empty_a + i * num_bars_stage_b + idx], bar_thread_count_b);
        }
    }

    __syncthreads();

    // Assign the computation of C tiles to SMs.
    for (int num_tiles_prev = 0;; num_tiles_prev += gridDim.x) {
        if ((num_tiles_prev + blockIdx.x) * (tile_dim * tile_dim) >= (M * N)) break;

        const int tile_offset_cd = get_tile_offset_cd(M, N, tile_dim, tile_group_m, num_tiles_prev + blockIdx.x);
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
        const int shmem_stride = segment_dim_k + SKEW_HALF;
        if (threadIdx.x < num_producer_warps * WARP_SIZE) {
            const int warp_id = threadIdx.x / WARP_SIZE;
            const int lane_id = threadIdx.x % WARP_SIZE;
            produce<DT>(empty, full, shmem, A, B, K, tile_dim, segment_dim_k, (tile_offset_cd / N) * K,
                (tile_offset_cd % N) * K, bar_step_rows_a, bar_step_rows_b, shmem_stride, SKEW_HALF, num_stages,
                num_producer_warps, num_consumer_warps, warp_id, lane_id);
        } else {
            const int warp_id = (threadIdx.x - num_producer_warps * WARP_SIZE) / WARP_SIZE;
            consume<DT, DT_ACC, NUM_ACC>(empty, full, warp_tile_c, shmem, K, tile_dim, segment_dim_k, bar_step_rows_a,
                bar_step_rows_b, shmem_stride, num_stages, num_consumer_warps, warp_id);
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
void gemm_shmem_tc_async_opt_port_kernel<half, float, 2>(
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
    int segment_dim_k,
    int bar_step_rows_a,
    int bar_step_rows_b,
    int tile_group_m,
    int num_stages,
    int num_producer_warps,
    int num_consumer_warps);
template
__global__
void gemm_shmem_tc_async_opt_port_kernel<half, float, 4>(
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
    int segment_dim_k,
    int bar_step_rows_a,
    int bar_step_rows_b,
    int tile_group_m,
    int num_stages,
    int num_producer_warps,
    int num_consumer_warps);
template
__global__
void gemm_shmem_tc_async_opt_port_kernel<half, float, 8>(
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
    int segment_dim_k,
    int bar_step_rows_a,
    int bar_step_rows_b,
    int tile_group_m,
    int num_stages,
    int num_producer_warps,
    int num_consumer_warps);
template
__global__
void gemm_shmem_tc_async_opt_port_kernel<half, float, 16>(
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
    int bar_step_rows_a,
    int bar_step_rows_b,
    int tile_group_m,
    int num_stages,
    int num_producer_warps,
    int num_consumer_warps);
template
__global__
void gemm_shmem_tc_async_opt_port_kernel<half, half, 2>(
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
    int segment_dim_k,
    int bar_step_rows_a,
    int bar_step_rows_b,
    int tile_group_m,
    int num_stages,
    int num_producer_warps,
    int num_consumer_warps);
template
__global__
void gemm_shmem_tc_async_opt_port_kernel<half, half, 4>(
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
    int segment_dim_k,
    int bar_step_rows_a,
    int bar_step_rows_b,
    int tile_group_m,
    int num_stages,
    int num_producer_warps,
    int num_consumer_warps);
template
__global__
void gemm_shmem_tc_async_opt_port_kernel<half, half, 8>(
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
    int segment_dim_k,
    int bar_step_rows_a,
    int bar_step_rows_b,
    int tile_group_m,
    int num_stages,
    int num_producer_warps,
    int num_consumer_warps);
template
__global__
void gemm_shmem_tc_async_opt_port_kernel<half, half, 16>(
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
    int segment_dim_k,
    int bar_step_rows_a,
    int bar_step_rows_b,
    int tile_group_m,
    int num_stages,
    int num_producer_warps,
    int num_consumer_warps);

template<typename DT, typename DT_ACC>
int get_shmem_req(
    int tile_dim,
    int segment_k_dim,
    int bar_step_rows_a,
    int bar_step_rows_b,
    int num_stages,
    int num_consumer_warps) {
    const int warp_tile_rows = get_warp_tile_rows(tile_dim, num_consumer_warps);
    const int warp_tile_cols = get_num_acc(tile_dim, num_consumer_warps) / warp_tile_rows;
    int num_bars_empty_a = num_stages;
    int num_bars_empty_b = num_stages;
    if (bar_step_rows_a == warp_tile_rows * WMMA_M && bar_step_rows_b == warp_tile_cols * WMMA_N) {
        num_bars_empty_a *= num_consumer_warps;
        num_bars_empty_b *= num_consumer_warps;
    } else if (bar_step_rows_a == WMMA_M && bar_step_rows_b == WMMA_N) {
        num_bars_empty_a *= num_consumer_warps * warp_tile_rows;
        num_bars_empty_b *= num_consumer_warps * warp_tile_cols;
    }
    const int bar_alloc_size = (1 + 2 * (num_bars_empty_a + num_bars_empty_b)) * sizeof(CudaBarrier);
    const int bar_alloc_size_aligned = bar_alloc_size + bar_alloc_size % SHMEM_ALIGNMENT;
    const int size_ab =
        num_stages * (2 * tile_dim * (segment_k_dim + SKEW_HALF) * sizeof(DT)) + bar_alloc_size_aligned;
    const int size_cd = tile_dim * tile_dim * sizeof(DT_ACC) + bar_alloc_size_aligned;
    return size_ab > size_cd ? size_ab : size_cd;
}

template<typename DT, typename DT_ACC>
void gemm_shmem_tc_async_opt_port(
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
    int segment_dim_k,
    int bar_step_rows_a,
    int bar_step_rows_b,
    int tile_group_m,
    int num_stages,
    int num_producer_warps,
    int num_consumer_warps,
    int num_sms) {
    const int warp_tile_rows = get_warp_tile_rows(tile_dim, num_consumer_warps);
    const int warp_tile_cols = get_num_acc(tile_dim, num_consumer_warps) / warp_tile_rows;
    assert(tile_dim == 64 || tile_dim == 128);
    assert((segment_dim_k >= 64) && !(segment_dim_k % 64));
    assert(!(M % tile_dim || N % tile_dim));
    assert(!(K % segment_dim_k));
    assert((bar_step_rows_a == tile_dim && bar_step_rows_b == tile_dim) ||
        (bar_step_rows_a == warp_tile_rows * WMMA_M && bar_step_rows_b == warp_tile_cols * WMMA_N) ||
        (bar_step_rows_a == WMMA_M && bar_step_rows_b == WMMA_N));
    assert(num_stages > 0);
    assert(num_producer_warps == 2 || num_producer_warps == 4 ||num_producer_warps == 8);
    assert(num_consumer_warps == 4 || num_consumer_warps == 8);

    const int shmem_req = get_shmem_req<DT, DT_ACC>(tile_dim, segment_dim_k, bar_step_rows_a, bar_step_rows_b,
        num_stages, num_consumer_warps);
    const int num_acc = get_num_acc(tile_dim, num_consumer_warps);
    dim3 gridDim(num_sms, 1, 1);
    dim3 blockDim((num_producer_warps + num_consumer_warps) * WARP_SIZE, 1, 1);
    if (shmem_req > 48 * 1024) {
        if (num_acc == 2) {
            checkCuda(cudaFuncSetAttribute(gemm_shmem_tc_async_opt_port_kernel<DT, DT_ACC, 2>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_req));
        }
        if (num_acc == 4) {
            checkCuda(cudaFuncSetAttribute(gemm_shmem_tc_async_opt_port_kernel<DT, DT_ACC, 4>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_req));
        }
        if (num_acc == 8) {
            checkCuda(cudaFuncSetAttribute(gemm_shmem_tc_async_opt_port_kernel<DT, DT_ACC, 8>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_req));
        }
        if (num_acc == 16) {
            checkCuda(cudaFuncSetAttribute(gemm_shmem_tc_async_opt_port_kernel<DT, DT_ACC, 16>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_req));
        }
    }
    if (num_acc == 2) {
        gemm_shmem_tc_async_opt_port_kernel<DT, DT_ACC, 2><<<gridDim, blockDim, shmem_req>>>(A, B, C, D, alpha, beta,
            M, N, K, tile_dim, segment_dim_k, bar_step_rows_a, bar_step_rows_b, tile_group_m, num_stages,
            num_producer_warps, num_consumer_warps);
    }
    if (num_acc == 4) {
        gemm_shmem_tc_async_opt_port_kernel<DT, DT_ACC, 4><<<gridDim, blockDim, shmem_req>>>(A, B, C, D, alpha, beta,
            M, N, K, tile_dim, segment_dim_k, bar_step_rows_a, bar_step_rows_b, tile_group_m, num_stages,
            num_producer_warps, num_consumer_warps);
    }
    if (num_acc == 8) {
        gemm_shmem_tc_async_opt_port_kernel<DT, DT_ACC, 8><<<gridDim, blockDim, shmem_req>>>(A, B, C, D, alpha, beta,
            M, N, K, tile_dim, segment_dim_k, bar_step_rows_a, bar_step_rows_b, tile_group_m, num_stages,
            num_producer_warps, num_consumer_warps);
    }
    if (num_acc == 16) {
        gemm_shmem_tc_async_opt_port_kernel<DT, DT_ACC, 16><<<gridDim, blockDim, shmem_req>>>(A, B, C, D, alpha, beta,
            M, N, K, tile_dim, segment_dim_k, bar_step_rows_a, bar_step_rows_b, tile_group_m, num_stages,
            num_producer_warps, num_consumer_warps);
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
    int segment_dim_k_start,
    int segment_dim_k_end,
    int tile_group_m_start,
    int tile_group_m_end,
    int num_stages_start,
    int num_stages_end,
    int num_producer_warps_start,
    int num_producer_warps_end,
    int num_consumer_warps_start,
    int num_consumer_warps_end,
    int num_reps,
    DT_ACC atol) {
    const int mem_size_a = M * K * sizeof(DT);
    const int mem_size_b = K * N * sizeof(DT);
    const int mem_size_c = M * N * sizeof(DT_ACC);
    const int mem_size_d = M * N * sizeof(DT_ACC);

    DT *A, *B;
    DT_ACC *C, *D;

    checkCuda(cudaMalloc(&A, mem_size_a));
    checkCuda(cudaMalloc(&B, mem_size_b));
    checkCuda(cudaMalloc(&C, mem_size_c));
    checkCuda(cudaMalloc(&D, mem_size_d));

    const int shmem_size = prop->sharedMemPerMultiprocessor;
    const int num_sms = prop->multiProcessorCount;

    for (int tile_dim = tile_dim_start; tile_dim <= tile_dim_end; tile_dim *= 2) {
        matmul_test::MatmulTestGemmSquare<DT, DT_ACC> mt(M, tile_dim);
        checkCuda(cudaMemcpy(A, mt.GetA(), mem_size_a, cudaMemcpyHostToDevice));
        checkCuda(cudaMemcpy(B, mt.GetB(), mem_size_b, cudaMemcpyHostToDevice));
        checkCuda(cudaMemcpy(C, mt.GetC(), mem_size_c, cudaMemcpyHostToDevice));
        for (int segment_dim_k = segment_dim_k_start; segment_dim_k <= segment_dim_k_end; segment_dim_k *= 2) {
            for (int tile_group_m = tile_group_m_start; tile_group_m <= tile_group_m_end; ++tile_group_m) {
                for (int num_stages = num_stages_start; num_stages <= num_stages_end; ++num_stages) {
                    for (int num_producer_warps = num_producer_warps_start; num_producer_warps <= num_producer_warps_end;
                         num_producer_warps *= 2) {
                        for (int num_consumer_warps = num_consumer_warps_start; num_consumer_warps <= num_consumer_warps_end;
                             num_consumer_warps *= 2) {
                            const int warp_tile_rows = get_warp_tile_rows(tile_dim, num_consumer_warps);
                            const int warp_tile_cols = get_num_acc(tile_dim, num_consumer_warps) / warp_tile_rows;
                            const int bar_step_rows_a[3] = {WMMA_M, warp_tile_rows * WMMA_M, tile_dim};
                            const int bar_step_rows_b[3] = {WMMA_N, warp_tile_cols * WMMA_N, tile_dim};
                            for (int i = 1; i < 3; ++i) {
                                if (get_shmem_req<DT, DT_ACC>(tile_dim, segment_dim_k, bar_step_rows_a[i],
                                        bar_step_rows_b[i], num_stages, num_consumer_warps) <= shmem_size &&
                                    !(K % segment_dim_k)) {
                                    bool res = true;
                                    if (segment_dim_k <= tile_dim) {
                                        for (int j = 0; j < num_reps; ++j) {
                                            checkCuda(cudaMemset(D, 0, mem_size_d));
                                            matmul_test::MatDim dim = mt.GetMatDim();
                                            gemm_shmem_tc_async_opt_port<DT, DT_ACC>(A, B, C, D, alpha, beta,
                                                dim.m, dim.n, dim.k, tile_dim, segment_dim_k, bar_step_rows_a[i],
                                                bar_step_rows_b[i], tile_group_m, num_stages, num_producer_warps,
                                                num_consumer_warps, num_sms);
                                            checkCuda(cudaMemcpy(mt.GetRes(), D, mem_size_d, cudaMemcpyDeviceToHost));
                                            res = res && mt.IsCorrect(mt.GetRes(), dim, alpha, beta, atol);
                                        }
                                        printf("(%d, %d, %d, %d, %d, %d, %d, %d): %s\n", tile_dim, segment_dim_k,
                                               bar_step_rows_a[i], bar_step_rows_b[i], tile_group_m, num_stages,
                                               num_producer_warps, num_consumer_warps, res ? "passed" : "failed");
                                    } else {
                                        checkCuda(cudaMemset(D, 0, mem_size_d));
                                        matmul_test::MatDim dim = mt.GetMatDimMax();
                                        gemm_shmem_tc_async_opt_port<DT, DT_ACC>(A, B, C, D, alpha, beta,
                                            dim.m, dim.n, dim.k, tile_dim, segment_dim_k, bar_step_rows_a[i],
                                            bar_step_rows_b[i], tile_group_m, num_stages, num_producer_warps,
                                            num_consumer_warps, num_sms);
                                        checkCuda(cudaMemcpy(mt.GetRes(), D, mem_size_d, cudaMemcpyDeviceToHost));
                                        res = res && mt.IsCorrect(mt.GetRes(), dim, alpha, beta, atol);
                                        printf("(%d, %d, %d, %d, %d, %d, %d, %d): %s\n", tile_dim, segment_dim_k,
                                               bar_step_rows_a[i], bar_step_rows_b[i], tile_group_m, num_stages,
                                               num_producer_warps, num_consumer_warps, res ? "passed" : "failed");
                                    }
                                }
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
    int segment_dim_k_start,
    int segment_dim_k_end,
    int tile_group_m_start,
    int tile_group_m_end,
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
    const int mem_size_a = M * K * sizeof(DT);
    const int mem_size_b = K * N * sizeof(DT);
    const int mem_size_c = M * N * sizeof(DT_ACC);
    const int mem_size_d = M * N * sizeof(DT_ACC);

    DT *A, *B;
    DT_ACC *C, *D;

    checkCuda(cudaMalloc(&A, mem_size_a));
    checkCuda(cudaMalloc(&B, mem_size_b));
    checkCuda(cudaMalloc(&C, mem_size_c));
    checkCuda(cudaMalloc(&D, mem_size_d));

    const int shmem_size = prop->sharedMemPerMultiprocessor;
    const int num_sms = prop->multiProcessorCount;

    for (int tile_dim = tile_dim_start; tile_dim <= tile_dim_end; tile_dim *= 2) {
        matmul_test::MatmulTestGemmSquare<DT, DT_ACC> mt(M, tile_dim, val_min, val_max);
        checkCuda(cudaMemcpy(A, mt.GetA(), mem_size_a, cudaMemcpyHostToDevice));
        checkCuda(cudaMemcpy(B, mt.GetB(), mem_size_b, cudaMemcpyHostToDevice));
        checkCuda(cudaMemcpy(C, mt.GetC(), mem_size_c, cudaMemcpyHostToDevice));
        for (int segment_dim_k = segment_dim_k_start; segment_dim_k <= segment_dim_k_end; segment_dim_k *= 2) {
            for (int tile_group_m = tile_group_m_start; tile_group_m <= tile_group_m_end; ++tile_group_m) {
                for (int num_stages = num_stages_start; num_stages <= num_stages_end; ++num_stages) {
                    for (int num_producer_warps = num_producer_warps_start; num_producer_warps <= num_producer_warps_end;
                         num_producer_warps *= 2) {
                        for (int num_consumer_warps = num_consumer_warps_start; num_consumer_warps <= num_consumer_warps_end;
                             num_consumer_warps *= 2) {
                            const int warp_tile_rows = get_warp_tile_rows(tile_dim, num_consumer_warps);
                            const int warp_tile_cols = get_num_acc(tile_dim, num_consumer_warps) / warp_tile_rows;
                            const int bar_step_rows_a[3] = {WMMA_M, warp_tile_rows * WMMA_M, tile_dim};
                            const int bar_step_rows_b[3] = {WMMA_N, warp_tile_cols * WMMA_N, tile_dim};
                            for (int i = 1; i < 3; ++i) {
                                if (get_shmem_req<DT, DT_ACC>(tile_dim, segment_dim_k, bar_step_rows_a[i],
                                        bar_step_rows_b[i], num_stages, num_consumer_warps) <= shmem_size &&
                                    !(K % segment_dim_k)) {
                                    int count = 0;
                                    if (segment_dim_k <= tile_dim) {
                                        for (int j = 0; j < num_reps; ++j) {
                                            checkCuda(cudaMemset(D, 0, mem_size_d));
                                            matmul_test::MatDim dim = mt.GetMatDim();
                                            gemm_shmem_tc_async_opt_port<DT, DT_ACC>(A, B, C, D, alpha, beta,
                                                dim.m, dim.n, dim.k, tile_dim, segment_dim_k, bar_step_rows_a[i],
                                                bar_step_rows_b[i], tile_group_m, num_stages, num_producer_warps,
                                                num_consumer_warps, num_sms);
                                            checkCuda(cudaMemcpy(mt.GetRes(), D, mem_size_d, cudaMemcpyDeviceToHost));
                                            count += mt.GetNumIncorrect(mt.GetRes(), dim, alpha, beta, atol);
                                        }
                                        printf("(%d, %d, %d, %d, %d, %d, %d, %d): %.2f / %d\n", tile_dim, segment_dim_k,
                                            bar_step_rows_a[i], bar_step_rows_b[i], tile_group_m, num_stages,
                                            num_producer_warps, num_consumer_warps, static_cast<float>(count) / num_reps,
                                            M * N);
                                    } else {
                                        checkCuda(cudaMemset(D, 0, mem_size_d));
                                        matmul_test::MatDim dim = mt.GetMatDimMax();
                                        gemm_shmem_tc_async_opt_port<DT, DT_ACC>(A, B, C, D, alpha, beta,
                                            dim.m, dim.n, dim.k, tile_dim, segment_dim_k, bar_step_rows_a[i],
                                            bar_step_rows_b[i], tile_group_m, num_stages, num_producer_warps,
                                            num_consumer_warps, num_sms);
                                        checkCuda(cudaMemcpy(mt.GetRes(), D, mem_size_d, cudaMemcpyDeviceToHost));
                                        count = mt.GetNumIncorrect(mt.GetRes(), dim, alpha, beta, atol);
                                        printf("(%d, %d, %d, %d, %d, %d, %d, %d): %.2f / %d\n", tile_dim, segment_dim_k,
                                            bar_step_rows_a[i], bar_step_rows_b[i], tile_group_m, num_stages,
                                            num_producer_warps, num_consumer_warps, static_cast<float>(count) / num_reps,
                                            M * N);
                                    }
                                }
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
    int segment_dim_k_start,
    int segment_dim_k_end,
    int tile_group_m_start,
    int tile_group_m_end,
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
    const int mem_size_a = M * K * sizeof(DT);
    const int mem_size_b = K * N * sizeof(DT);
    const int mem_size_c = M * N * sizeof(DT_ACC);
    const int mem_size_d = M * N * sizeof(DT_ACC);

    DT *A, *B;
    DT_ACC *C, *D;

    checkCuda(cudaMalloc(&A, mem_size_a));
    checkCuda(cudaMalloc(&B, mem_size_b));
    checkCuda(cudaMalloc(&C, mem_size_c));
    checkCuda(cudaMalloc(&D, mem_size_d));

    const int shmem_size = prop->sharedMemPerMultiprocessor;
    const int num_sms = prop->multiProcessorCount;

    cudaEvent_t start, stop;
    checkCuda(cudaEventCreate(&start));
    checkCuda(cudaEventCreate(&stop));

    for (int tile_dim = tile_dim_start; tile_dim <= tile_dim_end; tile_dim *= 2) {
        matmul_test::MatmulTestGemmSquare<DT, DT_ACC> mt(M, tile_dim, val_min, val_max);
        checkCuda(cudaMemcpy(A, mt.GetA(), mem_size_a, cudaMemcpyHostToDevice));
        checkCuda(cudaMemcpy(B, mt.GetB(), mem_size_b, cudaMemcpyHostToDevice));
        checkCuda(cudaMemcpy(C, mt.GetC(), mem_size_c, cudaMemcpyHostToDevice));
        for (int segment_dim_k = segment_dim_k_start; segment_dim_k <= segment_dim_k_end; segment_dim_k *= 2) {
            for (int tile_group_m = tile_group_m_start; tile_group_m <= tile_group_m_end; ++tile_group_m) {
                for (int num_stages = num_stages_start; num_stages <= num_stages_end; ++num_stages) {
                    for (int num_producer_warps = num_producer_warps_start; num_producer_warps <= num_producer_warps_end;
                         num_producer_warps *= 2) {
                        for (int num_consumer_warps = num_consumer_warps_start; num_consumer_warps <= num_consumer_warps_end;
                             num_consumer_warps *= 2) {
                            const int warp_tile_rows = get_warp_tile_rows(tile_dim, num_consumer_warps);
                            const int warp_tile_cols = get_num_acc(tile_dim, num_consumer_warps) / warp_tile_rows;
                            const int bar_step_rows_a[3] = {WMMA_M, warp_tile_rows * WMMA_M, tile_dim};
                            const int bar_step_rows_b[3] = {WMMA_N, warp_tile_cols * WMMA_N, tile_dim};
                            for (int i = 1; i < 2; ++i) {
                                if (get_shmem_req<DT, DT_ACC>(tile_dim, segment_dim_k, bar_step_rows_a[i],
                                        bar_step_rows_b[i], num_stages, num_consumer_warps) <= shmem_size &&
                                    !(K % segment_dim_k)) {
                                    float ms = 0.0f;
                                    matmul_test::MatDim dim = mt.GetMatDimMax();
                                    gemm_shmem_tc_async_opt_port<DT, DT_ACC>(A, B, C, D, alpha, beta,
                                        dim.m, dim.n, dim.k, tile_dim, segment_dim_k, bar_step_rows_a[i],
                                        bar_step_rows_b[i], tile_group_m, num_stages, num_producer_warps,
                                        num_consumer_warps, num_sms);
                                    checkCuda(cudaEventRecord(start, 0));
                                    for (int j = 0; j < num_reps; ++j) {
                                        gemm_shmem_tc_async_opt_port<DT, DT_ACC>(A, B, C, D, alpha, beta,
                                            dim.m, dim.n, dim.k, tile_dim, segment_dim_k, bar_step_rows_a[i],
                                            bar_step_rows_b[i], tile_group_m, num_stages, num_producer_warps,
                                            num_consumer_warps, num_sms);
                                    }
                                    checkCuda(cudaEventRecord(stop, 0));
                                    checkCuda(cudaEventSynchronize(stop));
                                    checkCuda(cudaEventElapsedTime(&ms, start, stop));
                                    printf("(%d, %d, %d, %d, %d, %d, %d, %d): %.2f\n", tile_dim, segment_dim_k,
                                        bar_step_rows_a[i], bar_step_rows_b[i], tile_group_m, num_stages,
                                        num_producer_warps, num_consumer_warps, 2 * M * N * 1e-9 * K * num_reps / ms);
                                }
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

    printf("\n\n%-30s", "gemm_shmem_tc_async_opt_port, <half, half>, (1024, 1024, 1024), correctness:\n");
    RunCorrectnessTestSquare<half, half>(
        &prop, 1.0f, 1.0f, M, N, K, 64, 128, 64, 256, 1, 6, 1, 10, 2, 8, 4, 8, 2, 0.001f);

    printf("\n\n%-30s", "gemm_shmem_tc_async_opt_port, <half, float>, (1024, 1024, 1024), correctness:\n");
    RunCorrectnessTestSquare<half, float>(
        &prop, 1.0f, 1.0f, M, N, K, 64, 128, 64, 256, 1, 6, 1, 10, 2, 8, 4, 8, 2, 0.001f);

    // Accuracy tests.

    M = 256;
    N = 256;
    K = 256;

    printf("\n\n%-30s", "gemm_shmem_tc_async_opt_port, <half, half>, (256, 256, 256), accuracy:\n");
    RunAccuracyTestSquare<half, half>(
        &prop, 1.0f, 1.0f, M, N, K, 64, 128, 64, 256, 2, 6, 1, 4, 2, 8, 4, 8, 2, -1.0f, 1.0f, 0.1f);

    printf("\n\n%-30s", "gemm_shmem_tc_async_opt_port, <half, float>, (256, 256, 256), accuracy:\n");
    RunAccuracyTestSquare<half, float>(
        &prop, 1.0f, 1.0f, M, N, K, 64, 128, 64, 256, 2, 6, 1, 4, 2, 8, 4, 8, 2, -1.0f, 1.0f, 0.1f);

    M = 512;
    N = 512;
    K = 512;

    printf("\n\n%-30s", "gemm_shmem_tc_async_opt_port, <half, half>, (512, 512, 512), accuracy:\n");
    RunAccuracyTestSquare<half, half>(
        &prop, 1.0f, 1.0f, M, N, K, 64, 128, 64, 256, 2, 6, 1, 4, 2, 8, 4, 8, 2, -1.0f, 1.0f, 0.1f);

    printf("\n\n%-30s", "gemm_shmem_tc_async_opt_port, <half, float>, (512, 512, 512), accuracy:\n");
    RunAccuracyTestSquare<half, float>(
        &prop, 1.0f, 1.0f, M, N, K, 64, 128, 64, 256, 2, 6, 1, 4, 2, 8, 4, 8, 2, -1.0f, 1.0f, 0.1f);

    // Performance tests.

    M = 16384;
    N = 16384;
    K = 16384;

    printf("\n\n%-30s", "gemm_shmem_tc_async_opt_port, <half, half>, (16384, 16384, 16384), [TFLOPS]:\n");
    RunPerformanceTestSquare<half, half>(
        &prop, 1.0f, 1.0f, M, N, K, 64, 128, 64, 256, 2, 6, 1, 4, 2, 8, 4, 8, 2, -1.0f, 1.0f, 0.1f);

    printf("\n\n%-30s", "gemm_shmem_tc_async_opt_port, <half, float>, (16384, 16384, 16384), [TFLOPS]:\n");
    RunPerformanceTestSquare<half, float>(
        &prop, 1.0f, 1.0f, M, N, K, 64, 128, 64, 256, 2, 6, 1, 4, 2, 8, 4, 8, 2, -1.0f, 1.0f, 0.1f);

    return 0;
}
