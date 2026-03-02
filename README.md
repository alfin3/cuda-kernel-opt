The objective of this project is to develop and implement innovative concepts related to matrix multiplication that may be different from the commonly used optimizations, and therefore could potentially enable to exceed the state-of-the-art performance in some cases of matrix multiplication in combination with other implementations.

# matmul_shmem_tc_async_opt_port_0.cu

The design of the `gemm_shmem_tc_async_opt_port` kernel enabled the investigation of the effect of different levels of workloads, mapped to registers by a compiler, in the context of matrix multiplication.

The evaluation of the `matmul_shmem` kernel (`matmul_shmem.cu`) showed that optimizing the number of accumulators per thread could improve performance without using Tensor Cores. An approach to optimizing the number of accumulators at the level of warps using Tensor Cores was then implemented in the `gemm_shmem_tc_async_opt_port` kernel. This kernel was portable and tunable.

# matmul_shmem_tc_async_opt_port_1.cu

The synchronization scheme, provided in the `gemm_shmem_tc_async_opt_port` kernel, was designed to i) decouple the consumer warps from each other, including at the level of accumulators, and ii) shift the start of the execution by each consumer warp according to the order of the load instructions for the A and transposed B segments of the K dimension. The order of the load instructions was from top to bottom. The earlier load instructions should result in an earlier start of matrix multiply and accumulate. The later load instructions should result in a later start of matrix multiply and accumulate. This shift should be preserved across the pipeline stages and enable better utilization of Tensor Cores.


# matmul_shmem_tc_async_opt_port_2.cu


"Load instruction tiling" was introduced as a method for splitting the K dimension to achieve an earlier start of data consumption in the synchronization scheme provided in `matmul_shmem_tc_async_opt_port_1.cu`.
