The objective of this project is to develop and implement innovative concepts related to matrix multiplication that may be different from the commonly used optimizations, and therefore could potentially enable to exceed the state-of-the-art performance in some cases of matrix multiplication in combination with other implementations.

# matmul_shmem_tc_async_opt_port_0.cu

The design of the `gemm_shmem_tc_async_opt_port` kernel enabled the investigation of the effect of different levels of workloads, mapped to registers by a compiler, in the context of matrix multiplication.

The evaluation of the `matmul_shmem` kernel (`matmul_shmem.cu`) showed that optimizing the number of accumulators per thread could improve performance without using Tensor Cores. An approach to optimizing the number of accumulators at the level of warps using Tensor Cores was then implemented in the `gemm_shmem_tc_async_opt_port` kernel. This kernel was portable and tunable.

# matmul_shmem_tc_async_opt_port_1.cu
