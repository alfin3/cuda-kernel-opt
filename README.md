# matmul_shmem.cu, matmul_shmem_tc_async_opt_port.cu


The design of the `matmul_shmem` and `gemm_shmem_tc_async_opt_port` kernels enabled the investigation of the effect of different levels of workloads, mapped to registers by a compiler, in the context of matrix multiplication.

The evaluation of the `matmul_shmem` kernel showed that optimizing the number of accumulators per thread could improve performance without using Tensor Cores. An approach to optimizing the number of accumulators at the level of warps using Tensor Cores was then implemented in the `gemm_shmem_tc_async_opt_port` kernel. This kernel was portable and tunable. It enabled the evaluation of different levels of matrix multiply and accumulate workloads across kernel configurations and GPU architectures.

The `gemm_shmem_tc_async_opt_port` kernel may also be used for selecting configurations across GPU architectures for further optimization with architecture-specific features that may limit portability.
