/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef __cplusplus
extern "C" {
#endif

#include "../mc_cuda.h"
#include "utils/ucc_math.h"

#ifdef __cplusplus
}
#endif


#define CUDA_EXECUTOR_REDUCE(NAME, OP)                                         \
template <typename T>                                                          \
__device__ void CUDA_EXECUTOR_REDUCE_ ## NAME (const T *s1, const T *s2, T *d, \
                                               size_t count)                   \
{                                                                              \
        size_t start = blockIdx.x * blockDim.x + threadIdx.x;                  \
        size_t step  = blockDim.x * gridDim.x;                                 \
        for (size_t i = start; i < count - 1; i+=step) {                           \
            d[i] = OP(s1[i], s2[i]);                                           \
        }                                                                      \
}                                                                              \

CUDA_EXECUTOR_REDUCE(MAX,  DO_OP_MAX)
CUDA_EXECUTOR_REDUCE(MIN,  DO_OP_MIN)
CUDA_EXECUTOR_REDUCE(SUM,  DO_OP_SUM)
CUDA_EXECUTOR_REDUCE(PROD, DO_OP_PROD)
CUDA_EXECUTOR_REDUCE(LAND, DO_OP_LAND)
CUDA_EXECUTOR_REDUCE(BAND, DO_OP_BAND)
CUDA_EXECUTOR_REDUCE(LOR,  DO_OP_LOR)
CUDA_EXECUTOR_REDUCE(BOR,  DO_OP_BOR)
CUDA_EXECUTOR_REDUCE(LXOR, DO_OP_LXOR)
CUDA_EXECUTOR_REDUCE(BXOR, DO_OP_BXOR)

__global__ void executor_kernel(volatile ucc_mc_cuda_executor_t *eee) {
    ucc_mc_task_status_t st;
    *eee->dev_status = UCC_MC_CUDA_EXECUTOR_STARTED;
    do {
        st = (ucc_mc_task_status_t)*eee->dev_status;
        if (st == UCC_MC_CUDA_TASK_POSTED) {
            switch (eee->dev_args->task_type)
            {
            case UCC_MC_EE_EXECUTOR_TASK_TYPE_REDUCE:
                if (eee->dev_args->dst.datatype != UCC_DT_INT32) {
                    printf("unsupported dataype\n");
                    break;
                }
                printf("start reduce\n");
                CUDA_EXECUTOR_REDUCE_SUM((int*)eee->dev_args->src1.buffer,
                                         (int*)eee->dev_args->src2.buffer,
                                         (int*)eee->dev_args->dst.buffer,
                                         eee->dev_args->dst.count);
                break;
            case UCC_MC_EE_EXECUTOR_TASK_TYPE_REDUCE_MULTI:
                if (eee->dev_args->dst.datatype != UCC_DT_INT32) {
                    printf("unsupported dataype\n");
                    break;
                }
                printf("start reduce multi\n");
                break;
            default:
                printf("unknown task\n");
            }
            *eee->dev_status = UCC_MC_CUDA_TASK_COMPLETED;
        }
    } while(st != UCC_MC_CUDA_EXECUTOR_SHUTDOWN);

    *eee->dev_status = UCC_MC_CUDA_EXECUTOR_SHUTDOWN_ACK;
    return;
}

#ifdef __cplusplus
extern "C" {
#endif

ucc_status_t ucc_mc_cuda_start_executor(ucc_mc_cuda_executor_t *eee)
{
    executor_kernel<<<1, 1, 0, (cudaStream_t)(eee->super.ee_context)>>>(eee);
    CUDACHECK(cudaGetLastError());
    return UCC_OK;
}

#ifdef __cplusplus
}
#endif
