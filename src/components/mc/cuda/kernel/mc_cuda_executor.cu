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
#include <inttypes.h>

#ifdef __cplusplus
}
#endif

#define align_pow2(_n, _p) ((_n) & ((_p) - 1))

__device__ inline void add_float4(float4 &d, const float4 &x, const float4 &y)
{
    d.x = x.x + y.x;
    d.y = x.y + y.y;
    d.z = x.z + y.z;
    d.w = x.w + y.w;
}

__device__ inline void add_float4(float4 &x, const float4 &y)
{
    x.x += y.x;
    x.y += y.y;
    x.z += y.z;
    x.w += y.w;
}

__device__ void executor_reduce_float_multi(float **s, float *d,
                                            size_t count, uint32_t size)
{
    float4 **s4 = (float4**)s;
    float4  *d4 = (float4*)d;
    const int idx  = threadIdx.x;
    const int step = blockDim.x;
    const int n = count / 4;
    const int num_iter = n / step + ((idx < n % step) ? 1 : 0);

    for(int i = 0; i < num_iter; i++) {
        float4 temp;
        add_float4(temp, s4[0][i * step + idx], s4[1][i * step + idx]);
#pragma unroll 2
        for(int j = 2; j < size; j++) {
            add_float4(temp, s4[j][i * step + idx]);
        }
        d4[i * step + idx] = temp;
    }
    // if (idx < count % sizeof(float4)) {
    //     d[count - idx - 1] = s1[count - idx - 1] + s2[count - idx - 1];
    // }

}

__device__ void executor_reduce_float(const float *s1, const float *s2,
                                      float *d, size_t count)
{
    const float4 *s14 = (const float4*)s1;
    const float4 *s24 = (const float4*)s2;
    float4       *d4  = (float4*)d;
    const size_t idx = threadIdx.x;
    const size_t step = blockDim.x;
    const int n = count / 4;
    const int num_iter = n / step + ((idx < n % step) ? 1 : 0);

    for(int i = 0; i < num_iter; i++) {
        float4 d  = d4[i * step + idx];
        add_float4(d, s14[i * step + idx], s24[i * step + idx]);
        d4[i * step + idx] = d;
    }
    if (idx < count % sizeof(float4)) {
        d[count - idx - 1] = s1[count - idx - 1] + s2[count - idx - 1];
    }

}

template <typename T>
__device__ void executor_reduce(const T* __restrict__ s1,
                                const T* __restrict__ s2,
                                T* __restrict__ d, size_t count)
{
    size_t start = threadIdx.x;
    const size_t step  = blockDim.x;

    for (size_t i = start; i < count; i+=step) {
        d[i] = s1[i] + s2[i];
    }
}

template <typename T>
__device__ void executor_copy(T* __restrict__ d, T* __restrict__ s,
                              size_t count)
{
    size_t start = threadIdx.x;
    const size_t step  = blockDim.x;

    for (size_t i = start; i < count; i+=step) {
        d[i] = s[i];
    }
}

template <typename T>
__device__ void executor_copy_aligned(T* __restrict__ d, T* __restrict__ s,
                                      size_t count)
{
    size_t idx = threadIdx.x;
    const size_t step  = blockDim.x;
    const int n = count / sizeof(T);
    const int num_iter = n / step + ((idx < n % step) ? 1 : 0);
    char1 *s1 = (char1*)s;
    char1 *d1 = (char1*)d;

#pragma unroll 4
    for(int i = 0; i < num_iter; i++) {
        d[i * step + idx] = s[i * step + idx];
    }

    if (idx < count % sizeof(T)) {
        d1[count - idx - 1] = s1[count - idx - 1];
    }
}

__global__ void executor_start(volatile ucc_mc_cuda_executor_t *eee)
{
    *eee->dev_cidx  = 0;
    *eee->dev_state = UCC_MC_CUDA_EXECUTOR_STARTED;
}

__global__ void executor_shutdown_ack(volatile ucc_mc_cuda_executor_t *eee)
{
    *eee->dev_state = UCC_MC_CUDA_EXECUTOR_SHUTDOWN_ACK;
}

__global__ void executor_kernel(volatile ucc_mc_cuda_executor_t *eee,
                                int q_size)
{
    const uint32_t  worker_id   = blockIdx.x;
    const uint32_t  num_workers = gridDim.x;
    bool            is_master   = (threadIdx.x == 0) ? true: false;
    int             cidx_local, pidx_local;
    volatile int *pidx, *cidx;
    ucc_ee_executor_task_t *tasks;
    __shared__ ucc_ee_executor_task_args_t args;
    __shared__ bool worker_done;

    if (is_master) {
        cidx_local = worker_id;
        pidx       = eee->dev_pidx;
        cidx       = eee->dev_cidx;
        tasks      = eee->dev_tasks;
    }

    worker_done = false;
    __syncthreads();
    while (1) {
        if (is_master) {
            while ((*cidx % num_workers) != worker_id);
            do {
                pidx_local = *pidx;
            } while (*cidx == pidx_local);
            (*cidx)++;
            worker_done = (pidx_local == -1);
            if (!worker_done) {
                args = tasks[cidx_local].args;
            }
        }
        __syncthreads();
        if (worker_done) {
            return;
        }
        switch (args.task_type) {
            bool aligned;
            case UCC_MC_EE_EXECUTOR_TASK_TYPE_REDUCE:
                aligned = !(align_pow2((intptr_t)args.bufs[0], 16) ||
                            align_pow2((intptr_t)args.bufs[1], 16) ||
                            align_pow2((intptr_t)args.bufs[2], 16));
                if (aligned) {
                    executor_reduce_float((float*)args.bufs[1],
                                          (float*)args.bufs[2],
                                          (float*)args.bufs[0],
                                          args.count);
                } else {
                    executor_reduce<float>((float*)args.bufs[1],
                                           (float*)args.bufs[2],
                                           (float*)args.bufs[0],
                                           args.count);
                }
                break;
            case UCC_MC_EE_EXECUTOR_TASK_TYPE_COPY:
                aligned = !(align_pow2((intptr_t)args.bufs[0], 16) ||
                            align_pow2((intptr_t)args.bufs[1], 16));
                if (aligned) {
                    executor_copy_aligned<uint4>((uint4*)args.bufs[0],
                                                 (uint4*)args.bufs[1],
                                                 args.count);

                } else {
                    executor_copy((char*)args.bufs[0],
                                  (char*)args.bufs[1],
                                   args.count);
                }
                break;
            case UCC_MC_EE_EXECUTOR_TASK_TYPE_REDUCE_MULTI:
                executor_reduce_float_multi((float**)&args.bufs[1],
                                            (float*)args.bufs[0],
                                            args.count,
                                            args.size);
                break;
        }
        __syncthreads();
        __threadfence_system();
        if (is_master) {
            tasks[cidx_local].status = UCC_OK;
            cidx_local = (cidx_local + num_workers) %q_size;
        }
    }
}

#ifdef __cplusplus
extern "C" {
#endif

ucc_status_t ucc_mc_cuda_start_executor(ucc_mc_cuda_executor_t *eee)
{
    cudaStream_t stream = (cudaStream_t)eee->super.ee_context;
    int          nb     = MC_CUDA_CONFIG->exec_num_workers;
    int          nt     = MC_CUDA_CONFIG->exec_num_threads;
    int          q_size = MC_CUDA_CONFIG->exec_max_tasks;

    executor_start<<<1, 1, 0, stream>>>(eee);
    executor_kernel<<<nb, nt, 0, stream>>>(eee, q_size);
    executor_shutdown_ack<<<1, 1, 0, stream>>>(eee);
    CUDACHECK(cudaGetLastError());

    return UCC_OK;
}


#ifdef __cplusplus
}
#endif
