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
    *eee->dev_state = UCC_MC_CUDA_EXECUTOR_STARTED;
}

__global__ void executor_shutdown_ack(volatile ucc_mc_cuda_executor_t *eee)
{
    *eee->dev_state = UCC_MC_CUDA_EXECUTOR_SHUTDOWN_ACK;
}

__global__ void executor_kernel(volatile ucc_mc_cuda_executor_t *eee)
{
    const uint32_t    tid         = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t    worker_id   = blockIdx.x;
    const uint32_t    num_threads = blockDim.x;
    uint8_t           cidx        = 0;
    volatile uint8_t *pidx;
    volatile ucc_mc_cuda_executor_state_t *state;
    __shared__ ucc_ee_executor_task_t *task;
    __shared__ bool worker_done[NUM_WORKERS];
    __shared__ bool aligned;

    if (tid % num_threads == 0) {
        state = eee->dev_state;
        pidx  = &eee->dev_pidx[worker_id];
        worker_done[worker_id] = false;
    }

    while (1) {
        if (tid % num_threads == 0) {
            if (cidx == *pidx) {
                if (*state != UCC_MC_CUDA_EXECUTOR_SHUTDOWN) {
                    continue;
                }
                worker_done[worker_id] = true;
            } else {
                task = &eee->dev_tasks[worker_id * 8 + cidx];
                aligned = !(align_pow2((intptr_t)task->args.src1.buffer, 16) ||
                            align_pow2((intptr_t)task->args.dst.buffer, 16));
            }
        }

        __syncthreads();
        if (worker_done[worker_id]) {
            break;
        }
        switch (task->args.task_type) {
            case UCC_MC_EE_EXECUTOR_TASK_TYPE_REDUCE:
                // executor_reduce<float>((float*)task->args.src1.buffer,
                //                        (float*)task->args.src2.buffer,
                //                        (float*)task->args.dst.buffer,
                //                        task->args.dst.count);
                executor_reduce_float((float*)task->args.src1.buffer,
                                      (float*)task->args.src2.buffer,
                                      (float*)task->args.dst.buffer,
                                      task->args.dst.count);

                break;
            case UCC_MC_EE_EXECUTOR_TASK_TYPE_COPY:
                if (aligned) {
                    executor_copy_aligned<uint4>((uint4*)task->args.dst.buffer,
                                                 (uint4*)task->args.src1.buffer,
                                                 task->args.dst.count);

                } else {
                    executor_copy((char*)task->args.dst.buffer,
                                  (char*)task->args.src1.buffer,
                                   task->args.dst.count);
                }
                break;
        }

        __syncthreads();
        __threadfence_system();
        if (tid % num_threads == 0) {
            cidx = (cidx + 1) % 8;
            task->status = UCC_OK;
        }
    }
    return;
}

#ifdef __cplusplus
extern "C" {
#endif

ucc_status_t ucc_mc_cuda_start_executor(ucc_mc_cuda_executor_t *eee)
{
    executor_start<<<1, 1, 0, (cudaStream_t)(eee->super.ee_context)>>>(eee);
    executor_kernel<<<NUM_WORKERS, 512, 0, (cudaStream_t)(eee->super.ee_context)>>>(eee);
    executor_shutdown_ack<<<1, 1, 0, (cudaStream_t)(eee->super.ee_context)>>>(eee);

    CUDACHECK(cudaGetLastError());
    return UCC_OK;
}

#ifdef __cplusplus
}
#endif
