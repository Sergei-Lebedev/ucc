#ifdef __cplusplus
extern "C" {
#endif
#include "../ec_cuda.h"
#include "utils/ucc_malloc.h"
#ifdef __cplusplus
}
#endif

#include "dietgpu/float/GpuFloatCodec.h"
#include "dietgpu/float/GpuFloatUtils.cuh"
#include "dietgpu/utils/StackDeviceMemory.h"

#ifdef __cplusplus
extern "C" {
#endif

namespace dg = dietgpu;

#define MAX_NUM_BUFS 1024
#define PROB_BITS 10

void ucc_ec_cuda_compress_resources_chunk_init(ucc_mpool_t *mp, void *obj, void *chunk)
{
    int dev;
    ucc_ec_cuda_executor_compress_resources_t *base = (ucc_ec_cuda_executor_compress_resources_t *) obj;

    cudaGetDevice(&dev);
    base->stack_memory = new dg::StackDeviceMemory(dev, 256 * 1024 * 1024);
    CUDA_FUNC(cudaHostAlloc(&base->out_size_host, MAX_NUM_BUFS * sizeof(uint32_t),
                            cudaHostAllocMapped));
    CUDA_FUNC(cudaHostGetDevicePointer(&base->out_size_dev, base->out_size_host,
                                       0));
    base->srcs  = (void**)ucc_malloc(MAX_NUM_BUFS * sizeof(*base->srcs), "comp resources srcs");
    base->dsts  = (void**)ucc_malloc(MAX_NUM_BUFS * sizeof(*base->dsts), "comp resources dsts");
    base->count = (uint32_t*)ucc_malloc(MAX_NUM_BUFS * sizeof(*base->count), "comp resources count");
}

void ucc_ec_cuda_compress_resources_chunk_cleanup(ucc_mpool_t *mp, void *obj)
{
    ucc_ec_cuda_executor_compress_resources_t *base = (ucc_ec_cuda_executor_compress_resources_t *) obj;
    delete static_cast<dg::StackDeviceMemory*>(base->stack_memory);
    cudaFreeHost(base->out_size_host);
    ucc_free(base->srcs);
    ucc_free(base->dsts);
    ucc_free(base->count);
}

static size_t get_count(uint64_t mask, const ucc_count_t *counts, int idx)
{
    if (mask & UCC_EE_TASK_COMPRESS_FLAG_COUNT64) {
        return ((uint64_t *)counts)[idx];
    } else {
        return ((uint32_t *)counts)[idx];
    }
}

static void set_count(uint64_t mask, const ucc_count_t *counts, int idx,
                      size_t count)
{
    if (mask & UCC_EE_TASK_COMPRESS_FLAG_COUNT64) {
        ((uint64_t *)counts)[idx] = count;
    } else {
        ((uint32_t *)counts)[idx] = count;
    }
}

static size_t get_displacement(uint64_t mask, const ucc_aint_t *displ, int idx)
{
    if (mask & UCC_EE_TASK_COMPRESS_FLAG_DISPLACEMENT64) {
        return ((uint64_t *)displ)[idx];
    } else {
        return ((uint32_t *)displ)[idx];
    }
}
void ucc_ec_cuda_compress_epilog(ucc_eee_task_compress_t *task,
                                 ucc_ec_cuda_executor_compress_resources_t *resources)
{
    for (int i = 0; i < task->size; i++) {
        set_count(task->flags, task->dst_counts, i, resources->out_size_host[i]);
    }
    return;
}

ucc_status_t ucc_ec_cuda_compress(ucc_eee_task_compress_t *task,
                                  ucc_ec_cuda_executor_compress_resources_t *resources,
                                  cudaStream_t stream)
{
    dg::StackDeviceMemory *temp_mem = (dg::StackDeviceMemory*)resources->stack_memory;
    size_t                 dt_size  = ucc_dt_size(task->dt);
    dg::FloatCompressConfig comp_config;
    dg::FloatType dt;

    if (!UCC_DT_IS_PREDEFINED(task->dt)) {
        ec_error(&ucc_ec_cuda.super, "user defined dt is not supported");
        return UCC_ERR_NOT_SUPPORTED;
    }

    switch(task->dt) {
    case UCC_DT_FLOAT32:
        dt = dg::kFloat32;
        break;
    case UCC_DT_FLOAT16:
        dt = dg::kFloat16;
        break;
    case UCC_DT_BFLOAT16:
        dt = dg::kBFloat16;
        break;
    default:
        ec_error(&ucc_ec_cuda.super, "datatype is not supported");
        return UCC_ERR_NOT_SUPPORTED;
    }

    for (int i = 0; i < task->size; i++) {
        resources->srcs[i]  = PTR_OFFSET(task->src, dt_size * get_displacement(task->flags, task->src_displacements, i));
        resources->dsts[i]  = PTR_OFFSET(task->dst, get_displacement(task->flags, task->dst_displacements, i));
        resources->count[i] = get_count(task->flags, task->src_counts, i);
    }

    comp_config = dg::FloatCompressConfig(dt, PROB_BITS, false);

    floatCompress(*temp_mem, comp_config, task->size,
                  (const void**)resources->srcs, resources->count,
                  resources->dsts, resources->out_size_dev, stream);
    return UCC_OK;
}

ucc_status_t ucc_ec_cuda_decompress(ucc_eee_task_compress_t *task,
                                    ucc_ec_cuda_executor_compress_resources_t *resources,
                                    cudaStream_t stream)
{
    dg::StackDeviceMemory *temp_mem = (dg::StackDeviceMemory*)resources->stack_memory;
    size_t                 dt_size  = ucc_dt_size(task->dt);
    dg::FloatCompressConfig decomp_config;
    dg::FloatType dt;

    if (!UCC_DT_IS_PREDEFINED(task->dt)) {
        ec_error(&ucc_ec_cuda.super, "user defined dt is not supported");
        return UCC_ERR_NOT_SUPPORTED;
    }

    switch(task->dt) {
    case UCC_DT_FLOAT32:
        dt = dg::kFloat32;
        break;
    case UCC_DT_FLOAT16:
        dt = dg::kFloat16;
        break;
    case UCC_DT_BFLOAT16:
        dt = dg::kBFloat16;
        break;
    default:
        ec_error(&ucc_ec_cuda.super, "datatype is not supported");
        return UCC_ERR_NOT_SUPPORTED;
    }

    for (int i = 0; i < task->size; i++) {
        resources->srcs[i]  = PTR_OFFSET(task->src, get_displacement(task->flags, task->src_displacements, i));
        resources->dsts[i]  = PTR_OFFSET(task->dst, dt_size * get_displacement(task->flags, task->dst_displacements, i));
        resources->count[i] = get_count(task->flags, task->dst_counts, i);
    }

    decomp_config = dg::FloatDecompressConfig(dt, PROB_BITS, false);

    floatDecompress(*temp_mem, decomp_config, task->size, (const void**)resources->srcs,
                    resources->dsts, resources->count, nullptr,
                    resources->out_size_dev, stream);
    return UCC_OK;
}

ucc_status_t ucc_ec_cuda_get_compress_size(size_t count_in,
                                           ucc_datatype_t dt,
                                           size_t *count_out)
{
    dg::FloatType dgdt;

    if (!UCC_DT_IS_PREDEFINED(dt)) {
        ec_error(&ucc_ec_cuda.super, "user defined dt is not supported");
        return UCC_ERR_NOT_SUPPORTED;
    }


    switch(dt) {
    case UCC_DT_FLOAT32:
        dgdt = dg::kFloat32;
        break;
    case UCC_DT_FLOAT16:
        dgdt = dg::kFloat16;
        break;
    case UCC_DT_BFLOAT16:
        dgdt = dg::kBFloat16;
        break;
    default:
        ec_error(&ucc_ec_cuda.super, "datatype is not supported");
        return UCC_ERR_NOT_SUPPORTED;
    }

    *count_out = getMaxFloatCompressedSize(dgdt, count_in);
    return UCC_OK;
}

#ifdef __cplusplus
}
#endif
