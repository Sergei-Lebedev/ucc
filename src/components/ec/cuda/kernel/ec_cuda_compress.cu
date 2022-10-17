#ifdef __cplusplus
extern "C" {
#endif
#include "../ec_cuda.h"
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

#define MAX_NUM_BUFS 32

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

ucc_status_t ucc_ec_cuda_compress(ucc_eee_task_compress_t *task,
                                  cudaStream_t stream)
{
    const int prob_bits = 10;
    auto temp_mem = dg::makeStackMemory();
    uint32_t *out_size, *out_size_dev;
    dg::FloatCompressConfig comp_config;
    dg::FloatType dt;
    size_t dt_size = ucc_dt_size(task->dt);
    void *src[MAX_NUM_BUFS], *dst[MAX_NUM_BUFS];
    uint32_t src_count[MAX_NUM_BUFS];

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

    cudaHostAlloc(&out_size, task->size * sizeof(uint32_t), cudaHostAllocMapped);
    cudaHostGetDevicePointer(&out_size_dev, out_size, 0);

    for (int i = 0; i < task->size; i++) {
        src[i] = PTR_OFFSET(task->src, dt_size * get_displacement(task->flags, task->src_displacements, i));
        dst[i] = PTR_OFFSET(task->dst, get_displacement(task->flags, task->dst_displacements, i));
        src_count[i] = get_count(task->flags, task->src_counts, i);
    }

    comp_config = dg::FloatCompressConfig(dt, prob_bits, false);

    floatCompress(temp_mem, comp_config, task->size,
                  (const void**)src, src_count,
                  dst, out_size_dev, stream);
    cudaStreamSynchronize(stream);

    for(int i = 0; i < task->size; i++) {
        set_count(task->flags, task->dst_counts, i, out_size[i]);
    }
    cudaFreeHost(out_size);
    return UCC_OK;
}

ucc_status_t ucc_ec_cuda_decompress(ucc_eee_task_compress_t *task,
                                    cudaStream_t stream)
{
    const int prob_bits = 10;
    auto temp_mem = dg::makeStackMemory();
    uint32_t *out_size, *out_size_dev;
    dg::FloatCompressConfig decomp_config;
    dg::FloatType dt;
    size_t dt_size = ucc_dt_size(task->dt);
    void *src[MAX_NUM_BUFS], *dst[MAX_NUM_BUFS];
    uint32_t dst_count[MAX_NUM_BUFS];

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

    cudaHostAlloc(&out_size, task->size * sizeof(uint32_t), cudaHostAllocMapped);
    cudaHostGetDevicePointer(&out_size_dev, out_size, 0);

    for (int i = 0; i < task->size; i++) {
        src[i] = PTR_OFFSET(task->src, get_displacement(task->flags, task->src_displacements, i));
        dst[i] = PTR_OFFSET(task->dst, dt_size * get_displacement(task->flags, task->dst_displacements, i));
        dst_count[i] = get_count(task->flags, task->dst_counts, i);
    }

    decomp_config = dg::FloatDecompressConfig(dt, prob_bits, false);

    floatDecompress(temp_mem, decomp_config, task->size, (const void**)src,
                    dst, dst_count, nullptr, out_size_dev, stream);
    cudaStreamSynchronize(stream);

    cudaFreeHost(out_size);
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
