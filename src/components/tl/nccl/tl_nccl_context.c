/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "tl_nccl.h"
#include "core/ucc_mc.h"
#include "core/ucc_ee.h"

ucc_status_t ucc_tl_nccl_collective_progress(ucc_coll_task_t *coll_task)
{
    ucc_tl_nccl_task_t *task = ucc_derived_of(coll_task, ucc_tl_nccl_task_t);
    ucc_status_t status;

    status = ucc_mc_ee_event_test(task->completed, UCC_EE_CUDA_STREAM);
    coll_task->super.status = status;
    return status;
}

static void ucc_tl_nccl_req_mpool_obj_init(ucc_mpool_t *mp, void *obj,
                                           void *chunk)
{
    ucc_tl_nccl_task_t *req = (ucc_tl_nccl_task_t*) obj;
    req->super.progress = ucc_tl_nccl_collective_progress;
}

static ucc_mpool_ops_t ucc_tl_nccl_req_mpool_ops = {
    .chunk_alloc   = ucc_mpool_hugetlb_malloc,
    .chunk_release = ucc_mpool_hugetlb_free,
    .obj_init      = ucc_tl_nccl_req_mpool_obj_init,
    .obj_cleanup   = NULL
};

static ucc_status_t ucc_tl_nccl_req_mapped_mpool_chunk_malloc(ucc_mpool_t *mp,
                                                              size_t *size_p,
                                                              void ** chunk_p)
{
    ucc_status_t status;

    // TODO check for error
    status = cudaHostAlloc((void**)chunk_p, *size_p, cudaHostAllocMapped);
    return status;
}

static void ucc_tl_nccl_req_mapped_mpool_chunk_free(ucc_mpool_t *mp,
                                                    void *chunk)
{
    cudaFreeHost(chunk);
}

static void ucc_tl_nccl_req_mapped_mpool_obj_init(ucc_mpool_t *mp, void *obj,
                                                  void *chunk)
{
    ucc_tl_nccl_task_t *req = (ucc_tl_nccl_task_t*) obj;

    // TODO check for error
    cudaHostGetDevicePointer((void**)(&req->dev_status),
                             (void *)&req->super.super.status, 0);
    req->super.progress = NULL;
}

static ucc_mpool_ops_t ucc_tl_nccl_req_mapped_mpool_ops = {
    .chunk_alloc   = ucc_tl_nccl_req_mapped_mpool_chunk_malloc,
    .chunk_release = ucc_tl_nccl_req_mapped_mpool_chunk_free,
    .obj_init      = ucc_tl_nccl_req_mapped_mpool_obj_init,
    .obj_cleanup   = NULL
};

UCC_CLASS_INIT_FUNC(ucc_tl_nccl_context_t,
                    const ucc_base_context_params_t *params,
                    const ucc_base_config_t *config)
{
    ucc_tl_nccl_context_config_t *tl_nccl_config =
        ucc_derived_of(config, ucc_tl_nccl_context_config_t);
    ucc_status_t status;
    int mem_ops_attr;
    CUresult cu_st;
    CUdevice cu_dev;

    UCC_CLASS_CALL_SUPER_INIT(ucc_tl_context_t, tl_nccl_config->super.tl_lib,
                              params->context);
    memcpy(&self->cfg, tl_nccl_config, sizeof(*tl_nccl_config));

//TODO add sync_type auto instead
    if (self->cfg.sync_type == UCC_TL_NCCL_COMPLETION_SYNC_TYPE_MEMOPS) {
        cu_st = cuCtxGetDevice(&cu_dev);
        if (cu_st == CUDA_SUCCESS) {
            cu_st = cuDeviceGetAttribute(&mem_ops_attr,
                                        CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS,
                                        cu_dev);
            if (mem_ops_attr == 0) {
                tl_info(self->super.super.lib, "memops disabled or not supported, "
                                               "fallback to event query completion sync");
                self->cfg.sync_type = UCC_TL_NCCL_COMPLETION_SYNC_TYPE_EVENT;
            } else {
                tl_info(self->super.super.lib, "using memops completion sync");
            }
        } else {
            tl_info(self->super.super.lib, "failed to get cuda device, "
                                           "fallback to event query completion sync");
            self->cfg.sync_type = UCC_TL_NCCL_COMPLETION_SYNC_TYPE_EVENT;
        }
    }

    if (self->cfg.sync_type == UCC_TL_NCCL_COMPLETION_SYNC_TYPE_MEMOPS) {
        status = ucc_mpool_init(&self->req_mp, 0, sizeof(ucc_tl_nccl_task_t), 0,
                                UCC_CACHE_LINE_SIZE, 8, UINT_MAX,
                                &ucc_tl_nccl_req_mapped_mpool_ops,
                                params->thread_mode, "tl_nccl_req_mp");
    } else {
        status = ucc_mpool_init(&self->req_mp, 0, sizeof(ucc_tl_nccl_task_t), 0,
                                UCC_CACHE_LINE_SIZE, 8, UINT_MAX,
                                &ucc_tl_nccl_req_mpool_ops, params->thread_mode,
                                "tl_nccl_req_mp");
    }
    if (status != UCC_OK) {
        tl_error(self->super.super.lib,
                 "failed to initialize tl_nccl_req mpool");
        return status;
    }
    tl_info(self->super.super.lib, "initialized tl context: %p", self);
    return UCC_OK;
}

UCC_CLASS_CLEANUP_FUNC(ucc_tl_nccl_context_t)
{
    tl_info(self->super.super.lib, "finalizing tl context: %p", self);
    ucc_mpool_cleanup(&self->req_mp, 1);
}

UCC_CLASS_DEFINE(ucc_tl_nccl_context_t, ucc_tl_context_t);

ucc_status_t ucc_tl_nccl_get_context_attr(const ucc_base_context_t *context, /* NOLINT */
                                          ucc_base_attr_t *attr) /* NOLINT */
{
    /* TODO */
    return UCC_ERR_NOT_IMPLEMENTED;
}
