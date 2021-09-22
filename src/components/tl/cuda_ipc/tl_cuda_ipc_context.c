/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "tl_cuda_ipc.h"
#include "tl_cuda_ipc_coll.h"
#include "core/ucc_mc.h"
#include "core/ucc_ee.h"

static void ucc_tl_cuda_ipc_req_mpool_obj_init(ucc_mpool_t *mp, void *obj,
                                               void *chunk)
{

    ucc_tl_cuda_ipc_task_t *task = obj;
    cudaEventCreateWithFlags(&task->event,
                             cudaEventDisableTiming |
                             cudaEventInterprocess);
    //TODO check status
}


static ucc_mpool_ops_t ucc_tl_cuda_ipc_req_mpool_ops = {
    .chunk_alloc   = ucc_mpool_hugetlb_malloc,
    .chunk_release = ucc_mpool_hugetlb_free,
    .obj_init      = ucc_tl_cuda_ipc_req_mpool_obj_init,
    .obj_cleanup   = NULL
};

//TODO can we merge it with req mpool?
static ucc_mpool_ops_t ucc_tl_cuda_ipc_schedule_mpool_ops = {
    .chunk_alloc   = ucc_mpool_hugetlb_malloc,
    .chunk_release = ucc_mpool_hugetlb_free,
    .obj_init      = NULL,
    .obj_cleanup   = NULL
};

UCC_CLASS_INIT_FUNC(ucc_tl_cuda_ipc_context_t,
                    const ucc_base_context_params_t *params,
                    const ucc_base_config_t *config)
{
    ucc_tl_cuda_ipc_context_config_t *tl_cuda_ipc_config =
        ucc_derived_of(config, ucc_tl_cuda_ipc_context_config_t);
    ucc_status_t status;

    UCC_CLASS_CALL_SUPER_INIT(ucc_tl_context_t, tl_cuda_ipc_config->super.tl_lib,
                              params->context);
    memcpy(&self->cfg, tl_cuda_ipc_config, sizeof(*tl_cuda_ipc_config));

    status = ucc_mpool_init(&self->req_mp, 0, sizeof(ucc_tl_cuda_ipc_task_t), 0,
                            UCC_CACHE_LINE_SIZE, 8, UINT_MAX,
                            &ucc_tl_cuda_ipc_req_mpool_ops, params->thread_mode,
                            "tl_cuda_ipc_req_mp");

    if (status != UCC_OK) {
        tl_error(self->super.super.lib,
                 "failed to initialize tl_cuda_ipc_req mpool");
        return status;
    }
    status = ucc_mpool_init(&self->schedule_mp, 0,
                            sizeof(ucc_tl_cuda_ipc_schedule_t), 0,
                            UCC_CACHE_LINE_SIZE, 8, UINT_MAX,
                            &ucc_tl_cuda_ipc_schedule_mpool_ops,
                            params->thread_mode, "tl_cuda_ipc_schedule_mp");

    if (status != UCC_OK) {
        tl_error(self->super.super.lib,
                 "failed to initialize tl_cuda_ipc_schedule mpool");
        return status;
    }

    self->ipc_cache = kh_init(tl_cuda_ipc_ep_hash);
    tl_info(self->super.super.lib, "initialized tl context: %p", self);

    return UCC_OK;
}

UCC_CLASS_CLEANUP_FUNC(ucc_tl_cuda_ipc_context_t)
{
    tl_info(self->super.super.lib, "finalizing tl context: %p", self);
    ucc_mpool_cleanup(&self->req_mp, 1);
}

UCC_CLASS_DEFINE(ucc_tl_cuda_ipc_context_t, ucc_tl_context_t);

ucc_status_t
ucc_tl_cuda_ipc_get_context_attr(const ucc_base_context_t *context, /* NOLINT */
                             ucc_base_ctx_attr_t      *attr)
{
    if (attr->attr.mask & UCC_CONTEXT_ATTR_FIELD_CTX_ADDR_LEN) {
        attr->attr.ctx_addr_len = 0;
    }
    attr->topo_required = 0;
    return UCC_OK;
}
