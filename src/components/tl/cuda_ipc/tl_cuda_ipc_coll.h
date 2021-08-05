/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_TL_NCCL_COLL_H_
#define UCC_TL_NCCL_COLL_H_

#include "tl_cuda_ipc.h"

#define MAX_STATIC_SIZE 16

typedef struct ucc_tl_cuda_ipc_task {
    ucc_coll_task_t      super;
    uint32_t             seq_num;
    uint32_t             early_posted;
    cudaStream_t         stream;
    cudaEvent_t          event;
    void                *data[MAX_STATIC_SIZE];
    struct {
        void                   *info;
        void                  **peer_map_addr;
        uint32_t                coll_id;
        uint32_t                n;
    } alltoallv;
} ucc_tl_cuda_ipc_task_t;

static inline void ucc_tl_cuda_ipc_task_reset(ucc_tl_cuda_ipc_task_t *task)
{
    task->super.super.status = UCC_INPROGRESS;
    task->early_posted       = 0;
}

static inline
ucc_tl_cuda_ipc_task_t *ucc_tl_cuda_ipc_get_task(ucc_tl_cuda_ipc_team_t *team)
{
    ucc_tl_cuda_ipc_context_t *ctx  = UCC_TL_CUDA_IPC_TEAM_CTX(team);
    ucc_tl_cuda_ipc_task_t    *task = ucc_mpool_get(&ctx->req_mp);;

    task->super.super.status = UCC_OPERATION_INITIALIZED;
    task->super.flags        = 0;
    task->super.team         = &team->super.super;
    ucc_tl_cuda_ipc_task_reset(task);
    return task;
}

static inline ucc_tl_cuda_ipc_task_t *
ucc_tl_cuda_ipc_init_task(ucc_base_coll_args_t *coll_args, ucc_base_team_t *team)
{
    ucc_tl_cuda_ipc_team_t *tl_team = ucc_derived_of(team,
                                                     ucc_tl_cuda_ipc_team_t);
    ucc_tl_cuda_ipc_task_t *task    = ucc_tl_cuda_ipc_get_task(tl_team);

    ucc_coll_task_init(&task->super, &coll_args->args, team);
    task->seq_num      = tl_team->seq_num++;
    task->early_posted = 0;
    return task;
}

static inline void ucc_tl_cuda_ipc_put_task(ucc_tl_cuda_ipc_task_t *task)
{
    ucc_mpool_put(task);
}

ucc_status_t ucc_tl_cuda_ipc_coll_init(ucc_base_coll_args_t *coll_args,
                                       ucc_base_team_t *team,
                                       ucc_coll_task_t **task_h);

ucc_status_t ucc_tl_cuda_ipc_alltoallv_init(ucc_base_coll_args_t *coll_args,
                                            ucc_base_team_t *team,
                                            ucc_coll_task_t **task);

#endif
