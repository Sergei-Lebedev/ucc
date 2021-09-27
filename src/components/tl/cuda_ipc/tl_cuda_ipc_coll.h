/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_TL_CUDA_IPC_COLL_H_
#define UCC_TL_CUDA_IPC_COLL_H_

#include "tl_cuda_ipc.h"
#include "core/ucc_mc.h"

#define UCC_TL_CUDA_IPC_N_DEFAULT_ALG_SELECT_STR 2
extern const char
    *ucc_tl_cuda_ipc_default_alg_select_str[UCC_TL_CUDA_IPC_N_DEFAULT_ALG_SELECT_STR];

enum {
    UCC_TL_CUDA_IPC_REDUCE_SCATTER_ALG_LINEAR,
    UCC_TL_CUDA_IPC_REDUCE_SCATTER_ALG_RING,
    UCC_TL_CUDA_IPC_REDUCE_SCATTER_ALG_LAST
};

enum {
    UCC_TL_CUDA_IPC_ALLGATHER_ALG_LINEAR,
    UCC_TL_CUDA_IPC_ALLGATHER_ALG_RING,
    UCC_TL_CUDA_IPC_ALLGATHER_ALG_LAST
};

#define MAX_STATIC_SIZE 16

typedef struct ucc_tl_cuda_ipc_task {
    ucc_coll_task_t      super;
    uint32_t             seq_num;
    uint32_t             early_posted;
    cudaStream_t         stream;
    cudaEvent_t          event;
    void                *data[MAX_STATIC_SIZE];
    union {
        struct {
            void                    *info;
            void                   **peer_map_addr;
            uint32_t                 coll_id;
            uint32_t                 n;
            ucc_ee_executor_task_t *exec_task[MAX_STATIC_SIZE];
        } alltoallv;
        struct {
            void                   *scratch;
            ucc_mc_buffer_header_t *scratch_mc_header;
            uint32_t                step;
            uint32_t                coll_id;
            int                     ring_id;
            int                     n_rings;
            void                   *peer_map_addr;
            ucc_ee_executor_task_t *exec_task;
        } reduce_scatter;
        struct {
            void                   **peer_map_addr;
            uint32_t                coll_id;
            ucc_ee_executor_task_t *exec_task[MAX_STATIC_SIZE];
        } reduce_scatter_linear;
        struct {
            // void                   *scratch;
            // ucc_mc_buffer_header_t *scratch_mc_header;
            uint32_t                step;
            uint32_t                coll_id;
            int                     ring_id;
            int                     n_rings;
            void                   *peer_map_addr;
            ucc_ee_executor_task_t *exec_task;
        } allgather;
        struct {
            void                   **peer_map_addr;
            uint32_t                coll_id;
            ucc_ee_executor_task_t *exec_task[MAX_STATIC_SIZE];
        } allgather_linear;
    };
} ucc_tl_cuda_ipc_task_t;

typedef struct ucc_tl_cuda_ipc_schedule {
    ucc_schedule_t super;
    int            eee_external;
    union {
        struct {
            ucc_ee_executor_t *eee;
        } reduce_scatter;
        struct {
            ucc_ee_executor_t *eee;
        } reduce_scatter_linear;
        struct {
            ucc_ee_executor_t *eee;
        } allgather;
        struct {
            ucc_ee_executor_t *eee;
        } allgather_linear;
    };
} ucc_tl_cuda_ipc_schedule_t;

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

    UCC_TL_CUDA_IPC_PROFILE_REQUEST_NEW(task, "tl_cuda_ipc_task", 0);
    task->super.super.status = UCC_OPERATION_INITIALIZED;
    task->super.flags        = 0;
    task->super.team         = &team->super.super;
    ucc_tl_cuda_ipc_task_reset(task);
    return task;
}

static inline
ucc_tl_cuda_ipc_schedule_t *ucc_tl_cuda_ipc_get_schedule(ucc_coll_args_t *args,
                                             ucc_tl_cuda_ipc_team_t *team)
{
    ucc_tl_cuda_ipc_context_t  *ctx      = UCC_TL_CUDA_IPC_TEAM_CTX(team);
    ucc_tl_cuda_ipc_schedule_t *schedule = ucc_mpool_get(&ctx->schedule_mp);

    ucc_schedule_init(&schedule->super, args, &team->super.super);
    return schedule;
}

static inline void ucc_tl_cuda_ipc_put_schedule(ucc_tl_cuda_ipc_schedule_t *schedule)
{
    ucc_mpool_put(schedule);
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
    UCC_TL_CUDA_IPC_PROFILE_REQUEST_FREE(task);
    ucc_mpool_put(task);
}

void ucc_tl_cuda_ipc_get_alloc_info(void *ptr, size_t length,
                                    void **base_address, size_t *alloc_length);

ucc_status_t ucc_tl_cuda_ipc_coll_init(ucc_base_coll_args_t *coll_args,
                                       ucc_base_team_t *team,
                                       ucc_coll_task_t **task_h);

ucc_status_t ucc_tl_cuda_ipc_alltoallv_init(ucc_base_coll_args_t *coll_args,
                                            ucc_base_team_t *team,
                                            ucc_coll_task_t **task);

ucc_status_t ucc_tl_cuda_ipc_reduce_scatter_ring_init(ucc_base_coll_args_t *coll_args,
                                                      ucc_base_team_t *team,
                                                      ucc_coll_task_t **task);

ucc_status_t ucc_tl_cuda_ipc_reduce_scatter_linear_init(ucc_base_coll_args_t *coll_args,
                                                        ucc_base_team_t *team,
                                                        ucc_coll_task_t **task);

ucc_status_t ucc_tl_cuda_ipc_allgather_ring_init(ucc_base_coll_args_t *coll_args,
                                                 ucc_base_team_t *team,
                                                 ucc_coll_task_t **task);

ucc_status_t ucc_tl_cuda_ipc_allgather_linear_init(ucc_base_coll_args_t *coll_args,
                                                   ucc_base_team_t *tl_team,
                                                   ucc_coll_task_t **task_p);

ucc_status_t ucc_tl_cuda_ipc_alg_id_to_init(int alg_id, const char *alg_id_str,
                                            ucc_coll_type_t   coll_type,
                                            ucc_memory_type_t mem_type,
                                            ucc_base_coll_init_fn_t *init);

#endif
