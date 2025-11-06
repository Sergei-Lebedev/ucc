/**
 * Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "config.h"
#include "components/tl/ucp/tl_ucp.h"
#include "components/tl/ucp/tl_ucp_coll.h"
#include "core/ucc_progress_queue.h"
#include "components/tl/ucp/tl_ucp_sendrecv.h"
#include "ucp/api/device/ucp_host.h"
#include "coll_score/ucc_coll_score.h"
#include "ucc/api/ucc.h"
#include "utils/ucc_math.h"
#include "ucp_cuda.h"
#include <cuda_runtime.h>

#define UCC_TLCP_UCP_CUDA_SCORE 100

ucc_tl_coll_plugin_iface_t ucc_tlcp_ucp_cuda;

typedef struct ucc_tlcp_ucp_cuda_config {
    char *score_str;
} ucc_tlcp_ucp_cuda_config_t;

#define CONFIG(_lib) ((ucc_tlcp_ucp_cuda_config_t*)((_lib)->tlcp_configs[ucc_tlcp_ucp_cuda.id]))

static ucc_config_field_t ucc_tlcp_ucp_cuda_table[] = {
    {"TLCP_UCP_CUDA_TUNE", "", "Collective score modifier",
     ucc_offsetof(ucc_tlcp_ucp_cuda_config_t, score_str), UCC_CONFIG_TYPE_STRING},

    {NULL}};

static ucs_config_global_list_entry_t ucc_tlcp_ucp_cuda_cfg_entry =
{
    .name   = "TLCP_UCP_CUDA",
    .prefix = "TL_UCP_",
    .table  = ucc_tlcp_ucp_cuda_table,
    .size   = sizeof(ucc_tlcp_ucp_cuda_config_t)
};

// ucc_status_t ucc_tlcp_ucp_cuda_start(ucc_coll_task_t *coll_task)
// {
//     ucc_tl_ucp_task_t *task        = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
//     ucc_tl_ucp_team_t *team        = TASK_TEAM(task);
//     ucc_coll_args_t   *args        = &TASK_ARGS(task);
//     ucp_context_h      ucp_context = UCC_TL_UCP_TEAM_CTX(team)->worker.ucp_context;
//     ucp_mem_h          mem_h;
//     ucs_status_t       ucp_status;
//     ucc_status_t       ucc_status;
//     void              *rkey_buffer;
//     size_t             rkey_size;
//     ucc_base_coll_args_t  allgather_args;
//     void                 *all_rkeys;


//     tl_info(TASK_LIB(task), "starting tl_ucp_cuda coll task");

//     ucp_mem_map_params_t map_params;
//     map_params.field_mask = UCP_MEM_MAP_PARAM_FIELD_ADDRESS |
//                             UCP_MEM_MAP_PARAM_FIELD_LENGTH;
//     map_params.address    = args->src.info.buffer;
//     map_params.length     = args->src.info.count * ucc_dt_size(args->src.info.datatype);

//     ucp_status = ucp_mem_map(ucp_context, &map_params, &mem_h);
//     if (ucp_status != UCS_OK) {
//         tl_error(TASK_LIB(task), "ucp_mem_map failed with error code: %d", ucp_status);
//         return ucs_status_to_ucc_status(ucp_status);
//     }

//     ucp_status = ucp_rkey_pack(ucp_context, mem_h, &rkey_buffer, &rkey_size);
//     if (ucp_status != UCS_OK) {
//         tl_error(TASK_LIB(task), "ucp_rkey_pack failed with error code: %d", ucp_status);
//         return ucs_status_to_ucc_status(ucp_status);
//     }
//     all_rkeys = ucc_malloc(rkey_size * UCC_TL_TEAM_SIZE(team), "all_rkeys");
//     if (all_rkeys == NULL) {
//         tl_error(TASK_LIB(task), "failed to allocate all_rkeys");
//         return UCC_ERR_NO_MEMORY;
//     }
//     allgather_args.mask                   = 0;
//     allgather_args.args.mask              = 0;
//     allgather_args.args.coll_type         = UCC_COLL_TYPE_ALLGATHER;
//     allgather_args.args.src.info.buffer   = rkey_buffer;
//     allgather_args.args.src.info.count    = rkey_size;
//     allgather_args.args.src.info.datatype = UCC_DT_UINT8;
//     allgather_args.args.src.info.mem_type = UCC_MEMORY_TYPE_HOST;

//     allgather_args.args.dst.info.buffer   = all_rkeys;
//     allgather_args.args.dst.info.count    = rkey_size * UCC_TL_TEAM_SIZE(team);
//     allgather_args.args.dst.info.datatype = UCC_DT_UINT8;
//     allgather_args.args.dst.info.mem_type = UCC_MEMORY_TYPE_HOST;


//     ucc_status = ucc_tl_ucp_allgather_ring_init(&allgather_args, &team->super.super, &coll_task);
//     if (ucc_status != UCC_OK) {
//         tl_error(TASK_LIB(task), "ucc_tl_ucp_allgather_ring_init failed with error code: %d", ucc_status);
//         return ucc_status;
//     }

//     ucc_status = coll_task->post(coll_task);
//     if (ucc_status != UCC_OK) {
//         tl_error(TASK_LIB(task), "ucc_tl_ucp_allgather_ring_post failed with error code: %d", ucc_status);
//         return ucc_status;
//     }

//     while (coll_task->super.status == UCC_INPROGRESS) {
//         ucc_context_progress(UCC_TL_UCP_TEAM_CTX(team)->super.super.ucc_context);
//     }
//     if (coll_task->super.status != UCC_OK) {
//         tl_error(TASK_LIB(task), "ucc_tl_ucp_allgather_ring_post failed with error code: %d", coll_task->super.status);
//         return coll_task->super.status;
//     }
//     task->super.finalize(coll_task);


//     tl_ucp_cuda_alltoall_pairwise(&TASK_ARGS(task));
//     ucc_progress_queue_enqueue(UCC_TL_CORE_CTX(team)->pq, &task->super);

//     return UCC_OK;
// }

// void ucc_tlcp_ucp_cuda_progress(ucc_coll_task_t *coll_task)
// {
//     ucc_tl_ucp_task_t *task = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);

//     tl_info(TASK_LIB(task), "completing tl_ucp_cuda coll task");
//     ucc_assert(UCC_TL_UCP_TASK_P2P_COMPLETE(task));
//     task->super.status = UCC_OK;
// }

static inline int alltoall_onesided_handle_completion(
    ucc_tl_ucp_task_t *task, uint32_t *posted, uint32_t *completed,
    uint32_t nreqs, int64_t npolls)
{
    int64_t polls = 0;

    if ((*posted - *completed) >= nreqs) {
        while (polls < npolls) {
            ucp_worker_progress(TASK_CTX(task)->worker.ucp_worker);
            ++polls;
            if ((*posted - *completed) < nreqs) {
                break;
            }
        }
        if (polls >= npolls) {
            return 0; /* Return 0 to indicate should return */
        }
    }
    return 1; /* Return 1 to indicate should continue */
}

/* Common helper function to wait for all operations to complete */
static inline void alltoall_onesided_wait_completion(
    ucc_tl_ucp_task_t *task, int64_t npolls)
{
    int64_t polls = 0;

    if (!UCC_TL_UCP_TASK_ONESIDED_P2P_COMPLETE(task)) {
        while (polls++ < npolls) {
            ucp_worker_progress(TASK_CTX(task)->worker.ucp_worker);
            if (UCC_TL_UCP_TASK_ONESIDED_P2P_COMPLETE(task)) {
                task->super.status = UCC_OK;
                return;
            }
        }
        return;
    }
    task->super.status = UCC_OK;
}

ucc_status_t ucc_tlcp_ucp_cuda_alltoall_onesided_sched_start(
    ucc_coll_task_t *ctask)
{
    return ucc_schedule_start(ctask);
}

ucc_status_t ucc_tlcp_ucp_cuda_alltoall_onesided_sched_finalize(
    ucc_coll_task_t *ctask)
{
    ucc_schedule_t *schedule = ucc_derived_of(ctask, ucc_schedule_t);
    ucc_status_t    status;

    status = ucc_schedule_finalize(ctask);
    ucc_tl_ucp_put_schedule(schedule);
    return status;
}

// void ucc_tlcp_ucp_cuda_alltoall_onesided_put_progress(ucc_coll_task_t *ctask)
// {
//     ucc_tl_ucp_task_t *task      = ucc_derived_of(ctask, ucc_tl_ucp_task_t);
//     ucc_tl_ucp_team_t *team      = TASK_TEAM(task);
//     ptrdiff_t          src       = (ptrdiff_t)TASK_ARGS(task).src.info.buffer;
//     ptrdiff_t          dest      = (ptrdiff_t)TASK_ARGS(task).dst.info.buffer;
//     ucc_rank_t         grank     = UCC_TL_TEAM_RANK(team);
//     ucc_rank_t         gsize     = UCC_TL_TEAM_SIZE(team);
//     uint32_t           ntokens   = gsize;
//     int64_t            npolls    = task->n_polls;
//     ucc_mem_map_mem_h  src_memh  = TASK_ARGS(task).src_memh.local_memh;
//     ucc_mem_map_mem_h *dst_memh  = TASK_ARGS(task).dst_memh.global_memh;
//     uint32_t          *posted    = &task->onesided.put_posted;
//     uint32_t          *completed = &task->onesided.put_completed;
//     ucc_rank_t         peer      = (grank + *posted + 1) % gsize;
//     size_t             nelems;

//     nelems = TASK_ARGS(task).src.info.count;
//     nelems = (nelems / gsize) * ucc_dt_size(TASK_ARGS(task).src.info.datatype);

//     for (; *posted < gsize; peer = (peer + 1) % gsize) {
//         UCPCHECK_GOTO(
//             ucc_tl_ucp_put_nb(PTR_OFFSET(src, peer * nelems),
//                               PTR_OFFSET(dest, grank * nelems), nelems,
//                               peer, src_memh, dst_memh, team, task),
//             task, out);
//         UCPCHECK_GOTO(ucc_tl_ucp_ep_flush(peer, team, task), task, out);

//         if (!alltoall_onesided_handle_completion(task, posted, completed,
//                                                  ntokens, npolls)) {
//             return;
//         }
//     }

//     alltoall_onesided_wait_completion(task, npolls);
// out:
//     return;
// }

void ucc_tlcp_ucp_cuda_alltoall_onesided_put_progress(ucc_coll_task_t *ctask)
{
    cudaError_t cuda_error;

    cuda_error = cudaDeviceSynchronize();
    if (cuda_error != cudaSuccess) {
        tl_error(UCC_TASK_LIB(ctask), "cudaDeviceSynchronize() failed: %s", cudaGetErrorString(cuda_error));
        ctask->status = UCC_ERR_NO_MESSAGE;
        return;
    }
    ctask->status = UCC_OK;
}

ucc_status_t ucc_tlcp_ucp_cuda_alltoall_onesided_start(ucc_coll_task_t *ctask)
{
    ucc_tl_ucp_task_t *task = ucc_derived_of(ctask, ucc_tl_ucp_task_t);
    ucc_tl_ucp_team_t *team = TASK_TEAM(task);

    ucc_tl_ucp_task_reset(task, UCC_INPROGRESS);
    tl_ucp_cuda_alltoall_pairwise(task);
    return ucc_progress_queue_enqueue(UCC_TL_CORE_CTX(team)->pq, &task->super);
}

ucc_status_t ucc_tlcp_ucp_cuda_alltoall_onesided_finalize(
    ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t *task = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    alltoall_device_task_t
        *a2a_task_data      = (alltoall_device_task_t *)(task->plugin_data);
    ucc_tl_ucp_team_t *team = TASK_TEAM(task);
    ucc_status_t       status;
    ucc_rank_t         r;

    for (r = 0; r < UCC_TL_TEAM_SIZE(team); r++) {
        ucp_device_mem_list_release(a2a_task_data->mem_list_h[r]);
    }

    status = ucc_tl_ucp_coll_finalize(coll_task);
    if (ucc_unlikely(UCC_OK != status)) {
        tl_error(UCC_TASK_LIB(coll_task), "failed to finalize collective");
    }

    return status;
}

ucc_status_t ucc_tlcp_ucp_cuda_coll_init(ucc_base_coll_args_t *coll_args,
                                         ucc_base_team_t *team,
                                         ucc_coll_task_t **task_h)
{
    ucc_schedule_t         *schedule = NULL;
    ucc_tl_ucp_team_t      *tl_team  = ucc_derived_of(team, ucc_tl_ucp_team_t);
    ucc_tl_ucp_schedule_t  *tl_schedule       = NULL;
    ucc_base_coll_args_t    barrier_coll_args = {0};
    ucc_coll_args_t        *args              = &coll_args->args;
    int                     segment           = 0;
    ucc_coll_task_t        *barrier_task;
    ucc_tl_ucp_task_t      *a2a_task;
    alltoall_device_task_t *a2a_task_data;
    ucc_status_t            status;
    ucc_rank_t              r;
    ucp_ep_h                ep;
    ucp_device_mem_list_params_t mem_list_params;
    ucp_device_mem_list_elem_t   elem;
    uint64_t                     remote_addr;
    ucp_rkey_h                   rkey;
    ucp_mem_h                    src_mem_h;
    ucs_status_t                 ucs_status = UCS_OK;

    if (!(coll_args->args.mask & UCC_COLL_ARGS_FIELD_FLAGS) ||
        (coll_args->args.mask & UCC_COLL_ARGS_FIELD_FLAGS &&
         (!(coll_args->args.flags & UCC_COLL_ARGS_FLAG_MEM_MAPPED_BUFFERS)))) {
        tl_error(
            UCC_TL_TEAM_LIB(tl_team),
            "non memory mapped buffers are not supported");
        status = UCC_ERR_NOT_SUPPORTED;
        return status;
    }

    if (!(coll_args->args.mask & UCC_COLL_ARGS_FIELD_MEM_MAP_SRC_MEMH)) {
        coll_args->args.src_memh.global_memh = NULL;
    }

    if (!(coll_args->args.mask & UCC_COLL_ARGS_FIELD_MEM_MAP_DST_MEMH)) {
        coll_args->args.dst_memh.global_memh = NULL;
    } else {
        if (!(coll_args->args.flags & UCC_COLL_ARGS_FLAG_DST_MEMH_GLOBAL)) {
            tl_error(
                UCC_TL_TEAM_LIB(tl_team),
                "onesided alltoall requires global memory handles for dst "
                "buffers");
            status = UCC_ERR_INVALID_PARAM;
            return status;
        }
    }

    status = ucc_tl_ucp_get_schedule(
        tl_team, coll_args, (ucc_tl_ucp_schedule_t **)&tl_schedule);
    if (ucc_unlikely(UCC_OK != status)) {
        return status;
    }
    schedule = &tl_schedule->super.super;
    ucc_schedule_init(schedule, coll_args, team);

    /* initialize schedule */
    schedule->super.post     = ucc_tlcp_ucp_cuda_alltoall_onesided_sched_start;
    schedule->super.progress = NULL;
    schedule->super.finalize = ucc_tlcp_ucp_cuda_alltoall_onesided_sched_finalize;

    /* initialize alltoall task */
    a2a_task                 = ucc_tl_ucp_init_task(coll_args, team);
    a2a_task->super.finalize = ucc_tlcp_ucp_cuda_alltoall_onesided_finalize;
    a2a_task->super.progress = ucc_tlcp_ucp_cuda_alltoall_onesided_put_progress;
    a2a_task->super.post     = ucc_tlcp_ucp_cuda_alltoall_onesided_start;

    /* initialize barrier task */
    barrier_coll_args.mask           = 0;
    barrier_coll_args.args.mask      = 0;
    barrier_coll_args.args.coll_type = UCC_COLL_TYPE_BARRIER;
    barrier_coll_args.team           = team->params.team;
    status = ucc_tl_ucp_coll_init(&barrier_coll_args, team, &barrier_task);
    if (status != UCC_OK) {
        goto out;
    }

    /* add alltoall and barrier tasks to schedule */
    ucc_schedule_add_task(schedule, &a2a_task->super);
    ucc_task_subscribe_dep(
        &schedule->super, &a2a_task->super, UCC_EVENT_SCHEDULE_STARTED);
    ucc_schedule_add_task(schedule, barrier_task);
    ucc_task_subscribe_dep(&a2a_task->super, barrier_task, UCC_EVENT_COMPLETED);

    a2a_task_data = (alltoall_device_task_t *)(a2a_task->plugin_data);
    mem_list_params.field_mask = UCP_DEVICE_MEM_LIST_PARAMS_FIELD_ELEMENTS |
                                 UCP_DEVICE_MEM_LIST_PARAMS_FIELD_ELEMENT_SIZE |
                                 UCP_DEVICE_MEM_LIST_PARAMS_FIELD_NUM_ELEMENTS;
    mem_list_params.element_size = sizeof(ucp_device_mem_list_elem_t);
    mem_list_params.num_elements = 1;
    mem_list_params.elements     = &elem;

    for (r = 0; r < UCC_TL_TEAM_SIZE(tl_team); r++) {
        status = ucc_tl_ucp_get_ep(tl_team, r, &ep);
        if (ucc_unlikely(UCC_OK != status)) {
            return status;
        }

        status = ucc_tl_ucp_get_memh(
            tl_team, args->src_memh.local_memh, (void **)&src_mem_h);
        if (ucc_unlikely(UCC_OK != status)) {
            return status;
        }

        status = ucc_tl_ucp_resolve_p2p_by_va(
            tl_team,
            args->dst.info.buffer,
            &ep,
            r,
            &remote_addr,
            &rkey,
            &segment,
            args->dst_memh.global_memh);
        if (ucc_unlikely(UCC_OK != status)) {
            return status;
        }

        elem.field_mask = UCP_DEVICE_MEM_LIST_ELEM_FIELD_MEMH |
                          UCP_DEVICE_MEM_LIST_ELEM_FIELD_RKEY |
                          UCP_DEVICE_MEM_LIST_ELEM_FIELD_LOCAL_ADDR |
                          UCP_DEVICE_MEM_LIST_ELEM_FIELD_REMOTE_ADDR |
                          UCP_DEVICE_MEM_LIST_ELEM_FIELD_LENGTH;

        elem.length = args->src.info.count *
                      ucc_dt_size(args->src.info.datatype);
        elem.local_addr  = args->src.info.buffer;
        elem.remote_addr = remote_addr;
        elem.rkey        = rkey;
        elem.memh        = src_mem_h;

        ucs_status       = ucp_device_mem_list_create(
            ep, &mem_list_params, &a2a_task_data->mem_list_h[r]);
        while (ucs_status == UCS_ERR_NOT_CONNECTED) {
            ucp_worker_progress(
                UCC_TL_UCP_TEAM_CTX(tl_team)->worker.ucp_worker);
            ucs_status = ucp_device_mem_list_create(
                ep, &mem_list_params, &a2a_task_data->mem_list_h[r]);
        }
        if (ucc_unlikely(UCS_OK != ucs_status)) {
            ucc_error(
                "ucp_device_mem_list_create failed with error code: %d",
                ucs_status);
            return ucs_status_to_ucc_status(ucs_status);
        }
    }

    *task_h = &schedule->super;
    return status;

out:
    if (tl_schedule) {
        ucc_tl_ucp_put_schedule(&tl_schedule->super.super);
    }

    return status;
}

ucc_status_t ucc_tlcp_ucp_cuda_get_scores(ucc_base_team_t *tl_team,
                                          ucc_coll_score_t **score_p)
{
    ucc_tl_ucp_team_t *team = ucc_derived_of(tl_team, ucc_tl_ucp_team_t);
    ucc_tl_ucp_lib_t  *lib  = UCC_TL_UCP_TEAM_LIB(team);
    const char        *score_str;
    ucc_coll_score_t  *score;
    ucc_status_t       status;

    /* There can be a different logic for different coll_type/mem_type.
       Right now just init everything the same way. */
    status = ucc_coll_score_alloc(&score);
    if (UCC_OK != status) {
        tl_error(lib, "failed to alloc score");
        return status;
    }

    status = ucc_coll_score_add_range(score, UCC_COLL_TYPE_ALLTOALL,
                                      UCC_MEMORY_TYPE_CUDA,
                                      0, 1048576, UCC_TLCP_UCP_CUDA_SCORE,
                                      ucc_tlcp_ucp_cuda_coll_init, tl_team);
    if (UCC_OK != status) {
        tl_error(lib, "failed to add range");
        return status;
    }
    score_str = CONFIG(lib)->score_str;
    if (strlen(score_str) > 0) {
        return UCC_ERR_INVALID_PARAM;
    }
    *score_p = score;
    return status;
}

UCC_CONFIG_REGISTER_TABLE_ENTRY(&ucc_tlcp_ucp_cuda_cfg_entry,
                                &ucc_config_global_list);

ucc_tl_coll_plugin_iface_t ucc_tlcp_ucp_cuda = {
    .super.name   = "tl_ucp_cuda",
    .super.score  = UCC_TLCP_UCP_CUDA_SCORE,
    .config.table = ucc_tlcp_ucp_cuda_table,
    .config.size  = sizeof(ucc_tlcp_ucp_cuda_config_t),
    .get_scores   = ucc_tlcp_ucp_cuda_get_scores
};
