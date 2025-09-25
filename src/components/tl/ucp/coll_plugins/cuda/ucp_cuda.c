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
#include "coll_patterns/recursive_knomial.h"
#include "coll_score/ucc_coll_score.h"
#include "ucc/api/ucc.h"
#include "ucp/api/ucp_compat.h"
#include "utils/ucc_math.h"
#include "components/tl/ucp/allgather/allgather.h"

#define UCC_TLCP_UCP_CUDA_SCORE 100

ucc_tl_coll_plugin_iface_t ucc_tlcp_ucp_cuda;

ucc_status_t tl_ucp_cuda_alltoall_pairwise(ucc_coll_args_t *args);

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

ucc_status_t ucc_tlcp_ucp_cuda_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t *task        = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_tl_ucp_team_t *team        = TASK_TEAM(task);
    ucc_coll_args_t   *args        = &TASK_ARGS(task);
    ucp_context_h      ucp_context = UCC_TL_UCP_TEAM_CTX(team)->worker.ucp_context;
    ucp_mem_h          mem_h;
    ucs_status_t       ucp_status;
    ucc_status_t       ucc_status;
    void              *rkey_buffer;
    size_t             rkey_size;
    ucc_base_coll_args_t  allgather_args;
    void                 *all_rkeys;


    tl_info(TASK_LIB(task), "starting tl_ucp_cuda coll task");

    ucp_mem_map_params_t map_params;
    map_params.field_mask = UCP_MEM_MAP_PARAM_FIELD_ADDRESS |
                            UCP_MEM_MAP_PARAM_FIELD_LENGTH;
    map_params.address    = args->src.info.buffer;
    map_params.length     = args->src.info.count * ucc_dt_size(args->src.info.datatype);

    ucp_status = ucp_mem_map(ucp_context, &map_params, &mem_h);
    if (ucp_status != UCS_OK) {
        tl_error(TASK_LIB(task), "ucp_mem_map failed with error code: %d", ucp_status);
        return ucs_status_to_ucc_status(ucp_status);
    }

    ucp_status = ucp_rkey_pack(ucp_context, mem_h, &rkey_buffer, &rkey_size);
    if (ucp_status != UCS_OK) {
        tl_error(TASK_LIB(task), "ucp_rkey_pack failed with error code: %d", ucp_status);
        return ucs_status_to_ucc_status(ucp_status);
    }
    all_rkeys = ucc_malloc(rkey_size * UCC_TL_TEAM_SIZE(team), "all_rkeys");
    if (all_rkeys == NULL) {
        tl_error(TASK_LIB(task), "failed to allocate all_rkeys");
        return UCC_ERR_NO_MEMORY;
    }
    allgather_args.mask                   = 0;
    allgather_args.args.mask              = 0;
    allgather_args.args.coll_type   = UCC_COLL_TYPE_ALLGATHER;
    allgather_args.args.src.info.buffer   = rkey_buffer;
    allgather_args.args.src.info.count    = rkey_size;
    allgather_args.args.src.info.datatype = UCC_DT_UINT8;
    allgather_args.args.src.info.mem_type = UCC_MEMORY_TYPE_HOST;

    allgather_args.args.dst.info.buffer   = all_rkeys;
    allgather_args.args.dst.info.count    = rkey_size * UCC_TL_TEAM_SIZE(team);
    allgather_args.args.dst.info.datatype = UCC_DT_UINT8;
    allgather_args.args.dst.info.mem_type = UCC_MEMORY_TYPE_HOST;


    ucc_status = ucc_tl_ucp_allgather_ring_init(&allgather_args, &team->super.super, &coll_task);
    if (ucc_status != UCC_OK) {
        tl_error(TASK_LIB(task), "ucc_tl_ucp_allgather_ring_init failed with error code: %d", ucc_status);
        return ucc_status;
    }

    ucc_status = coll_task->post(coll_task);
    if (ucc_status != UCC_OK) {
        tl_error(TASK_LIB(task), "ucc_tl_ucp_allgather_ring_post failed with error code: %d", ucc_status);
        return ucc_status;
    }

    while (coll_task->super.status == UCC_INPROGRESS) {
        ucc_context_progress(UCC_TL_UCP_TEAM_CTX(team)->super.super.ucc_context);
    }
    if (coll_task->super.status != UCC_OK) {
        tl_error(TASK_LIB(task), "ucc_tl_ucp_allgather_ring_post failed with error code: %d", coll_task->super.status);
        return coll_task->super.status;
    }
    task->super.finalize(coll_task);


    tl_ucp_cuda_alltoall_pairwise(&TASK_ARGS(task));
    ucc_progress_queue_enqueue(UCC_TL_CORE_CTX(team)->pq, &task->super);

    return UCC_OK;
}

void ucc_tlcp_ucp_cuda_progress(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t *task = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);

    tl_info(TASK_LIB(task), "completing tl_ucp_cuda coll task");
    ucc_assert(UCC_TL_UCP_TASK_P2P_COMPLETE(task));
    task->super.status = UCC_OK;
}

ucc_status_t ucc_tlcp_ucp_cuda_coll_init(ucc_base_coll_args_t *coll_args,
                                         ucc_base_team_t *team,
                                         ucc_coll_task_t **task_h)
{
    ucc_tl_ucp_team_t    *tl_team = ucc_derived_of(team, ucc_tl_ucp_team_t);
    ucc_tl_ucp_task_t    *task    = ucc_tl_ucp_get_task(tl_team);

    task = ucc_tl_ucp_init_task(coll_args, team);
    task->tagged.tag     = tl_team->seq_num;
    tl_team->seq_num     = (tl_team->seq_num + 1) % UCC_TL_UCP_MAX_COLL_TAG;
    task->super.finalize = ucc_tl_ucp_coll_finalize;
    task->super.post     = ucc_tlcp_ucp_cuda_start;
    task->super.progress = ucc_tlcp_ucp_cuda_progress;
    *task_h              = &task->super;
    return UCC_OK;
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