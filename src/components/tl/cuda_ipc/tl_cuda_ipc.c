/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "tl_cuda_ipc.h"
#include "core/ucc_team.h"
#include "components/mc/base/ucc_mc_base.h"

ucc_status_t ucc_tl_cuda_ipc_get_lib_attr(const ucc_base_lib_t *lib,
                                      ucc_base_lib_attr_t  *base_attr);

ucc_status_t ucc_tl_cuda_ipc_get_context_attr(const ucc_base_context_t *context,
                                          ucc_base_ctx_attr_t      *base_attr);

static ucc_config_field_t ucc_tl_cuda_ipc_lib_config_table[] = {
    {"", "", NULL, ucc_offsetof(ucc_tl_cuda_ipc_lib_config_t, super),
     UCC_CONFIG_TYPE_TABLE(ucc_tl_lib_config_table)},

    {"MAX_CONCURRENT", "8",
     "Maximum number of outstanding colls",
     ucc_offsetof(ucc_tl_cuda_ipc_lib_config_t, max_concurrent),
     UCC_CONFIG_TYPE_UINT},

    {NULL}};

static ucs_config_field_t ucc_tl_cuda_ipc_context_config_table[] = {
    {"", "", NULL, ucc_offsetof(ucc_tl_cuda_ipc_context_config_t, super),
     UCC_CONFIG_TYPE_TABLE(ucc_tl_context_config_table)},

    {NULL}};

UCC_CLASS_DEFINE_NEW_FUNC(ucc_tl_cuda_ipc_lib_t, ucc_base_lib_t,
                          const ucc_base_lib_params_t *,
                          const ucc_base_config_t *);

UCC_CLASS_DEFINE_DELETE_FUNC(ucc_tl_cuda_ipc_lib_t, ucc_base_lib_t);

UCC_CLASS_DEFINE_NEW_FUNC(ucc_tl_cuda_ipc_context_t, ucc_base_context_t,
                          const ucc_base_context_params_t *,
                          const ucc_base_config_t *);

UCC_CLASS_DEFINE_DELETE_FUNC(ucc_tl_cuda_ipc_context_t, ucc_base_context_t);

UCC_CLASS_DEFINE_NEW_FUNC(ucc_tl_cuda_ipc_team_t, ucc_base_team_t,
                          ucc_base_context_t *, const ucc_base_team_params_t *);

ucc_status_t ucc_tl_cuda_ipc_team_create_test(ucc_base_team_t *tl_team);

ucc_status_t ucc_tl_cuda_ipc_team_destroy(ucc_base_team_t *tl_team);

ucc_status_t ucc_tl_cuda_ipc_coll_init(ucc_base_coll_args_t *coll_args,
                                   ucc_base_team_t *team,
                                   ucc_coll_task_t **task);

ucc_status_t ucc_tl_cuda_ipc_team_get_scores(ucc_base_team_t   *tl_team,
                                         ucc_coll_score_t **score_p);
UCC_TL_IFACE_DECLARE(cuda_ipc, CUDA_IPC);


static inline ucc_context_addr_header_t *
ucc_tl_cuda_ipc_get_team_ep_header(ucc_tl_cuda_ipc_team_t *team, ucc_rank_t rank)

{
    return ucc_get_team_ep_header(UCC_TL_CORE_CTX(team), team->super.super.team,
                                  ucc_ep_map_eval(team->map, rank));
}

ucc_cuda_ipc_cache_t* ucc_cuda_ipc_get_cache(ucc_tl_cuda_ipc_team_t *team,
                                             ucc_rank_t rank)
{
    ucc_tl_cuda_ipc_context_t      *ctx      = UCC_TL_CUDA_IPC_TEAM_CTX(team);
    ucc_context_addr_header_t *h        = NULL;
    ucc_cuda_ipc_cache_t *cache;
    ucc_status_t          status;

    h = ucc_tl_cuda_ipc_get_team_ep_header(team, rank);
    cache = tl_cuda_ipc_hash_get(ctx->ipc_cache, h->ctx_id);
    if (ucc_unlikely(NULL == cache)) {
        status =  ucc_cuda_ipc_create_cache(&cache, "ipc-cache");
        if (status != UCC_OK) {
            tl_error(UCC_TL_TEAM_LIB(team), "failed to create ipc cache");
            return NULL;
        }
        tl_cuda_ipc_hash_put(ctx->ipc_cache, h->ctx_id, cache);
    }
    return cache;
}
