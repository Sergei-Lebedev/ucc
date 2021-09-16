/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "tl_cuda_ipc.h"
#include "tl_cuda_ipc_coll.h"
#include "core/ucc_mc.h"
#include "core/ucc_ee.h"
#include "core/ucc_team.h"
#include "coll_score/ucc_coll_score.h"
#include <sys/shm.h>

UCC_CLASS_INIT_FUNC(ucc_tl_cuda_ipc_team_t, ucc_base_context_t *tl_context,
                    const ucc_base_team_params_t *params)
{
    ucc_tl_cuda_ipc_context_t *ctx    =
        ucc_derived_of(tl_context, ucc_tl_cuda_ipc_context_t);
    ucc_tl_cuda_ipc_lib_t *lib =
        ucc_derived_of(tl_context->lib, ucc_tl_cuda_ipc_lib_t);
    ucc_status_t               status;

    UCC_CLASS_CALL_SUPER_INIT(ucc_tl_team_t, &ctx->super, params->team);

    if (!ucc_all_ranks_on_local_node(params->team, params->map)) {
        tl_debug(tl_context->lib,
                 "can't create cuda_ipc team for multi-node team");
        return UCC_ERR_INVALID_PARAM;
    }
    self->oob       = params->params.oob;
    self->size      = self->oob.n_oob_eps;
    self->rank      = params->rank;
    self->map       = params->map;
    self->status    = UCC_INPROGRESS;
    self->seq_num   = 1;

    int shm_id = 0;
    self->shm_ids =  ucc_malloc(self->size*sizeof(int), "shm_ids");
    if (!self->shm_ids) {
        tl_error(tl_context->lib, "failed to alloc shmids");
        return UCC_ERR_NO_MEMORY;
    }

    size_t ctrl_size = lib->cfg.max_concurrent * (sizeof(mem_info_t)  +
        sizeof(mem_info_data_t) * (self->size - 1)) * self->size;
    if (self->rank == 0) {
        shm_id = shmget(IPC_PRIVATE, ctrl_size, IPC_CREAT | 0666);
        if (shm_id < 0) {
            tl_error(tl_context->lib, "Failed to shmget with IPC_PRIVATE, "
                     "size %zd, IPC_CREAT; errno %d:%s", ctrl_size,
                     errno, strerror(errno));
        } else {
            self->mem_info = shmat(shm_id, NULL, 0);
            shmctl(shm_id, IPC_RMID, NULL);
            if (self->mem_info == (void *) -1) {
                tl_error(tl_context->lib, "Failed to shmat errno:%d(%s)", errno, strerror(errno));
                shm_id = -1;
            } else {
                memset(self->mem_info, 0, ctrl_size);
            }
        }
    }
    status = self->oob.allgather(&shm_id, self->shm_ids, sizeof(int),
                                 self->oob.coll_info, &self->oob_req);
    if (UCC_OK != status) {
        tl_error(tl_context->lib, "failed to start oob allgather");
        ucc_free(self->shm_ids);
    }
    tl_info(tl_context->lib, "posted tl team: %p", self);
    return status;
}

UCC_CLASS_CLEANUP_FUNC(ucc_tl_cuda_ipc_team_t)
{
    tl_info(self->super.super.context->lib, "finalizing tl team: %p", self);
}

UCC_CLASS_DEFINE_DELETE_FUNC(ucc_tl_cuda_ipc_team_t, ucc_base_team_t);
UCC_CLASS_DEFINE(ucc_tl_cuda_ipc_team_t, ucc_tl_team_t);

ucc_status_t ucc_tl_cuda_ipc_team_destroy(ucc_base_team_t *tl_team)
{
    UCC_CLASS_DELETE_FUNC_NAME(ucc_tl_cuda_ipc_team_t)(tl_team);
    return UCC_OK;
}

ucc_status_t ucc_tl_cuda_ipc_team_create_test(ucc_base_team_t *tl_team)
{
    ucc_tl_cuda_ipc_team_t *team           =
        ucc_derived_of(tl_team, ucc_tl_cuda_ipc_team_t);
    ucc_status_t            status;
    int                     shm_id;

    if (team->status == UCC_OK) {
        return UCC_OK;
    }

    status = team->oob.req_test(team->oob_req);
    if (UCC_INPROGRESS == status) {
        return UCC_INPROGRESS;
    }
    if (status < 0) {
        tl_error(tl_team->context->lib, "oob allgather failed");
        return status;
    }
    team->oob.req_free(team->oob_req);

    shm_id = team->shm_ids[0];
    if (-1 == shm_id) {
        tl_error(tl_team->context->lib, "got error shm_id");
        return UCC_ERR_NO_MESSAGE;
    }
    ucc_free(team->shm_ids);
    if (team->rank != 0) {
        team->mem_info = shmat(shm_id, NULL, 0);
        if (team->mem_info == (void *) -1) {
            tl_error(tl_team->context->lib, "failed to shamt errno: %d (%s)",
                     errno, strerror(errno));
            return UCC_ERR_NO_MEMORY;
        }
    }
    CUDACHECK_GOTO(cudaStreamCreateWithFlags(&team->stream,
                   cudaStreamNonBlocking), err, status,
                   tl_team->context->lib);

    tl_info(tl_team->context->lib, "initialized tl team: %p", team);
    return UCC_OK;

err:
    return status;
}


ucc_status_t ucc_tl_cuda_ipc_team_get_scores(ucc_base_team_t   *tl_team,
                                         ucc_coll_score_t **score_p)
{
    ucc_tl_cuda_ipc_team_t *team = ucc_derived_of(tl_team,
                                                  ucc_tl_cuda_ipc_team_t);
    ucc_base_lib_t         *lib  = UCC_TL_TEAM_LIB(team);
    ucc_coll_score_t       *score;
    ucc_status_t            status;

    status = ucc_coll_score_alloc(&score);
    if (UCC_OK != status) {
        tl_error(lib, "faild to alloc score_t");
        return status;
    }

    status = ucc_coll_score_add_range(score, UCC_COLL_TYPE_ALLTOALLV,
                                      UCC_MEMORY_TYPE_CUDA, 0, UCC_MSG_MAX,
                                      UCC_TL_CUDA_IPC_DEFAULT_SCORE,
                                      ucc_tl_cuda_ipc_alltoallv_init, tl_team);
    if (UCC_OK != status) {
        tl_error(lib, "faild to add alltoallv range to score_t");
        return status;
    }

    status = ucc_coll_score_add_range(score, UCC_COLL_TYPE_REDUCE_SCATTER,
                                      UCC_MEMORY_TYPE_CUDA, 0, UCC_MSG_MAX,
                                      UCC_TL_CUDA_IPC_DEFAULT_SCORE,
                                      ucc_tl_cuda_ipc_reduce_scatter_init,
                                      tl_team);
    if (UCC_OK != status) {
        tl_error(lib, "faild to add reduce scatter range to score_t");
        return status;
    }
#if 1
    status = ucc_coll_score_add_range(score, UCC_COLL_TYPE_ALLGATHER,
                                      UCC_MEMORY_TYPE_CUDA, 0, UCC_MSG_MAX,
                                      UCC_TL_CUDA_IPC_DEFAULT_SCORE,
                                      ucc_tl_cuda_ipc_allgather_init,
                                      tl_team);
    if (UCC_OK != status) {
        tl_error(lib, "faild to add allgather range to score_t");
        return status;
    }
#endif
    *score_p = score;
    return UCC_OK;
}
