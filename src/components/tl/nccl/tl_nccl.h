/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_TL_NCCL_H_
#define UCC_TL_NCCL_H_

#include "components/tl/ucc_tl.h"
#include "components/tl/ucc_tl_log.h"
#include "utils/ucc_mpool.h"

#include <nccl.h>
#include <cuda.h>

#ifndef UCC_TL_NCCL_DEFAULT_SCORE
#define UCC_TL_NCCL_DEFAULT_SCORE 20
#endif

typedef struct ucc_tl_nccl_iface {
    ucc_tl_iface_t super;
} ucc_tl_nccl_iface_t;

extern ucc_tl_nccl_iface_t ucc_tl_nccl;

typedef struct ucc_tl_nccl_lib_config {
    ucc_tl_lib_config_t super;
    int                 pp_allreduce;
    uint32_t            n_frags;
    uint32_t            pipeline_depth;
    size_t              frag_thresh;
    size_t              frag_size;
} ucc_tl_nccl_lib_config_t;

typedef struct ucc_tl_nccl_context_config {
    ucc_tl_context_config_t super;
} ucc_tl_nccl_context_config_t;

typedef struct ucc_tl_nccl_lib {
    ucc_tl_lib_t super;
    ucc_tl_nccl_lib_config_t cfg;
} ucc_tl_nccl_lib_t;
UCC_CLASS_DECLARE(ucc_tl_nccl_lib_t, const ucc_base_lib_params_t *,
                  const ucc_base_config_t *);

typedef struct ucc_tl_nccl_context {
    ucc_tl_context_t             super;
    ucc_tl_nccl_context_config_t cfg;
    ucc_mpool_t                  req_mp;
} ucc_tl_nccl_context_t;
UCC_CLASS_DECLARE(ucc_tl_nccl_context_t, const ucc_base_context_params_t *,
                  const ucc_base_config_t *);

#define NUM_NCCL_COMMS 3
typedef struct ucc_tl_nccl_team {
    ucc_tl_team_t        super;
    ncclUniqueId        *unique_id;
    void                *oob_req;
    ucc_team_oob_coll_t  oob;
    ncclComm_t           nccl_comm;
    ucc_rank_t           rank;
    ucc_rank_t           size;
    ucc_rank_t           local_rank;
    ucc_rank_t           local_size;
    ucc_rank_t           num_nodes;
    ucc_rank_t           nodeid;
    cudaStream_t         stream;
//comms[0] intranode comm for reduce scatter
//comms[1] internode comm for allreduce
//comms[2] intranode comm for allgahter
    ncclComm_t           nccl_comms[NUM_NCCL_COMMS];
    cudaStream_t         streams[NUM_NCCL_COMMS];
} ucc_tl_nccl_team_t;

typedef struct ucc_tl_nccl_task {
    ucc_coll_task_t     super;
    ucc_tl_nccl_team_t *team;
    ucc_coll_args_t     args;
    cudaEvent_t         completed;
} ucc_tl_nccl_task_t;

#define UCC_TL_NCCL_SUPPORTED_COLLS                         \
    (UCC_COLL_TYPE_ALLTOALL  | UCC_COLL_TYPE_ALLTOALLV  |   \
     UCC_COLL_TYPE_ALLGATHER | UCC_COLL_TYPE_ALLGATHERV |   \
     UCC_COLL_TYPE_ALLREDUCE | UCC_COLL_TYPE_BCAST)

UCC_CLASS_DECLARE(ucc_tl_nccl_team_t, ucc_base_context_t *,
                  const ucc_base_team_params_t *);

#define NCCLCHECK_GOTO(_cmd, _label, _status, _lib)                            \
    do {                                                                       \
        ncclResult_t e = _cmd;                                                 \
        if (ncclSuccess != e) {                                                \
            tl_error(_lib, "NCCL error %d %s", e, ncclGetErrorString(e));      \
            _status = UCC_ERR_NO_MESSAGE;                                      \
            goto _label;                                                       \
        }                                                                      \
    } while (0)

#define CUDACHECK_GOTO(_cmd, _label, _status, _lib)                            \
    do {                                                                       \
        cudaError_t e = _cmd;                                                  \
        if (cudaSuccess != e) {                                                \
            tl_error(_lib, "CUDA error %d %s", e, cudaGetErrorName(e));        \
            _status = UCC_ERR_NO_MESSAGE;                                      \
            goto _label;                                                       \
        }                                                                      \
    } while (0)

#define UCC_TL_NCCL_TEAM_LIB(_team)                                            \
    (ucc_derived_of((_team)->super.super.context->lib, ucc_tl_nccl_lib_t))

#endif
