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
#include "tl_cuda_ipc_cache.h"
#include "tl_cuda_ipc_ep_hash.h"
#include <cuda.h>
#include <cuda_runtime.h>
#ifndef UCC_TL_CUDA_IPC_DEFAULT_SCORE
#define UCC_TL_CUDA_IPC_DEFAULT_SCORE 30
#endif

typedef struct ucc_tl_cuda_ipc_iface {
    ucc_tl_iface_t super;
} ucc_tl_cuda_ipc_iface_t;

extern ucc_tl_cuda_ipc_iface_t ucc_tl_cuda_ipc;

typedef struct ucc_tl_cuda_ipc_lib_config {
    ucc_tl_lib_config_t super;
    uint32_t            max_concurrent;
    uint32_t            num_rings;
} ucc_tl_cuda_ipc_lib_config_t;


typedef struct ucc_tl_cuda_ipc_context_config {
    ucc_tl_context_config_t            super;
} ucc_tl_cuda_ipc_context_config_t;

typedef struct ucc_tl_cuda_ipc_lib {
    ucc_tl_lib_t                 super;
    ucc_tl_cuda_ipc_lib_config_t cfg;
} ucc_tl_cuda_ipc_lib_t;
UCC_CLASS_DECLARE(ucc_tl_cuda_ipc_lib_t, const ucc_base_lib_params_t *,
                  const ucc_base_config_t *);

typedef struct {
    cudaIpcEventHandle_t ev_handle;
    size_t               displ;
} mem_info_data_t;

typedef struct {
    void              *d_ptr;
    size_t             size;
    uint32_t           seq_num[16];
    cudaIpcMemHandle_t handle;
    size_t             offset;
    mem_info_data_t    data[1];
} mem_info_t;

typedef struct ucc_tl_cuda_ipc_context {
    ucc_tl_context_t                 super;
    ucc_tl_cuda_ipc_context_config_t cfg;
    ucc_mpool_t                      req_mp;
    ucc_mpool_t                      schedule_mp;
    tl_cuda_ipc_ep_hash_t           *ipc_cache;
} ucc_tl_cuda_ipc_context_t;
UCC_CLASS_DECLARE(ucc_tl_cuda_ipc_context_t, const ucc_base_context_params_t *,
                  const ucc_base_config_t *);

typedef struct ucc_tl_cuda_ipc_team {
    ucc_tl_team_t        super;
    ucc_status_t         status;
    void                *oob_req;
    ucc_team_oob_coll_t  oob;
    ucc_rank_t           rank;
    ucc_rank_t           size;
    ucc_ep_map_t         map;
    cudaStream_t         stream;
    uint32_t             seq_num;
    mem_info_t          *mem_info;
    int                 *shm_ids;
} ucc_tl_cuda_ipc_team_t;

#define TASK_TEAM(_task)                                                       \
    (ucc_derived_of((_task)->super.team, ucc_tl_cuda_ipc_team_t))
#define TASK_CTX(_task)                                                        \
    (ucc_derived_of((_task)->super.team->context, ucc_tl_cuda_ipc_context_t))
#define TASK_LIB(_task)                                                        \
    (ucc_derived_of((_task)->super.team->context->lib, ucc_tl_cuda_ipc_lib_t))

#define UCC_TL_CUDA_IPC_SUPPORTED_COLLS             \
    (UCC_COLL_TYPE_ALLTOALLV)

UCC_CLASS_DECLARE(ucc_tl_cuda_ipc_team_t, ucc_base_context_t *,
                  const ucc_base_team_params_t *);

#define CUDACHECK_GOTO(_cmd, _label, _status, _lib)                            \
    do {                                                                       \
        cudaError_t e = _cmd;                                                  \
        if (cudaSuccess != e) {                                                \
            tl_error(_lib, "CUDA error %d %s", e, cudaGetErrorName(e));        \
            _status = UCC_ERR_NO_MESSAGE;                                      \
            goto _label;                                                       \
        }                                                                      \
    } while (0)

#define CUDACHECK(cmd) do {                                                    \
        cudaError_t e = cmd;                                                   \
        if(e != cudaSuccess) {                                                 \
            printf("cuda: %s failed with ret:%d(%s)\n", UCS_PP_MAKE_STRING(cmd), e,     \
                     cudaGetErrorString(e));                                   \
            return UCC_ERR_NO_MESSAGE;                                         \
        }                                                                      \
} while(0)

#define CUDACHECK_NO_RET(cmd) do {                                                    \
        cudaError_t e = cmd;                                                   \
        if(e != cudaSuccess) {                                                 \
            printf("cuda:%s failed with ret:%d(%s)\n", UCS_PP_MAKE_STRING(cmd), e,     \
                     cudaGetErrorString(e));                                   \
        }                                                                      \
} while(0)


#define UCC_TL_CUDA_IPC_TEAM_LIB(_team)                                            \
    (ucc_derived_of((_team)->super.super.context->lib, ucc_tl_cuda_ipc_lib_t))

#define UCC_TL_CUDA_IPC_TEAM_CTX(_team)                                             \
    (ucc_derived_of((_team)->super.super.context, ucc_tl_cuda_ipc_context_t))

#define GET_MEM_INFO(_team, _coll_id, _rank)                            \
    ({                                                                 \
        size_t _ctrl_size_rank = (sizeof(mem_info_t)  +                 \
                                  sizeof(mem_info_data_t) * ((_team)->size - 1)) ; \
        size_t _ctrl_size = _ctrl_size_rank * (_team)->size;                 \
        void *_mi = PTR_OFFSET((_team)->mem_info, _ctrl_size * (_coll_id)); \
        _mi  = PTR_OFFSET(_mi, _ctrl_size_rank * (_rank));              \
        (mem_info_t*)_mi;                                               \
    })

ucc_cuda_ipc_cache_t* ucc_cuda_ipc_get_cache(ucc_tl_cuda_ipc_team_t *team,
                                             ucc_rank_t rank);

#endif
