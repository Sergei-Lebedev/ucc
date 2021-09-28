/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "tl_cuda_ipc_coll.h"
#include "core/ucc_mc.h"
#include "core/ucc_ee.h"
#include "utils/ucc_math.h"
#include "utils/ucc_coll_utils.h"

#define UCC_TL_CUDA_IPC_REDUCE_SCATTER_DEFAULT_ALG_SELECT_STR                  \
    "reduce_scatter:0-512k:cuda:@0#reduce_scatter:512k-inf:cuda:@1"

#define UCC_TL_CUDA_IPC_ALLGATHER_DEFAULT_ALG_SELECT_STR                       \
    "allgather:0-512k:cuda:@0#allgather:512k-inf:cuda:@1"

const char
    *ucc_tl_cuda_ipc_default_alg_select_str[UCC_TL_CUDA_IPC_N_DEFAULT_ALG_SELECT_STR] = {
    UCC_TL_CUDA_IPC_REDUCE_SCATTER_DEFAULT_ALG_SELECT_STR,
    UCC_TL_CUDA_IPC_ALLGATHER_DEFAULT_ALG_SELECT_STR};

ucc_status_t ucc_tl_cuda_ipc_alg_id_to_init(int alg_id, const char *alg_id_str,
                                            ucc_coll_type_t   coll_type,
                                            ucc_memory_type_t mem_type, //NOLINT
                                            ucc_base_coll_init_fn_t *init)
{
    ucc_status_t status = UCC_OK;
    switch (coll_type) {
    case UCC_COLL_TYPE_REDUCE_SCATTER:
        switch (alg_id) {
        case UCC_TL_CUDA_IPC_REDUCE_SCATTER_ALG_LINEAR:
            *init = ucc_tl_cuda_ipc_reduce_scatter_linear_init;
            break;
        case UCC_TL_CUDA_IPC_REDUCE_SCATTER_ALG_RING:
            *init = ucc_tl_cuda_ipc_reduce_scatter_ring_init;
            break;
        default:
            status = UCC_ERR_INVALID_PARAM;
            break;

        };
        break;
    case UCC_COLL_TYPE_ALLGATHER:
        switch (alg_id) {
        case UCC_TL_CUDA_IPC_ALLGATHER_ALG_LINEAR:
            *init = ucc_tl_cuda_ipc_allgather_linear_init;
            break;
        case UCC_TL_CUDA_IPC_ALLGATHER_ALG_RING:
            *init = ucc_tl_cuda_ipc_allgather_ring_init;
            break;
        default:
            status = UCC_ERR_INVALID_PARAM;
            break;
        };
        break;
    default:
        status = UCC_ERR_NOT_SUPPORTED;
        break;
    }
    return status;
}
