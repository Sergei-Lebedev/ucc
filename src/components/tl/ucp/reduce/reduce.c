/**
 * Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */
#include "reduce.h"
#include "components/mc/ucc_mc.h"

ucc_base_coll_alg_info_t
    ucc_tl_ucp_reduce_algs[UCC_TL_UCP_REDUCE_ALG_LAST + 1] = {
        [UCC_TL_UCP_REDUCE_ALG_KNOMIAL] =
            {.id   = UCC_TL_UCP_REDUCE_ALG_KNOMIAL,
             .name = "knomial",
             .desc = "reduce over knomial tree with arbitrary radix "
                     "(optimized for latency)"},
        [UCC_TL_UCP_REDUCE_ALG_DBT] =
            {.id   = UCC_TL_UCP_REDUCE_ALG_DBT,
             .name = "dbt",
             .desc = "reduce over double binary tree where a leaf in one tree "
                     "will be intermediate in other (optimized for BW)"},
        [UCC_TL_UCP_REDUCE_ALG_LAST] = {
            .id = 0, .name = NULL, .desc = NULL}};

ucc_status_t ucc_tl_ucp_reduce_init(ucc_tl_ucp_task_t *task)
{
    return ucc_tl_ucp_reduce_knomial_init_common(task);
}
