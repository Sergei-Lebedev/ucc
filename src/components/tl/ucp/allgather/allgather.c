/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */
#include "config.h"
#include "tl_ucp.h"
#include "allgather.h"

ucc_base_coll_alg_info_t
ucc_tl_ucp_allgather_algs[UCC_TL_UCP_ALLGATHER_ALG_LAST + 1] = {
            [UCC_TL_UCP_ALLGATHER_ALG_RING] =
            {.id   = UCC_TL_UCP_ALLGATHER_ALG_RING,
             .name = "ring",
             .desc = "O(N) ring algorithm"},
                    [UCC_TL_UCP_ALLGATHER_ALG_KNOMIAL] =
            {.id   = UCC_TL_UCP_ALLGATHER_ALG_KNOMIAL,
             .name = "knomial",
             .desc = "recursive k-ing with arbitrary radix "},
            [UCC_TL_UCP_ALLGATHER_ALG_LAST] = {
                .id = 0, .name = NULL, .desc = NULL
            }
};

ucc_status_t ucc_tl_ucp_allgather_ring_start(ucc_coll_task_t *task);
ucc_status_t ucc_tl_ucp_allgather_ring_progress(ucc_coll_task_t *task);

ucc_status_t ucc_tl_ucp_allgather_init(ucc_tl_ucp_task_t *task)
{
    if ((task->super.args.src.info.datatype == UCC_DT_USERDEFINED) ||
        (task->super.args.dst.info.datatype == UCC_DT_USERDEFINED)) {
        tl_error(UCC_TASK_LIB(task), "user defined datatype is not supported");
        return UCC_ERR_NOT_SUPPORTED;
    }
    task->super.post     = ucc_tl_ucp_allgather_ring_start;
    task->super.progress = ucc_tl_ucp_allgather_ring_progress;
    return UCC_OK;
}

ucc_status_t ucc_tl_ucp_allgather_ring_init(ucc_base_coll_args_t *coll_args,
                                            ucc_base_team_t *     team,
                                            ucc_coll_task_t **    task_h)
{
    ucc_tl_ucp_team_t *tl_team = ucc_derived_of(team, ucc_tl_ucp_team_t);
    ucc_tl_ucp_task_t *task;

    if ((coll_args->args.src.info.datatype == UCC_DT_USERDEFINED) ||
        (coll_args->args.dst.info.datatype == UCC_DT_USERDEFINED)) {
        tl_error(UCC_TL_TEAM_LIB(tl_team),
                 "user defined datatype is not supported");
        return UCC_ERR_NOT_SUPPORTED;

    }
    task                 = ucc_tl_ucp_init_task(coll_args, team);
    task->super.post     = ucc_tl_ucp_allgather_ring_start;
    task->super.progress = ucc_tl_ucp_allgather_ring_progress;
    *task_h              = &task->super;
    return UCC_OK;

}
