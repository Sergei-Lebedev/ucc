/**
 * Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "config.h"
#include "tl_ucp.h"
#include "alltoallv.h"
#include "core/ucc_progress_queue.h"
#include "utils/ucc_math.h"
#include "utils/ucc_coll_utils.h"
#include "tl_ucp_sendrecv.h"

static inline ucc_rank_t get_recv_peer(ucc_rank_t rank, ucc_rank_t size,
                                       ucc_rank_t step)
{
    return (rank + step + 1) % size;
}

static inline ucc_rank_t get_send_peer(ucc_rank_t rank, ucc_rank_t size,
                                       ucc_rank_t step)
{
    return (rank - step + size - 1) % size;
}

static void ucc_tl_ucp_alltoallv_pairwise_progress(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t *task  = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_tl_ucp_team_t *team  = TASK_TEAM(task);
    ptrdiff_t          sbuf  = (ptrdiff_t)TASK_ARGS(task).src.info_v.buffer;
    ptrdiff_t          rbuf  = (ptrdiff_t)TASK_ARGS(task).dst.info_v.buffer;
    ucc_memory_type_t  smem  = TASK_ARGS(task).src.info_v.mem_type;
    ucc_memory_type_t  rmem  = TASK_ARGS(task).dst.info_v.mem_type;
    ucc_rank_t         grank = UCC_TL_TEAM_RANK(team);
    ucc_rank_t         gsize = UCC_TL_TEAM_SIZE(team);
    int                polls = 0;
    ucc_rank_t         peer;
    int                posts, nreqs;
    size_t             rdt_size, sdt_size, data_size;
    ptrdiff_t          data_displ;
    ucc_status_t       status;

    posts    = UCC_TL_UCP_TEAM_LIB(team)->cfg.alltoallv_pairwise_num_posts;
    nreqs    = (posts > gsize || posts == 0) ? gsize : posts;
    rdt_size = ucc_dt_size(TASK_ARGS(task).dst.info_v.datatype);
    sdt_size = ucc_dt_size(TASK_ARGS(task).src.info_v.datatype);
    while ((task->tagged.send_posted < (gsize - 1) ||
            task->tagged.recv_posted < (gsize - 1)) &&
           (polls++ < task->n_polls)) {
        ucp_worker_progress(UCC_TL_UCP_TEAM_CTX(team)->worker.ucp_worker);
        while ((task->tagged.recv_posted < (gsize - 1)) &&
               ((task->tagged.recv_posted - task->tagged.recv_completed) <
                nreqs)) {
            peer = get_recv_peer(grank, gsize, task->tagged.recv_posted);
            data_size =
                ucc_coll_args_get_count(
                    &TASK_ARGS(task), TASK_ARGS(task).dst.info_v.counts, peer) *
                rdt_size;
            data_displ = ucc_coll_args_get_displacement(
                             &TASK_ARGS(task),
                             TASK_ARGS(task).dst.info_v.displacements, peer) *
                         rdt_size;
            UCPCHECK_GOTO(ucc_tl_ucp_recv_nz((void *)(rbuf + data_displ),
                                             data_size, rmem, peer, team, task),
                          task, out);
            polls = 0;
        }
        while ((task->tagged.send_posted < (gsize - 1)) &&
               ((task->tagged.send_posted - task->tagged.send_completed) <
                nreqs)) {
            peer = get_send_peer(grank, gsize, task->tagged.send_posted);
            data_size =
                ucc_coll_args_get_count(
                    &TASK_ARGS(task), TASK_ARGS(task).src.info_v.counts, peer) *
                sdt_size;
            data_displ = ucc_coll_args_get_displacement(
                             &TASK_ARGS(task),
                             TASK_ARGS(task).src.info_v.displacements, peer) *
                         sdt_size;
            UCPCHECK_GOTO(ucc_tl_ucp_send_nz((void *)(sbuf + data_displ),
                                             data_size, smem, peer, team, task),
                          task, out);
            polls = 0;
        }
    }
    if ((task->tagged.send_posted < (gsize - 1)) ||
        (task->tagged.recv_posted < (gsize - 1))) {
        return;
    }

    status = ucc_tl_ucp_test(task);
    if (status != UCC_OK) {
        task->super.status = status;
        return;
    }

    status = ucc_ee_executor_task_test(task->alltoallv_pairwise.etask);
    if (status == UCC_INPROGRESS) {
        return;
    }
    ucc_ee_executor_task_finalize(task->alltoallv_pairwise.etask);
    task->super.status = status;
out:
    if (task->super.status != UCC_INPROGRESS) {
        UCC_TL_UCP_PROFILE_REQUEST_EVENT(coll_task,
                                         "ucp_alltoallv_pairwise_done", 0);
    }
}

static ucc_status_t ucc_tl_ucp_alltoallv_pairwise_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t *task  = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_tl_ucp_team_t *team  = TASK_TEAM(task);
    ucc_rank_t         grank = UCC_TL_TEAM_RANK(team);
    ucc_coll_args_t   *args  = &TASK_ARGS(task);
    void              *sbuf  = args->src.info_v.buffer;
    void              *rbuf  = args->dst.info_v.buffer;
    ucc_ee_executor_t *exec;
    ucc_ee_executor_task_args_t eargs;
    size_t data_size;
    ucc_status_t status;
    ptrdiff_t sdispl, ddispl;

    UCC_TL_UCP_PROFILE_REQUEST_EVENT(coll_task, "ucp_alltoallv_pairwise_start", 0);
    ucc_tl_ucp_task_reset(task, UCC_INPROGRESS);
    data_size = ucc_dt_size(args->src.info_v.datatype) *
                ucc_coll_args_get_count(args, args->src.info_v.counts, grank);
    sdispl = ucc_dt_size(args->src.info_v.datatype) *
             ucc_coll_args_get_displacement(args,
                                            args->src.info_v.displacements, grank);
    ddispl = ucc_dt_size(args->dst.info_v.datatype) *
             ucc_coll_args_get_displacement(args,
                                            args->dst.info_v.displacements, grank);

    eargs.task_type = UCC_EE_EXECUTOR_TASK_COPY;
    eargs.copy.src  = PTR_OFFSET(sbuf, sdispl);
    eargs.copy.dst  = PTR_OFFSET(rbuf, ddispl);
    eargs.copy.len  = data_size;

    status = ucc_coll_task_get_executor(&task->super, &exec);
    if (ucc_unlikely(status != UCC_OK)) {
        return status;
    }

    status = ucc_ee_executor_task_post(exec, &eargs,
                                       &task->alltoallv_pairwise.etask);
    if (ucc_unlikely(status != UCC_OK)) {
        return status;
    }

    return ucc_progress_queue_enqueue(UCC_TL_CORE_CTX(team)->pq, &task->super);
}

ucc_status_t ucc_tl_ucp_alltoallv_pairwise_init_common(ucc_tl_ucp_task_t *task)
{
    ucc_tl_ucp_team_t *team = TASK_TEAM(task);
    ucc_rank_t         size = UCC_TL_TEAM_SIZE(team);
    ucc_coll_args_t   *args = &TASK_ARGS(task);

    task->super.post     = ucc_tl_ucp_alltoallv_pairwise_start;
    task->super.progress = ucc_tl_ucp_alltoallv_pairwise_progress;
    task->super.flags    |= UCC_COLL_TASK_FLAG_EXECUTOR;

    task->n_polls = ucc_max(1, task->n_polls);
    if (UCC_TL_UCP_TEAM_CTX(team)->cfg.pre_reg_mem) {
        if (args->flags & UCC_COLL_ARGS_FLAG_CONTIG_SRC_BUFFER) {
            ucc_tl_ucp_pre_register_mem(
                team, args->src.info_v.buffer,
                (ucc_coll_args_get_total_count(args, args->src.info_v.counts,
                                               size) *
                 ucc_dt_size(args->src.info_v.datatype)),
                args->src.info_v.mem_type);
        }

        if (args->flags & UCC_COLL_ARGS_FLAG_CONTIG_DST_BUFFER) {
            ucc_tl_ucp_pre_register_mem(
                team, args->dst.info_v.buffer,
                (ucc_coll_args_get_total_count(args, args->dst.info_v.counts,
                                               size) *
                 ucc_dt_size(args->dst.info_v.datatype)),
                args->dst.info_v.mem_type);
        }
    }
    return UCC_OK;
}
