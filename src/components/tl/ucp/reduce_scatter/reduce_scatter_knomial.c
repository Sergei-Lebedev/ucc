/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "config.h"
#include "tl_ucp.h"
#include "tl_ucp_coll.h"
#include "tl_ucp_sendrecv.h"
#include "core/ucc_progress_queue.h"
#include "core/ucc_mc.h"
#include "coll_patterns/sra_knomial.h"
#include "utils/ucc_math.h"
#include "utils/ucc_coll_utils.h"

#define SAVE_STATE(_phase)                                                     \
    do {                                                                       \
        task->reduce_scatter_kn.phase = _phase;                                \
    } while (0)

ucc_status_t
ucc_tl_ucp_reduce_scatter_knomial_progress(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t     *task  = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_coll_args_t       *args  = &coll_task->args;
    ucc_tl_ucp_team_t     *team  = TASK_TEAM(task);
    ucc_kn_radix_t         radix = task->reduce_scatter_kn.p.radix;
    uint8_t                node_type = task->reduce_scatter_kn.p.node_type;
    ucc_knomial_pattern_t *p         = &task->reduce_scatter_kn.p;
    void                  *scratch   = task->reduce_scatter_kn.scratch;
    void                  *rbuf      = args->dst.info.buffer;
    ucc_memory_type_t      mem_type  = args->dst.info.mem_type;
    size_t                 count     = args->dst.info.count;
    ucc_datatype_t         dt        = args->dst.info.datatype;
    void                  *sbuf      = UCC_IS_INPLACE(*args) ?
        rbuf : args->src.info.buffer;
    size_t                 dt_size   = ucc_dt_size(dt);
    size_t                 data_size = count * dt_size;
    ucc_rank_t             size      = team->size;
    ucc_rank_t             rank      = team->rank;
    uint32_t               n_eee_tasks = UCC_TL_UCP_TEAM_LIB(team)->cfg.allreduce_sra_kn_n_eee_tasks;
    ptrdiff_t              peer_seg_offset, local_seg_offset, offset;
    ucc_rank_t             peer, step_radix, peer_seg_index, local_seg_index;
    ucc_status_t           status;
    ucc_kn_radix_t         loop_step;
    size_t                 block_count, peer_seg_count, local_seg_count;
    void                  *reduce_data, *local_data;

    local_seg_count = 0;
    block_count     = ucc_sra_kn_compute_block_count(count, rank, p);
    UCC_KN_GOTO_PHASE(task->reduce_scatter_kn.phase);

    if (KN_NODE_EXTRA == node_type) {
        peer = ucc_knomial_pattern_get_proxy(p, rank);
        UCPCHECK_GOTO(
            ucc_tl_ucp_send_nb(sbuf, data_size, mem_type, peer, team, task),
            task, out);
    }

    if (KN_NODE_PROXY == node_type) {
        peer = ucc_knomial_pattern_get_extra(p, rank);
        UCPCHECK_GOTO(
            ucc_tl_ucp_recv_nb(scratch, data_size, mem_type, peer, team, task),
            task, out);
    }

UCC_KN_PHASE_EXTRA:
    if ((KN_NODE_PROXY == node_type) || (KN_NODE_EXTRA == node_type)) {
        if (UCC_INPROGRESS == ucc_tl_ucp_test(task)) {
            SAVE_STATE(UCC_KN_PHASE_EXTRA);
            return task->super.super.status;
        }
        if (KN_NODE_EXTRA == node_type) {
            goto out;
        } else {
            if (UCC_OK != (status = ucc_dt_reduce(sbuf, scratch, rbuf, count,
                                                  dt, mem_type, args))) {
                tl_error(UCC_TASK_LIB(task), "failed to perform dt reduction");
                task->super.super.status = status;
                return status;
            }
        }
    }
    while (!ucc_knomial_pattern_loop_done(p)) {
        step_radix  = ucc_sra_kn_compute_step_radix(rank, size, p);
        block_count = ucc_sra_kn_compute_block_count(count, rank, p);
        sbuf        = (p->iteration == 0)
            ? ((KN_NODE_PROXY == node_type || UCC_IS_INPLACE(*args)) ?
               args->dst.info.buffer : args->src.info.buffer)
                          : task->reduce_scatter_kn.scratch;
        for (loop_step = 1; loop_step < radix; loop_step++) {
            peer = ucc_knomial_pattern_get_loop_peer(p, rank, size, loop_step);
            if (peer == UCC_KN_PEER_NULL)
                continue;

            peer_seg_index =
                ucc_sra_kn_compute_seg_index(peer, p->radix_pow, p);
            peer_seg_count = ucc_sra_kn_compute_seg_size(
                block_count, step_radix, peer_seg_index);
            peer_seg_offset = ucc_sra_kn_compute_seg_offset(
                block_count, step_radix, peer_seg_index);
            UCPCHECK_GOTO(
                ucc_tl_ucp_send_nb(PTR_OFFSET(sbuf, peer_seg_offset * dt_size),
                                   peer_seg_count * dt_size, mem_type, peer,
                                   team, task),
                task, out);
        }

        local_seg_index = ucc_sra_kn_compute_seg_index(rank, p->radix_pow, p);
        local_seg_count = ucc_sra_kn_compute_seg_size(block_count, step_radix,
                                                      local_seg_index);

        rbuf = task->reduce_scatter_kn.scratch;
        if (p->iteration != 0) {
            rbuf = PTR_OFFSET(rbuf, block_count * dt_size);
        }
        for (loop_step = 1; loop_step < radix; loop_step++) {
            peer = ucc_knomial_pattern_get_loop_peer(p, rank, size, loop_step);
            if (peer == UCC_KN_PEER_NULL)
                continue;
            UCPCHECK_GOTO(ucc_tl_ucp_recv_nb(rbuf, local_seg_count * dt_size,
                                             mem_type, peer, team, task),
                          task, out);
            rbuf = PTR_OFFSET(rbuf, local_seg_count * dt_size);
        }
    UCC_KN_PHASE_LOOP:
        if (UCC_INPROGRESS == ucc_tl_ucp_test(task)) {
            SAVE_STATE(UCC_KN_PHASE_LOOP);
            return task->super.super.status;
        }
        if (task->send_posted > p->iteration * (radix - 1)) {

            sbuf       = (p->iteration == 0)
                ? ((KN_NODE_PROXY == node_type  || UCC_IS_INPLACE(*args)) ?
                   args->dst.info.buffer : args->src.info.buffer)
                             : task->reduce_scatter_kn.scratch;
            rbuf       = (p->iteration != 0)
                             ? PTR_OFFSET(task->reduce_scatter_kn.scratch,
                                    block_count * dt_size)
                             : task->reduce_scatter_kn.scratch;
            step_radix = ucc_sra_kn_compute_step_radix(rank, size, p);
            local_seg_index =
                ucc_sra_kn_compute_seg_index(rank, p->radix_pow, p);
            local_seg_count = ucc_sra_kn_compute_seg_size(
                block_count, step_radix, local_seg_index);
            local_seg_offset = ucc_sra_kn_compute_seg_offset(
                block_count, step_radix, local_seg_index);
            local_data  = PTR_OFFSET(sbuf, local_seg_offset * dt_size);
            reduce_data = task->reduce_scatter_kn.scratch;
            if (ucc_knomial_pattern_loop_last(p)) {
                size_t lsc2;
                ucc_sra_kn_get_offset_and_seglen(count, dt_size, rank, size, radix,
                                                 &offset, &lsc2);
                ucc_assert(lsc2 == local_seg_count);
                if (args->coll_type != UCC_COLL_TYPE_ALLREDUCE) {
                    offset = 0;
                }
                reduce_data = PTR_OFFSET(args->dst.info.buffer, offset);
            }

            if (!task->reduce_scatter_kn.eee) {
                if (UCC_OK != (status = ucc_dt_reduce_multi(
                                   local_data, rbuf, reduce_data,
                                   task->send_posted - p->iteration * (radix - 1),
                                   local_seg_count, local_seg_count * dt_size, dt,
                                   mem_type, args))) {
                    tl_error(UCC_TASK_LIB(task), "failed to perform dt reduction");
                    task->super.super.status = status;
                    return status;
                }
            } else {
                int t;
                for (t = 0; t < n_eee_tasks; t++) {
                    size_t count_per_task = local_seg_count / n_eee_tasks;
                    size_t left           = local_seg_count % n_eee_tasks;

                    offset = t * count_per_task + left;
                    if (t < left) {
                        count_per_task++;
                        offset -= left - t;
                    }

                    ucc_ee_executor_task_args_t exec_args;
                    exec_args.task_type     = UCC_MC_EE_EXECUTOR_TASK_TYPE_REDUCE_MULTI;
                    exec_args.dst.buffer    = PTR_OFFSET(reduce_data, offset * dt_size);
                    exec_args.dst.count     = count_per_task;
                    exec_args.dst.datatype  = dt;
                    exec_args.op            = UCC_OP_SUM; //tTODO from args
                    int n_received_vecs = task->send_posted - p->iteration * (radix - 1);
                    exec_args.src3_size     = n_received_vecs + 1;
                    exec_args.src3[0] = PTR_OFFSET(local_data, offset * dt_size);
                    int j;
                    for (j=0; j < n_received_vecs; j++) {
                        exec_args.src3[j+1] = PTR_OFFSET(rbuf, (local_seg_count * j + offset) * dt_size);
                    }
                    task->reduce_scatter_kn.ids[t] = nvtxRangeStartA("eee_reduce");
                    status = ucc_ee_executor_task_post(&exec_args,
                                                       &task->reduce_scatter_kn.exec_tasks[t],
                                                       task->reduce_scatter_kn.eee);
                    if (ucc_unlikely(status != UCC_OK)) {
                        task->super.super.status = status;
                        return status;
                    }
                }
            }
        }

    UCC_KN_PHASE_REDUCE:
        {
            int t;
            for (t = 0; t < n_eee_tasks; t++) {
                if (task->reduce_scatter_kn.exec_tasks[t]) {
                    status = ucc_ee_executor_task_test(task->reduce_scatter_kn.exec_tasks[t]);
                    if (UCC_OK != status) {
                        if (status > 0) {
                            status = UCC_INPROGRESS;
                        }
                        SAVE_STATE(UCC_KN_PHASE_REDUCE);
                        task->super.super.status = status;
                        return task->super.super.status;
                    }
                    task->reduce_scatter_kn.exec_tasks[t] = NULL;
                    nvtxRangeEnd(task->reduce_scatter_kn.ids[t]);
                }
            }
        }
        ucc_knomial_pattern_next_iteration(p);
    }

UCC_KN_PHASE_PROXY: /* unused label */
out:
    nvtxRangeEnd(coll_task->id);
    UCC_TL_UCP_PROFILE_REQUEST_EVENT(coll_task, "ucp_reduce_scatter_kn_done",
                                     0);
    task->super.super.status = UCC_OK;
    return task->super.super.status;
}

ucc_status_t ucc_tl_ucp_reduce_scatter_knomial_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t *task = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_coll_args_t   *args = &coll_task->args;
    ucc_tl_ucp_team_t *team = TASK_TEAM(task);
    ucc_status_t       status;
    uint8_t            node_type;

    coll_task->id = nvtxRangeStartA("rs_kn_start");
    UCC_TL_UCP_PROFILE_REQUEST_EVENT(coll_task, "ucp_reduce_scatter_kn_start",
                                     0);
    ucc_tl_ucp_task_reset(task);
    memset(task->reduce_scatter_kn.exec_tasks, 0,
           sizeof(task->reduce_scatter_kn.exec_tasks));
    ucc_knomial_pattern_init(team->size, team->rank,
                             task->reduce_scatter_kn.p.radix,
                             &task->reduce_scatter_kn.p);
    node_type = task->reduce_scatter_kn.p.node_type;
    if (!(UCC_IS_INPLACE(*args) || (KN_NODE_PROXY == node_type) ||
            args->coll_type == UCC_COLL_TYPE_REDUCE_SCATTER)) {
        task->reduce_scatter_kn.scratch = args->dst.info.buffer;
    }
    task->reduce_scatter_kn.phase = UCC_KN_PHASE_INIT;

    status = ucc_tl_ucp_reduce_scatter_knomial_progress(&task->super);
    if (UCC_INPROGRESS == status) {
        ucc_progress_enqueue(UCC_TL_CORE_CTX(team)->pq, &task->super);
        return UCC_OK;
    }
    return ucc_task_complete(coll_task);
}

ucc_status_t
ucc_tl_ucp_reduce_scatter_knomial_finalize(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t *task      = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    uint8_t            node_type = task->reduce_scatter_kn.p.node_type;

    if (UCC_IS_INPLACE(coll_task->args) || (KN_NODE_PROXY == node_type) ||
        coll_task->args.coll_type == UCC_COLL_TYPE_REDUCE_SCATTER) {
        ucc_mc_free(task->reduce_scatter_kn.scratch_mc_header);
    }
    return ucc_tl_ucp_coll_finalize(coll_task);
}

ucc_status_t ucc_tl_ucp_reduce_scatter_knomial_init_r(
    ucc_base_coll_args_t *coll_args, ucc_base_team_t *team,
    ucc_coll_task_t **task_h, ucc_kn_radix_t radix)
{
    ucc_tl_ucp_team_t *tl_team   = ucc_derived_of(team, ucc_tl_ucp_team_t);
    ucc_rank_t         size      = tl_team->size;
    ucc_rank_t         rank      = tl_team->rank;
    size_t             count     = coll_args->args.dst.info.count;
    ucc_datatype_t     dt        = coll_args->args.dst.info.datatype;
    size_t             dt_size   = ucc_dt_size(dt);
    size_t             data_size = count * dt_size;
    ucc_memory_type_t  mem_type  = coll_args->args.dst.info.mem_type;
    ucc_tl_ucp_task_t *task;
    ucc_status_t       status;

    task                 = ucc_tl_ucp_init_task(coll_args, team);
    task->super.post     = ucc_tl_ucp_reduce_scatter_knomial_start;
    task->super.progress = ucc_tl_ucp_reduce_scatter_knomial_progress;
    task->super.finalize = ucc_tl_ucp_reduce_scatter_knomial_finalize;

    if (coll_args->mask & UCC_BASE_COLL_ARGS_FIELD_EEE) {
        task->reduce_scatter_kn.eee = coll_args->eee;
    } else {
        task->reduce_scatter_kn.eee = NULL;
    }

    ucc_assert(coll_args->args.src.info.mem_type ==
               coll_args->args.dst.info.mem_type);
    ucc_knomial_pattern_init(size, rank, radix, &task->reduce_scatter_kn.p);

    if (UCC_IS_INPLACE(coll_args->args) ||
        (KN_NODE_PROXY == task->reduce_scatter_kn.p.node_type) ||
        coll_args->args.coll_type == UCC_COLL_TYPE_REDUCE_SCATTER) {
        status = ucc_mc_alloc(&task->reduce_scatter_kn.scratch_mc_header,
                              data_size, mem_type);
        task->reduce_scatter_kn.scratch =
            task->reduce_scatter_kn.scratch_mc_header->addr;
        if (UCC_OK != status) {
            return status;
        }
    }

    *task_h = &task->super;
    return UCC_OK;
}

ucc_status_t
ucc_tl_ucp_reduce_scatter_knomial_init(ucc_base_coll_args_t *coll_args,
                                       ucc_base_team_t *     team,
                                       ucc_coll_task_t **    task_h)
{
    ucc_tl_ucp_team_t *tl_team = ucc_derived_of(team, ucc_tl_ucp_team_t);
    ucc_rank_t         size    = tl_team->size;
    size_t             count   = coll_args->args.dst.info.count;
    ucc_kn_radix_t     radix;

    radix = ucc_min(UCC_TL_UCP_TEAM_LIB(tl_team)->cfg.reduce_scatter_kn_radix,
                    size);
    if (((count + radix - 1) / radix * (radix - 1) > count) ||
        ((radix - 1) > count)) {
        radix = 2;
    }
    return ucc_tl_ucp_reduce_scatter_knomial_init_r(coll_args, team, task_h,
                                                    radix);
}
