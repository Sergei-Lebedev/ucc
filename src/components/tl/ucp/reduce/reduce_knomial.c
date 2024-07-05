/**
 * Copyright (c) 2021-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "config.h"
#include "reduce.h"
#include "core/ucc_progress_queue.h"
#include "tl_ucp_sendrecv.h"
#include "utils/ucc_math.h"
#include "utils/ucc_dt_reduce.h"

#define SAVE_STATE(_phase)                                                     \
    do {                                                                       \
        task->reduce_kn.phase = _phase;                                        \
    } while (0)

void ucc_tl_ucp_reduce_knomial_progress(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t *task       = ucc_derived_of(coll_task,
                                                   ucc_tl_ucp_task_t);
    ucc_coll_args_t   *args       = &TASK_ARGS(task);
    ucc_tl_ucp_team_t *team       = TASK_TEAM(task);
    int                avg_pre_op =
        UCC_TL_UCP_TEAM_LIB(team)->cfg.reduce_avg_pre_op;
    ucc_rank_t         rank       = UCC_TL_TEAM_RANK(team);
    ucc_rank_t         size       = UCC_TL_TEAM_SIZE(team);
    ucc_rank_t         root       = (ucc_rank_t)args->root;
    uint32_t           radix      = task->reduce_kn.radix;
    ucc_rank_t         vrank      = (rank - root + size) % size;
    void              *rbuf       = (rank == root) ? args->dst.info.buffer :
                                                      task->reduce_kn.scratch;
    ucc_memory_type_t  mtype;
    ucc_datatype_t     dt;
    size_t             count, data_size;
    void              *received_vectors, *scratch_offset;
    ucc_rank_t         vpeer, peer, vroot_at_level, root_at_level, pos;
    uint32_t           i;
    ucc_status_t       status;
    int                is_avg;

    if (root == rank) {
        count = args->dst.info.count;
        data_size = count * ucc_dt_size(args->dst.info.datatype);
        mtype = args->dst.info.mem_type;
        dt = args->dst.info.datatype;
    } else {
        count = args->src.info.count;
        data_size = count * ucc_dt_size(args->src.info.datatype);
        mtype = args->src.info.mem_type;
        dt = args->src.info.datatype;
    }
    received_vectors = PTR_OFFSET(task->reduce_kn.scratch, data_size);

UCC_REDUCE_KN_PHASE_PROGRESS:
    if (UCC_INPROGRESS == ucc_tl_ucp_test(task)) {
        return;
    }

    UCC_REDUCE_KN_GOTO_PHASE(task->reduce_kn.phase);

UCC_REDUCE_KN_PHASE_INIT:

    while (task->reduce_kn.dist <= task->reduce_kn.max_dist) {
        if (vrank % task->reduce_kn.dist == 0) {
            pos = (vrank / task->reduce_kn.dist) % radix;
            if (pos == 0) {
                scratch_offset = received_vectors;
                task->reduce_kn.children_per_cycle = 0;
                for (i = 1; i < radix; i++) {
                    vpeer = vrank + i * task->reduce_kn.dist;
                    if (vpeer >= size) {
                    	break;
                    } else {
                        task->reduce_kn.children_per_cycle += 1;
                        peer = (vpeer + root) % size;
                        UCPCHECK_GOTO(ucc_tl_ucp_recv_nb(scratch_offset,
                                          data_size, mtype, peer, team, task),
                                          task, out);
                        scratch_offset = PTR_OFFSET(scratch_offset, data_size);
                    }
                }
                SAVE_STATE(UCC_REDUCE_KN_PHASE_MULTI);
                goto UCC_REDUCE_KN_PHASE_PROGRESS;
UCC_REDUCE_KN_PHASE_MULTI:
                if (task->reduce_kn.children_per_cycle && count > 0) {
                    is_avg = args->op == UCC_OP_AVG &&
                             (avg_pre_op ? (task->reduce_kn.dist == 1)
                                         : (task->reduce_kn.dist ==
                                            task->reduce_kn.max_dist));
                    status = ucc_dt_reduce_strided(
                        (task->reduce_kn.dist == 1) ? args->src.info.buffer
                                                    : rbuf,
                        received_vectors, rbuf,
                        task->reduce_kn.children_per_cycle, count, data_size,
                        dt, args,
                        is_avg ? UCC_EEE_TASK_FLAG_REDUCE_WITH_ALPHA : 0,
                        AVG_ALPHA(task), task->reduce_kn.executor,
                        &task->reduce_kn.etask);
                    if (ucc_unlikely(UCC_OK != status)) {
                        tl_error(UCC_TASK_LIB(task),
                                 "failed to perform dt reduction");
                        task->super.status = status;
                        return;
                    }
                    EXEC_TASK_WAIT(task->reduce_kn.etask);
                }
            } else {
                vroot_at_level = vrank - pos * task->reduce_kn.dist;
                root_at_level  = (vroot_at_level + root) % size;
                UCPCHECK_GOTO(ucc_tl_ucp_send_nb(task->reduce_kn.scratch,
                                  data_size, mtype, root_at_level, team, task),
                                  task, out);
            }
        }
        task->reduce_kn.dist *= radix;
        SAVE_STATE(UCC_REDUCE_KN_PHASE_INIT);
        goto UCC_REDUCE_KN_PHASE_PROGRESS;
    }

    ucc_assert(UCC_TL_UCP_TASK_P2P_COMPLETE(task));
    task->super.status = UCC_OK;
    UCC_TL_UCP_PROFILE_REQUEST_EVENT(coll_task, "ucp_reduce_kn_done", 0);
out:
    return;
}

ucc_status_t ucc_tl_ucp_reduce_knomial_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t *task       =
        ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_coll_args_t   *args       = &TASK_ARGS(task);
    ucc_tl_ucp_team_t *team       = TASK_TEAM(task);
    uint32_t           radix      = task->reduce_kn.radix;
    ucc_rank_t         root       = (ucc_rank_t)args->root;
    ucc_rank_t         rank       = UCC_TL_TEAM_RANK(team);
    ucc_rank_t         size       = UCC_TL_TEAM_SIZE(team);
    ucc_rank_t         vrank      = (rank - root + size) % size;
    int                isleaf     =
        (vrank % radix != 0 || vrank == size - 1);
    int                avg_pre_op =
        UCC_TL_UCP_TEAM_LIB(team)->cfg.reduce_avg_pre_op;
    int                self_avg   = (args->op == UCC_OP_AVG &&
        avg_pre_op && vrank % radix == 0);
    size_t             count;
    ucc_datatype_t     dt;
    ucc_status_t       status;

    if (root == rank) {
        count = args->dst.info.count;
        dt    = args->dst.info.datatype;
    } else {
        count = args->src.info.count;
        dt    = args->src.info.datatype;
    }

    UCC_TL_UCP_PROFILE_REQUEST_EVENT(coll_task, "ucp_reduce_kn_start", 0);
    ucc_tl_ucp_task_reset(task, UCC_INPROGRESS);

    if (UCC_IS_INPLACE(*args) && (rank == root)) {
        args->src.info.buffer = args->dst.info.buffer;
    }

    if (isleaf && !self_avg) {
    	task->reduce_kn.scratch = args->src.info.buffer;
    }

    if (args->coll_type != UCC_COLL_TYPE_FANIN) {
        status =
            ucc_coll_task_get_executor(&task->super, &task->reduce_kn.executor);
        if (ucc_unlikely(status != UCC_OK)) {
            return status;
        }
    }

    if (isleaf && self_avg) {
        /* In case of avg_pre_op, single leaf process which does not take part
           in first iteration reduction must divide itself by team_size */

        status =
            ucc_dt_reduce(args->src.info.buffer, args->src.info.buffer,
                          task->reduce_kn.scratch, count, dt, args,
                          UCC_EEE_TASK_FLAG_REDUCE_WITH_ALPHA,
                          1.0 / (double)(UCC_TL_TEAM_SIZE(TASK_TEAM(task)) * 2),
                          task->reduce_kn.executor, &task->reduce_kn.etask);
        if (ucc_unlikely(UCC_OK != status)) {
            tl_error(UCC_TASK_LIB(task),
                     "failed to perform dt reduction");
            task->super.status = status;
            return status;
        }
        EXEC_TASK_WAIT(task->reduce_kn.etask, status);
    }

    task->reduce_kn.dist = 1;
    task->reduce_kn.phase = UCC_REDUCE_KN_PHASE_INIT;

    return ucc_progress_queue_enqueue(UCC_TL_CORE_CTX(team)->pq, &task->super);
}

ucc_status_t ucc_tl_ucp_reduce_knomial_finalize(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t *task      =
        ucc_derived_of(coll_task, ucc_tl_ucp_task_t);

    if (task->reduce_kn.scratch_mc_header) {
        ucc_mc_free(task->reduce_kn.scratch_mc_header);
    }
    return ucc_tl_ucp_coll_finalize(coll_task);
}

static void
ucc_tl_ucp_reduce_knomial_get_pipeline_params(ucc_tl_ucp_team_t *team,
                                              ucc_coll_args_t *args,
                                              ucc_pipeline_params_t *pp)
{
    ucc_tl_ucp_lib_config_t *cfg = &team->cfg;
    ucc_mc_attr_t mc_attr;

    if (!ucc_pipeline_params_is_auto(&cfg->reduce_kn_pipeline)) {
        *pp = cfg->reduce_kn_pipeline;
        return;
    }

    if (args->src.info.mem_type == UCC_MEMORY_TYPE_CUDA) {
        mc_attr.field_mask = UCC_MC_ATTR_FIELD_FAST_ALLOC_SIZE;
        ucc_mc_get_attr(&mc_attr, UCC_MEMORY_TYPE_CUDA);
        pp->threshold = mc_attr.fast_alloc_size;
        pp->n_frags   = 2;
        pp->frag_size = mc_attr.fast_alloc_size;
        pp->order     = UCC_PIPELINE_PARALLEL;
        pp->pdepth    = 2;
    } else {
        pp->threshold = SIZE_MAX;
        pp->n_frags   = 0;
        pp->frag_size = 0;
        pp->pdepth    = 1;
        pp->order     = UCC_PIPELINE_PARALLEL;
    }
}

static ucc_status_t
ucc_tl_ucp_reduce_knomial_frag_start(ucc_coll_task_t *task)
{
    return ucc_schedule_start(task);
}

static ucc_status_t
ucc_tl_ucp_reduce_knomial_frag_finalize(ucc_coll_task_t *task)
{
    ucc_schedule_t *schedule = ucc_derived_of(task, ucc_schedule_t);
    ucc_status_t    status;

    status = ucc_schedule_finalize(task);
    ucc_tl_ucp_put_schedule(schedule);
    return status;
}

static ucc_status_t
ucc_tl_ucp_reduce_knomial_frag_init(ucc_base_coll_args_t *coll_args,
                                    ucc_schedule_pipelined_t *sp, //NOLINT
                                    ucc_base_team_t *team,
                                    ucc_schedule_t **frag_p)
{
    ucc_tl_ucp_team_t   *tl_team  = ucc_derived_of(team, ucc_tl_ucp_team_t);
    ucc_base_coll_args_t args     = *coll_args;
    ucc_status_t         status;
    ucc_schedule_t      *schedule;
    ucc_coll_task_t     *task;

    status = ucc_tl_ucp_get_schedule(tl_team, coll_args,
                                     (ucc_tl_ucp_schedule_t **)&schedule);
    if (ucc_unlikely(UCC_OK != status)) {
        return status;
    }

    UCC_CHECK_GOTO(
        ucc_tl_ucp_reduce_knomial_init(&args, team, &task),
        out, status);

    UCC_CHECK_GOTO(ucc_schedule_add_task(schedule, task), out, status);
    UCC_CHECK_GOTO(ucc_task_subscribe_dep(&schedule->super, task,
                                          UCC_EVENT_SCHEDULE_STARTED),
                   out, status);
    schedule->super.finalize = ucc_tl_ucp_reduce_knomial_frag_finalize;
    schedule->super.post     = ucc_tl_ucp_reduce_knomial_frag_start;
    *frag_p                  = schedule;
    return UCC_OK;
out:
    return status;
}

static ucc_status_t
ucc_tl_ucp_reduce_knomial_frag_setup(ucc_schedule_pipelined_t *schedule_p,
                                     ucc_schedule_t *frag, int frag_num)
{
    ucc_coll_args_t   *args       = &schedule_p->super.super.bargs.args;
    int                n_frags    = schedule_p->super.n_tasks;
    size_t             count;
    size_t             frag_count = ucc_buffer_block_count(args->dst.info.count,
                                                           n_frags, frag_num);
    size_t             offset     = ucc_buffer_block_offset(args->dst.info.count,
                                                            n_frags, frag_num);
    ucc_tl_ucp_task_t *task       = ucc_derived_of(frag->tasks[0],
                                                   ucc_tl_ucp_task_t);
    ucc_tl_ucp_team_t *team       = TASK_TEAM(task);
    ucc_datatype_t     dt;
    ucc_coll_args_t   *targs;

    if (args->root == UCC_TL_TEAM_RANK(team)) {
        count = args->dst.info.count;
        dt    = args->dst.info.datatype;
    } else {
        count = args->src.info.count;
        dt    = args->src.info.datatype;
    }

    frag_count = ucc_buffer_block_count(count, n_frags, frag_num);
    offset = ucc_buffer_block_offset(count, n_frags, frag_num);
    targs = &task->super.bargs.args;
    targs->src.info.buffer = PTR_OFFSET(args->src.info.buffer,
                                        offset * ucc_dt_size(dt));
    targs->src.info.count  = frag_count;
    targs->dst.info.buffer = PTR_OFFSET(args->dst.info.buffer,
                                        offset * ucc_dt_size(dt));
    targs->dst.info.count  = frag_count;

    return UCC_OK;
}

ucc_status_t ucc_tl_ucp_reduce_knomial_init_common(ucc_tl_ucp_task_t *task)
{
    ucc_coll_args_t          *args      = &TASK_ARGS(task);
    ucc_tl_ucp_team_t        *team      = TASK_TEAM(task);
    ucc_rank_t                myrank    = UCC_TL_TEAM_RANK(team);
    ucc_rank_t                team_size = UCC_TL_TEAM_SIZE(team);
    ucc_rank_t                root      = args->root;
    ucc_rank_t                vrank     = (myrank - root + team_size) % team_size;
    ucc_status_t              status    = UCC_OK;
    ucc_memory_type_t         mtype;
    ucc_datatype_t            dt;
    size_t                    count, data_size;
    int                       isleaf;
    int                       self_avg;

    if (root == myrank) {
        count = args->dst.info.count;
        dt    = args->dst.info.datatype;
        mtype = args->dst.info.mem_type;
    } else {
        count = args->src.info.count;
        dt    = args->src.info.datatype;
        mtype = args->src.info.mem_type;
    }

    data_size = count * ucc_dt_size(dt);
    task->super.flags    |= UCC_COLL_TASK_FLAG_EXECUTOR;
    task->super.post      = ucc_tl_ucp_reduce_knomial_start;
    task->super.progress  = ucc_tl_ucp_reduce_knomial_progress;
    task->super.finalize  = ucc_tl_ucp_reduce_knomial_finalize;
    task->reduce_kn.radix =
        ucc_min(UCC_TL_UCP_TEAM_LIB(team)->cfg.reduce_kn_radix, team_size);
    CALC_KN_TREE_DIST(team_size, task->reduce_kn.radix,
                      task->reduce_kn.max_dist);
    isleaf   = (vrank % task->reduce_kn.radix != 0 || vrank == team_size - 1);
    self_avg = (vrank % task->reduce_kn.radix == 0 && args->op == UCC_OP_AVG &&
                UCC_TL_UCP_TEAM_LIB(team)->cfg.reduce_avg_pre_op);
    task->reduce_kn.scratch_mc_header = NULL;

    if (!isleaf || self_avg) {
    	/* scratch of size radix to fit up to radix - 1 recieved vectors
    	from its children at each step,
    	and an additional 1 for previous step reduce multi result */
        status = ucc_mc_alloc(&task->reduce_kn.scratch_mc_header,
                              task->reduce_kn.radix * data_size, mtype);
        task->reduce_kn.scratch =
                        task->reduce_kn.scratch_mc_header->addr;
    }

    return status;
}

static ucc_status_t
ucc_tl_ucp_reduce_knomial_pipelined_finalize(ucc_coll_task_t *task)
{
    ucc_schedule_t *schedule = ucc_derived_of(task, ucc_schedule_t);
    ucc_status_t status;

    UCC_TL_UCP_PROFILE_REQUEST_EVENT(schedule, "ucp_reduce_kn_pipelined_done", 0);
    status = ucc_schedule_pipelined_finalize(task);
    ucc_tl_ucp_put_schedule(schedule);
    return status;
}

ucc_status_t ucc_tl_ucp_reduce_knomial_pipelined_start(ucc_coll_task_t *task)
{
    UCC_TL_UCP_PROFILE_REQUEST_EVENT(task, "ucp_reduce_kn_pipelined_start", 0);
    return ucc_schedule_pipelined_post(task);
}

ucc_status_t ucc_tl_ucp_reduce_knomial_pipelined_init(ucc_base_coll_args_t *coll_args,
                                                      ucc_base_team_t *team,
                                                      ucc_coll_task_t **task_h)
{
    ucc_tl_ucp_team_t        *tl_team = ucc_derived_of(team, ucc_tl_ucp_team_t);
    ucc_coll_args_t          *args  = &coll_args->args;
    ucc_rank_t                trank = UCC_TL_TEAM_RANK(tl_team);
    ucc_rank_t                root  = args->root;
    ucc_status_t              status;
    ucc_schedule_pipelined_t *schedule_p;
    ucc_base_coll_args_t      bargs;
    ucc_pipeline_params_t     pipeline_params;
    size_t                    max_frag_count;
    int                       n_frags, pipeline_depth;
    ucc_datatype_t            dt;
    size_t                    count;

    ucc_tl_ucp_reduce_knomial_get_pipeline_params(tl_team, args,
                                                  &pipeline_params);
    if (pipeline_params.n_frags == 1) {
        /* pipelining is not needed, use regular knomial algorithm */
        return ucc_tl_ucp_reduce_knomial_init(coll_args, team, task_h);
    }

    if (root == trank) {
        count = args->dst.info.count;
        dt    = args->dst.info.datatype;
    } else {
        count = args->src.info.count;
        dt    = args->src.info.datatype;
    }

    status  = ucc_tl_ucp_get_schedule(tl_team, coll_args,
                                      (ucc_tl_ucp_schedule_t **)&schedule_p);
    if (ucc_unlikely(UCC_OK != status)) {
        return status;
    }

    bargs = *coll_args;
    max_frag_count = (bargs.mask & UCC_BASE_CARGS_MAX_FRAG_COUNT) ?
                     bargs.max_frag_count: count;

    ucc_pipeline_nfrags_pdepth(&pipeline_params,
                               max_frag_count * ucc_dt_size(dt),
                               &n_frags, &pipeline_depth);

    status = ucc_schedule_pipelined_init(&bargs, team,
                                         ucc_tl_ucp_reduce_knomial_frag_init,
                                         ucc_tl_ucp_reduce_knomial_frag_setup,
                                         pipeline_depth, n_frags,
                                         pipeline_params.order, schedule_p);
    if (ucc_unlikely(UCC_OK != status)) {
        tl_error(team->context->lib, "failed to init pipelined schedule");
        ucc_tl_ucp_put_schedule(&schedule_p->super);
        return status;
    }

    schedule_p->super.super.finalize = ucc_tl_ucp_reduce_knomial_pipelined_finalize;
    schedule_p->super.super.post     = ucc_tl_ucp_reduce_knomial_pipelined_start;
    *task_h = &schedule_p->super.super;

    return UCC_OK;
}

ucc_status_t ucc_tl_ucp_reduce_knomial_init(ucc_base_coll_args_t *coll_args,
                                            ucc_base_team_t *team,
                                            ucc_coll_task_t **task_h)
{
    ucc_tl_ucp_task_t *task;
    ucc_status_t       status;

    task = ucc_tl_ucp_init_task(coll_args, team);
    if (ucc_unlikely(!task)) {
        return UCC_ERR_NO_MEMORY;
    }

    *task_h = &task->super;

    status = ucc_tl_ucp_reduce_knomial_init_common(task);
    if (ucc_unlikely(status != UCC_OK)) {
        ucc_tl_ucp_put_task(task);
    }
    return status;
}
