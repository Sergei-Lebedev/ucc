/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "cl_hier.h"
#include "core/ucc_mc.h"
#include "core/ucc_team.h"
#include "utils/ucc_coll_utils.h"

ucc_status_t ucc_cl_hier_coll_init(ucc_base_coll_args_t *coll_args,
                                    ucc_base_team_t *team,
                                    ucc_coll_task_t **task)
{
    return UCC_ERR_NOT_IMPLEMENTED;
}

static ucc_status_t ucc_cl_hier_alltoallv_post(ucc_coll_task_t *task)
{
    ucc_schedule_t *schedule = ucc_derived_of(task, ucc_schedule_t);

    return ucc_schedule_start(schedule);
}

static ucc_status_t
ucc_cl_hier_alltoallv_early_triggered_post(ucc_coll_task_t *coll_task)
{
    ucc_schedule_t  *schedule = ucc_derived_of(coll_task, ucc_schedule_t);
    ucc_coll_task_t *task;
    ucc_status_t     status;
    int              i;

    for (i = 0; i < schedule->n_tasks; i++) {
        task = schedule->tasks[i];
        if (task->early_triggered_post) {
            task->ee = coll_task->ee;
            status   = task->early_triggered_post(task);
            if (UCC_OK != status) {
                cl_error(coll_task->team->context->lib,
                         "failure in early_triggered_post, task %p", task);
                coll_task->super.status = status;
                return status;
            }
        }
    }
    return UCC_OK;
}

static ucc_status_t ucc_cl_hier_alltoallv_finalize(ucc_coll_task_t *task)
{
    ucc_status_t status = UCC_OK;
    ucc_schedule_t *schedule = ucc_derived_of(task, ucc_schedule_t);

    ucc_assert(schedule->n_tasks == 2);
    ucc_free(schedule->tasks[1]->args.src.info_v.counts);

    status = ucc_schedule_finalize(&schedule->super);
    ucc_free(schedule);
    return status;
}

ucc_status_t ucc_cl_hier_alltoallv_init(ucc_base_coll_args_t *coll_args,
                                        ucc_base_team_t *team,
                                        ucc_coll_task_t **task)
{
    ucc_cl_hier_team_t    *cl_team = ucc_derived_of(team, ucc_cl_hier_team_t);
    ucc_cl_hier_lib_t     *cl_lib  = UCC_CL_HIER_TEAM_LIB(cl_team);
    ucc_status_t            status = UCC_OK;
    ucc_base_coll_init_fn_t init;
    ucc_base_team_t        *bteam;
    ucc_base_coll_args_t  args;
    ucc_coll_task_t *task_node, *task_full;
    ucc_schedule_t    *schedule;

    uint32_t *sc_full, *sd_full, *rc_full, *rd_full;
    uint32_t *sc_node, *sd_node, *rc_node, *rd_node;

    ucc_rank_t full_size, node_size;
    ucc_datatype_t sdt, rdt;
    uint32_t scount, rcount;
    ucc_sbgp_t *sbgp;
    int i;

    if ((coll_args->args.mask & UCC_COLL_ARGS_FIELD_FLAGS) &&
        ((coll_args->args.flags & UCC_COLL_ARGS_FLAG_DISPLACEMENTS_64BIT) ||
         (coll_args->args.flags & UCC_COLL_ARGS_FLAG_COUNT_64BIT))) {
        //for now
        return UCC_ERR_NOT_SUPPORTED;
    }
    schedule = ucc_malloc(sizeof(*schedule), "hier schedule");
    if (!schedule) {
        cl_error(team->context->lib, "failed to allocate %zd bytes for hybrid schedule",
                 sizeof(*schedule));
        return UCC_ERR_NO_MEMORY;
    }
    memcpy(&args, coll_args, sizeof(args));
    ucc_schedule_init(schedule, &args.args, team);

    full_size = cl_team->sbgps[UCC_HIER_SBGP_FULL].sbgp->group_size;
    node_size = cl_team->sbgps[UCC_HIER_SBGP_NODE].sbgp->group_size;

    sc_full = ucc_malloc(sizeof(uint32_t) * (full_size + node_size) * 4, "counts");
    if (!sc_full) {
        ucc_free(schedule);
        cl_error(team->context->lib, "failed to allocate %zd bytes for full counts",
                 sizeof(uint32_t) * (full_size + node_size) * 4);
        return UCC_ERR_NO_MEMORY;
    }
    sd_full = &sc_full[full_size];
    rc_full = &sc_full[full_size * 2];
    rd_full = &sc_full[full_size * 3];

    sc_node = &sc_full[full_size * 4];
    sd_node = &sc_node[node_size];
    rc_node = &sc_node[node_size * 2];
    rd_node = &sc_node[node_size * 3];

    /* Duplicate FULL a2av info and alloc task */
    sbgp = cl_team->sbgps[UCC_HIER_SBGP_FULL].sbgp;
    ucc_assert(sbgp->group_size == team->team->size);
    sdt = coll_args->args.src.info_v.datatype;
    rdt = coll_args->args.dst.info_v.datatype;
    /* Setup full counts */
    for (i = 0; i < sbgp->group_size; i++) {
        scount = ((uint32_t*)coll_args->args.src.info_v.counts)[i];
        rcount = ((uint32_t*)coll_args->args.dst.info_v.counts)[i];
        int is_local = ucc_rank_on_local_node(i, team->team);
        if ((scount * ucc_dt_size(sdt) > cl_lib->cfg.a2av_node_thresh) &&
            is_local) {
            sc_full[i] = 0;
        } else {
            sc_full[i] = scount;
            sd_full[i] = ((uint32_t*)coll_args->args.src.info_v.displacements)[i];
        }
        if ((rcount * ucc_dt_size(rdt) > cl_lib->cfg.a2av_node_thresh) &&
            is_local) {
            rc_full[i] = 0;
        } else {
            rc_full[i] = rcount;
            rd_full[i] = ((uint32_t*)coll_args->args.dst.info_v.displacements)[i];
        }
    }

    args.args.src.info_v.counts = (ucc_aint_t*)sc_full;
    args.args.dst.info_v.counts = (ucc_aint_t*)rc_full;
    args.args.src.info_v.displacements = (ucc_aint_t*)sd_full;
    args.args.dst.info_v.displacements = (ucc_aint_t*)rd_full;
    status = ucc_coll_score_map_lookup(cl_team->sbgps[UCC_HIER_SBGP_FULL].score_map,
                                       &args, &init, &bteam);
    if (UCC_OK != status) {
        cl_error(team->context->lib, "failed to lookup full a2av task");
        ucc_free(schedule);
        ucc_free(sc_full);
        return status;
    }

    status = init(&args, bteam, &task_full);
    if (UCC_OK != status) {
        cl_error(team->context->lib, "failed to init full a2av task");
        ucc_free(schedule);
        ucc_free(sc_full);
        return status;
    }

    /* Setup NODE a2av */
    sbgp = cl_team->sbgps[UCC_HIER_SBGP_NODE].sbgp;
    for (i = 0; i < sbgp->group_size; i++) {
        ucc_rank_t r = ucc_ep_map_eval(sbgp->map, i);
        scount = ((uint32_t*)coll_args->args.src.info_v.counts)[r];
        rcount = ((uint32_t*)coll_args->args.dst.info_v.counts)[r];

        if (scount * ucc_dt_size(sdt) <= cl_lib->cfg.a2av_node_thresh) {
            sc_node[i] = 0;
        } else {
            sc_node[i] = scount;
            sd_node[i] = ((uint32_t*)coll_args->args.src.info_v.displacements)[r];
        }
        if (rcount * ucc_dt_size(rdt) <= cl_lib->cfg.a2av_node_thresh) {
            rc_node[i] = 0;
        } else {
            rc_node[i] = rcount;
            rd_node[i] = ((uint32_t*)coll_args->args.dst.info_v.displacements)[r];
        }
    }

    args.args.src.info_v.counts = (ucc_aint_t*)sc_node;
    args.args.dst.info_v.counts = (ucc_aint_t*)rc_node;
    args.args.src.info_v.displacements = (ucc_aint_t*)sd_node;
    args.args.dst.info_v.displacements = (ucc_aint_t*)rd_node;
    status = ucc_coll_score_map_lookup(cl_team->sbgps[UCC_HIER_SBGP_NODE].score_map,
                                       &args, &init, &bteam);
    if (UCC_OK != status) {
        cl_error(team->context->lib, "failed to lookup full a2av task");
        ucc_free(sc_full);
        ucc_free(task_full->args.src.info_v.counts);
        ucc_collective_finalize(&task_full->super);
        ucc_free(schedule);
        return status;
    }

    status = init(&args, bteam, &task_node);
    if (UCC_OK != status) {
        cl_error(team->context->lib, "failed to init full a2av task");
        ucc_free(sc_full);
        ucc_free(task_full->args.src.info_v.counts);
        ucc_collective_finalize(&task_full->super);
        ucc_free(schedule);
        return status;
    }
    ucc_schedule_add_task(schedule, task_node);
    ucc_schedule_add_task(schedule, task_full);
    ucc_task_subscribe_dep(&schedule->super, task_node, UCC_EVENT_SCHEDULE_STARTED);
    ucc_task_subscribe_dep(&schedule->super, task_full, UCC_EVENT_SCHEDULE_STARTED);

    schedule->super.post     = ucc_cl_hier_alltoallv_post;
    schedule->super.progress = NULL;
    schedule->super.finalize = ucc_cl_hier_alltoallv_finalize;
    schedule->super.early_triggered_post = ucc_cl_hier_alltoallv_early_triggered_post;
    schedule->super.triggered_post = ucc_core_triggered_post;
    *task = &schedule->super;
    return UCC_OK;
}
