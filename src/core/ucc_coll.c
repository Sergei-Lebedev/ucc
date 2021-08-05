/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include "config.h"
#include "ucc_team.h"
#include "ucc_context.h"
#include "ucc_mc.h"
#include "components/cl/ucc_cl.h"
#include "utils/ucc_malloc.h"
#include "utils/ucc_log.h"
#include "utils/ucc_coll_utils.h"
#include "utils/profile/ucc_profile_core.h"
#include "schedule/ucc_schedule.h"
#include "ucc_ee.h"

#define UCC_BUFFER_INFO_CHECK_MEM_TYPE(_info) do {                             \
    if ((_info).mem_type == UCC_MEMORY_TYPE_UNKNOWN) {                         \
        ucc_mem_attr_t mem_attr;                                               \
        ucc_status_t st;                                                       \
        mem_attr.field_mask = UCC_MEM_ATTR_FIELD_MEM_TYPE;                     \
        st = ucc_mc_get_mem_attr((_info).buffer, &mem_attr);                   \
        if (ucc_unlikely(st != UCC_OK)) {                                      \
            return st;                                                         \
        }                                                                      \
        (_info).mem_type = mem_attr.mem_type;                                  \
    }                                                                          \
} while(0)

#define UCC_BUFFER_INFO_CHECK_DATATYPE(_info1, _info2) do {                    \
    if ((_info1).datatype != (_info2).datatype) {                              \
        ucc_error("datatype missmatch");                                       \
        return UCC_ERR_INVALID_PARAM;                                          \
    }                                                                          \
} while(0)

#define UCC_IS_ROOT(_args, _myrank) ((_args).root == (_myrank))

#if ENABLE_DEBUG == 1
static ucc_status_t ucc_check_coll_args(const ucc_coll_args_t *coll_args,
                                        ucc_rank_t rank)
{
    switch (coll_args->coll_type) {
    case UCC_COLL_TYPE_ALLREDUCE:
    case UCC_COLL_TYPE_REDUCE_SCATTER:
        if (!UCC_IS_INPLACE(*coll_args)) {
            UCC_BUFFER_INFO_CHECK_DATATYPE(coll_args->src.info,
                                           coll_args->dst.info);
        }
        break;
    case UCC_COLL_TYPE_REDUCE:
        if (!UCC_IS_INPLACE(*coll_args) && UCC_IS_ROOT(*coll_args, rank)) {
            UCC_BUFFER_INFO_CHECK_DATATYPE(coll_args->src.info,
                                           coll_args->dst.info);
        }
        break;
    default:
        return UCC_OK;
    }
    return UCC_OK;
}
#else
#define ucc_check_coll_args(...) UCC_OK
#endif

static ucc_status_t ucc_coll_args_check_mem_type(ucc_coll_args_t *coll_args,
                                                 ucc_rank_t rank)
{
    switch (coll_args->coll_type) {
    case UCC_COLL_TYPE_BARRIER:
    case UCC_COLL_TYPE_FANIN:
    case UCC_COLL_TYPE_FANOUT:
        return UCC_OK;
    case UCC_COLL_TYPE_BCAST:
        UCC_BUFFER_INFO_CHECK_MEM_TYPE(coll_args->src.info);
        return UCC_OK;
    case UCC_COLL_TYPE_ALLREDUCE:
        UCC_BUFFER_INFO_CHECK_MEM_TYPE(coll_args->dst.info);
        if (!UCC_IS_INPLACE(*coll_args)) {
            UCC_BUFFER_INFO_CHECK_MEM_TYPE(coll_args->src.info);
        } else {
            coll_args->src.info.mem_type = coll_args->dst.info.mem_type;
        }
        return UCC_OK;
    case UCC_COLL_TYPE_ALLGATHER:
    case UCC_COLL_TYPE_ALLTOALL:
    case UCC_COLL_TYPE_REDUCE_SCATTER:
        UCC_BUFFER_INFO_CHECK_MEM_TYPE(coll_args->dst.info);
        if (!UCC_IS_INPLACE(*coll_args)) {
            UCC_BUFFER_INFO_CHECK_MEM_TYPE(coll_args->src.info);
        }
        return UCC_OK;
    case UCC_COLL_TYPE_ALLGATHERV:
    case UCC_COLL_TYPE_REDUCE_SCATTERV:
        UCC_BUFFER_INFO_CHECK_MEM_TYPE(coll_args->dst.info_v);
        if (!UCC_IS_INPLACE(*coll_args)) {
            UCC_BUFFER_INFO_CHECK_MEM_TYPE(coll_args->src.info);
        }
        return UCC_OK;
    case UCC_COLL_TYPE_ALLTOALLV:
        UCC_BUFFER_INFO_CHECK_MEM_TYPE(coll_args->dst.info_v);
        if (!UCC_IS_INPLACE(*coll_args)) {
            UCC_BUFFER_INFO_CHECK_MEM_TYPE(coll_args->src.info_v);
        }
        return UCC_OK;
    case UCC_COLL_TYPE_GATHER:
    	// TODO: Check logic for Gather once implemented
    case UCC_COLL_TYPE_REDUCE:
        if (!UCC_IS_ROOT(*coll_args, rank)) {
            UCC_BUFFER_INFO_CHECK_MEM_TYPE(coll_args->src.info);
        } else {
            UCC_BUFFER_INFO_CHECK_MEM_TYPE(coll_args->dst.info);
            if (!UCC_IS_INPLACE(*coll_args)) {
                UCC_BUFFER_INFO_CHECK_MEM_TYPE(coll_args->src.info);
        	}
        }
        return UCC_OK;
    case UCC_COLL_TYPE_GATHERV:
        if (UCC_IS_ROOT(*coll_args, rank)) {
            UCC_BUFFER_INFO_CHECK_MEM_TYPE(coll_args->dst.info_v);
        }
        if (!(UCC_IS_INPLACE(*coll_args) && UCC_IS_ROOT(*coll_args, rank))) {
            UCC_BUFFER_INFO_CHECK_MEM_TYPE(coll_args->src.info);
        }
        return UCC_OK;
    case UCC_COLL_TYPE_SCATTER:
        if (UCC_IS_ROOT(*coll_args, rank)) {
            UCC_BUFFER_INFO_CHECK_MEM_TYPE(coll_args->src.info);
        }
        if (!(UCC_IS_INPLACE(*coll_args) && UCC_IS_ROOT(*coll_args, rank))) {
            UCC_BUFFER_INFO_CHECK_MEM_TYPE(coll_args->dst.info);
        }
        return UCC_OK;
    case UCC_COLL_TYPE_SCATTERV:
        if (UCC_IS_ROOT(*coll_args, rank)) {
            UCC_BUFFER_INFO_CHECK_MEM_TYPE(coll_args->src.info_v);
        }
        if (!(UCC_IS_INPLACE(*coll_args) && UCC_IS_ROOT(*coll_args, rank))) {
            UCC_BUFFER_INFO_CHECK_MEM_TYPE(coll_args->dst.info);
        }
        return UCC_OK;
    default:
        ucc_error("unknown collective type");
        return UCC_ERR_INVALID_PARAM;
    };
}

UCC_CORE_PROFILE_FUNC(ucc_status_t, ucc_collective_init,
                      (coll_args, request, team), ucc_coll_args_t *coll_args,
                      ucc_coll_req_h *request, ucc_team_h team)
{
    ucc_coll_task_t       *task;
    ucc_base_coll_args_t   op_args;
    ucc_status_t           status;
    ucc_base_coll_init_fn_t init;
    ucc_base_team_t        *bteam;

    status = ucc_coll_args_check_mem_type(coll_args, team->rank);
    if (ucc_unlikely(status != UCC_OK)) {
        ucc_error("memory type detection failed");
        return status;
    }
    status = ucc_check_coll_args(coll_args, team->rank);
    if (ucc_unlikely(status != UCC_OK)) {
        ucc_error("collective arguments check failed");
        return status;
    }

    /* TO discuss: maybe we want to pass around user pointer ? */
    op_args.mask = 0;
    memcpy(&op_args.args, coll_args, sizeof(ucc_coll_args_t));
    op_args.team = team;

    status =
        ucc_coll_score_map_lookup(team->score_map, &op_args, &init, &bteam);
    if (UCC_OK != status) {
        return status;
    }

    status = init(&op_args, bteam, &task);
    if (UCC_ERR_NOT_SUPPORTED == status) {
        ucc_debug("failed to init collective: not supported");
        return status;
    } else if (ucc_unlikely(status < 0)) {
        ucc_error("failed to init collective: %s", ucc_status_string(status));
        return status;
    }
    if (coll_args->mask & UCC_COLL_ARGS_FIELD_CB) {
        task->cb = coll_args->cb;
        task->flags |= UCC_COLL_TASK_FLAG_CB;
    }
    if (ucc_global_config.log_component.log_level >= UCC_LOG_LEVEL_DEBUG) {
        char coll_debug_str[256];
        ucc_coll_str(&op_args, coll_debug_str, sizeof(coll_debug_str));
        ucc_debug("coll_init: %s, req %p", coll_debug_str, task);
    }
    *request = &task->super;
    return UCC_OK;
}

ucc_status_t ucc_collective_post(ucc_coll_req_h request)
{
    ucc_coll_task_t *task = ucc_derived_of(request, ucc_coll_task_t);

    ucc_debug("coll_post: req %p", task);
    return task->post(task);
}

ucc_status_t ucc_collective_triggered_post(ucc_ee_h ee, ucc_ev_t *ev)
{
    ucc_coll_task_t *task = ucc_derived_of(ev->req, ucc_coll_task_t);

    ucc_debug("coll_triggered_post: req %p", task);
    task->ee = ee;
    return task->triggered_post(ee, ev, task);
}

UCC_CORE_PROFILE_FUNC(ucc_status_t, ucc_collective_finalize, (request),
                      ucc_coll_req_h request)
{
    ucc_coll_task_t *task = ucc_derived_of(request, ucc_coll_task_t);

    ucc_debug("coll_finalize: req %p", task);
    return task->finalize(task);
}




static ucc_status_t
ucc_triggered_coll_complete(ucc_coll_task_t *parent_task, //NOLINT
                            ucc_coll_task_t *coll_task)
{
    ucc_trace_req("triggered collective complete, task %p", coll_task);
    return ucc_mc_ee_task_end(coll_task->ee_task, coll_task->ee->ee_type);
}

static ucc_status_t
ucc_event_trigger_complete(ucc_coll_task_t *parent_task,
                           ucc_coll_task_t *coll_task)
{
    ucc_status_t status;

    ucc_trace_req("event triggered. ev_task:%p coll_task:%p",
                  parent_task, coll_task);

    coll_task->ee_task = parent_task->ee_task;
    status             = coll_task->post(coll_task);
    if (ucc_unlikely(status != UCC_OK)) {
        ucc_error("failed post the triggered collecitve, task %p", coll_task);
        return status;
    }

    if (coll_task->super.status == UCC_OK) {
        return ucc_triggered_coll_complete(coll_task, coll_task);
    } else {
        ucc_assert(coll_task->super.status == UCC_INPROGRESS);
        if (coll_task->ee_task) {
            ucc_event_manager_init(&coll_task->em);
            ucc_event_manager_subscribe(&coll_task->em, UCC_EVENT_COMPLETED,
                                        coll_task,
                                        ucc_triggered_coll_complete);
        }
    }
    return UCC_OK;
}

static ucc_status_t ucc_wait_for_event_trigger(ucc_coll_task_t *coll_task)
{
    ucc_context_t     *ctx = coll_task->team->context->ucc_context;
    ucc_ev_t          *post_event, *ev;
    ucc_status_t       status;

    if (coll_task->ev == NULL) {
        if (coll_task->ee->ee_type == UCC_EE_CUDA_STREAM) {
            /* implicit event triggered */
            coll_task->ev = (ucc_ev_t *) 0xFFFF; /* dummy event */
            coll_task->ee_task = NULL;
        } else if (UCC_OK == ucc_ee_get_event_internal(coll_task->ee, &ev,
                                                &coll_task->ee->event_in_queue)) {
            ucc_trace_req("triggered event arrived. ev_task: %p", coll_task);
            coll_task->ev      = ev;
            coll_task->ee_task = NULL;
        } else {
            return UCC_OK;
        }
    }

    if (coll_task->ee_task == NULL) {
        /*
         * run early triggered post if it's there
         * currently only alltoallv supports it and skip for all other collectives
         */
        if (ctx->triggered_overlap) {
            status = ucc_mc_ee_task_enqueue(coll_task->ee->ee_context,
                                            coll_task->ee->ee_type,
                                            &coll_task->ee_task);
            if (ucc_unlikely(status != UCC_OK)) {
                ucc_error("error in ee task enqueue");
                coll_task->super.status = status;
                return status;
            }
        }
        if (coll_task->triggered_task->early_triggered_post) {
            coll_task->triggered_task->ee = coll_task->ee;
            status = coll_task->triggered_task->early_triggered_post(coll_task->triggered_task);
            assert(status == UCC_OK);
        }

        if (ctx->triggered_overlap) {
            status = ucc_mc_ee_task_sync(coll_task->ee_task,
                                         coll_task->ee->ee_type);
        } else {
            status = ucc_mc_ee_task_post(coll_task->ee->ee_context,
                                         coll_task->ee->ee_type,
                                         &coll_task->ee_task);
        }
        if (ucc_unlikely(status != UCC_OK)) {
            ucc_error("error in ee task post");
            coll_task->super.status = status;
            return status;
        }

        /* TODO: mpool */
        post_event = ucc_malloc(sizeof(ucc_ev_t), "event");
        if (ucc_unlikely(post_event == NULL)) {
            ucc_error("failed to allocate memory for event");
            return UCC_ERR_NO_MEMORY;
        }

        post_event->ev_type         = UCC_EVENT_COLLECTIVE_POST;
        post_event->ev_context_size = 0;
        post_event->req             = &coll_task->triggered_task->super;
        ucc_ee_set_event_internal(coll_task->ee, post_event,
                                  &coll_task->ee->event_out_queue);
    }

    if (coll_task->ee_task == NULL ||
        (UCC_OK == ucc_mc_ee_task_query(coll_task->ee_task,
                                        coll_task->ee->ee_type))) {
        coll_task->super.status = UCC_OK;
    }

    return UCC_OK;
}

static ucc_status_t ucc_core_triggered_coll_finalize(ucc_coll_task_t *coll_task)
{
    ucc_trace("finalizing ev_task %p", coll_task);
    ucc_free(coll_task);
    return UCC_OK;
}

ucc_status_t ucc_core_triggered_post(ucc_ee_h ee, ucc_ev_t *ev, //NOLINT
                                     ucc_coll_task_t *coll_task)
{
    ucc_coll_task_t *ev_task;
    ucc_status_t     status;

    ev_task = ucc_malloc(sizeof(*ev_task));
    if (!ev_task) {
        ucc_error("failed to allocate %zd bytes for ev_task",
                  sizeof(*ev_task));
        return UCC_ERR_NO_MEMORY;
    }

    ucc_coll_task_init(ev_task, NULL, coll_task->team);
    ev_task->ee             = ee;
    ev_task->ev             = NULL;
    ev_task->triggered_task = coll_task;
    ev_task->flags          = UCC_COLL_TASK_FLAG_INTERNAL;
    ev_task->finalize       = ucc_core_triggered_coll_finalize;
    ev_task->super.status   = UCC_INPROGRESS;

    ucc_debug("triggered post, ev_task %p coll_task %p",
              &ev_task->super, coll_task);
    ev_task->progress = ucc_wait_for_event_trigger;
    ucc_event_manager_init(&ev_task->em);
    ucc_event_manager_subscribe(&ev_task->em, UCC_EVENT_COMPLETED, coll_task,
                                ucc_event_trigger_complete);

    status = ucc_wait_for_event_trigger(ev_task);
    if (ucc_unlikely(status != UCC_OK)) {
        return status;
    }
    if (ev_task->super.status == UCC_OK) {
        ucc_event_trigger_complete(ev_task, coll_task);
        ucc_core_triggered_coll_finalize(ev_task);
        return UCC_OK;
    }
    ucc_progress_enqueue(UCC_TASK_CORE_CTX(ev_task)->pq, ev_task);

    return UCC_OK;
}
