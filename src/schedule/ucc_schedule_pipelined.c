/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */
#include "ucc_schedule.h"
#include "ucc_schedule_pipelined.h"

ucc_status_t ucc_frag_start_handler(ucc_coll_task_t *parent, /* NOLINT */
                                    ucc_coll_task_t *task)
{
    ucc_schedule_pipelined_t *schedule = ucc_derived_of(parent, ucc_schedule_pipelined_t);
    ucc_schedule_frag_t *frag = ucc_derived_of(task, ucc_schedule_frag_t);
    ucc_status_t status;
    if(schedule->frag_setup) {
        status = schedule->frag_setup(schedule, frag, schedule->n_frags_started);
        if (UCC_OK != status) {
            ucc_error("failed to setup fragment %d of pipelined schedule",
                      schedule->n_frags_started);
            return status;
        }
    }
    schedule->n_frags_started++;
    return ucc_schedule_start(&frag->super);
}

ucc_status_t ucc_schedule_frag_init(ucc_schedule_frag_t *schedule, ucc_context_t *ctx)
{
    ucc_status_t status;
    status = ucc_schedule_init(&schedule->super, ctx);
    if (UCC_OK != status) {
        ucc_error("failed to init schedule");
        return status;
    }
    return UCC_OK;
}

void ucc_schedule_frag_add_task(ucc_schedule_frag_t *schedule, ucc_coll_task_t *task)
{
    schedule->tasks[schedule->super.n_tasks] = task;
    return ucc_schedule_add_task(&schedule->super, task);
}

static ucc_status_t
ucc_schedule_pipelined_completed_handler(ucc_coll_task_t *parent_task, //NOLINT
                                         ucc_coll_task_t *task)
{
    ucc_schedule_pipelined_t *self = ucc_container_of(task, ucc_schedule_pipelined_t, super);
    ucc_schedule_frag_t *frag = ucc_derived_of(parent_task, ucc_schedule_frag_t);

    self->super.n_completed_tasks += 1;
    self->n_frags_in_pipeline--;
    //  printf("completed frag %p, n_completed %d, n_started %d, n_total %d\n",
    //         frag, self->super.n_completed_tasks, self->n_frags_started, self->super.n_tasks);
    if (self->super.n_completed_tasks == self->super.n_tasks) {
        self->super.super.super.status = UCC_OK;
        ucc_event_manager_notify(&self->super.super, UCC_EVENT_COMPLETED);
    } else if (self->super.n_completed_tasks + self->n_frags_in_pipeline <
               self->super.n_tasks){
        /* need to post more frags*/
        int i;
        ucc_assert(frag == self->frags[0]);
        for (i = 1; i < self->n_frags; i++) {
            self->frags[i-1] = self->frags[i];
        }
        self->frags[i-1] = frag;
        frag->super.super.super.status   = UCC_OPERATION_INITIALIZED;
        frag->super.n_completed_tasks    = 0;
        for (i = 0; i<frag->super.n_tasks; i++) {
            frag->tasks[i]->n_deps += frag->tasks[i]->n_deps_orig;
            frag->tasks[i]->super.status = UCC_OPERATION_INITIALIZED;
        }
        self->n_frags_in_pipeline++;
        ucc_frag_start_handler(&self->super.super, &frag->super.super);
    }
    return UCC_OK;
}

static ucc_status_t ucc_schedule_pipelined_finalize(ucc_coll_task_t *task)
{
    int i, j;
    ucc_schedule_pipelined_t *schedule_p = ucc_derived_of(task, ucc_schedule_pipelined_t);
    ucc_schedule_frag_t **frags = schedule_p->frags;
    // printf("schedule pipelined %p is complete\n", schedule_p);

    for (i = 0; i < schedule_p->n_frags; i++) {
        for (j = 0; j < frags[i]->super.n_tasks; j++) {
            frags[i]->tasks[j]->finalize(frags[i]->tasks[j]);
        }
        schedule_p->frag_finalize(frags[i]);
    }
    ucc_free(schedule_p);

    return UCC_OK;
}

static ucc_status_t ucc_schedule_pipelined_post(ucc_coll_task_t *task)
{
    int i, j;
    ucc_schedule_pipelined_t *schedule_p = ucc_derived_of(task, ucc_schedule_pipelined_t);
    ucc_schedule_frag_t **frags = schedule_p->frags;

    schedule_p->super.super.super.status = UCC_OPERATION_INITIALIZED;
    schedule_p->super.n_completed_tasks = 0;
    schedule_p->n_frags_started         = 0;
    for (i = 0; i < schedule_p->n_frags; i++) {
        for (j = 0; j < frags[0]->super.n_tasks; j++) {
            frags[i]->tasks[j]->n_deps = frags[i]->tasks[j]->n_deps_orig;
            frags[i]->tasks[j]->n_deps_satisfied = 0;
            if (i == 0) {
                frags[0]->tasks[j]->n_deps_satisfied++;
            }
        }
    }

    return ucc_schedule_start(&schedule_p->super);
}

ucc_status_t ucc_schedule_pipelined_init(ucc_base_coll_args_t *coll_args,
                                         ucc_base_team_t *team,
                                         ucc_schedule_frag_init_fn_t frag_init,
                                         ucc_schedule_frag_finalize_fn_t  frag_finalize,
                                         ucc_schedule_frag_setup_fn_t frag_setup,
                                         int n_frags,
                                         int n_frags_total,
                                         ucc_schedule_pipelined_t **schedule_p)
{
    int i,j;
    ucc_status_t status;
    ucc_schedule_pipelined_t *schedule;
    ucc_schedule_frag_t **frags;
    //TODO use mpool
    schedule = ucc_malloc(sizeof(*schedule), "piplined_sched");
    if (!schedule) {
        ucc_error("failed to allocate %zd bytes for pipelined schedule",
                  sizeof(*schedule));
        return UCC_ERR_NO_MEMORY;
    }

    ucc_coll_task_init(&schedule->super.super);
    if (coll_args) {
        memcpy(&schedule->args, coll_args, sizeof(*coll_args));
    }
    schedule->super.super.handlers[UCC_EVENT_COMPLETED] =
        ucc_schedule_pipelined_completed_handler;
    schedule->super.n_tasks        = n_frags_total; //TODO compute
    schedule->n_frags              = n_frags;
    schedule->frag_finalize        = frag_finalize;
    schedule->frag_setup           = frag_setup;
    schedule->super.super.post     = ucc_schedule_pipelined_post;
    schedule->super.super.finalize = ucc_schedule_pipelined_finalize;
    frags                          = schedule->frags;
    for (i = 0; i < n_frags; i++) {
        status = frag_init(coll_args, team, &frags[i]);
        if (UCC_OK != status) {
            ucc_error("failed to initialize fragment for pipeline");
            goto err;
        }
    }
    for (i = 0; i < n_frags; i++) {
        if (i > 0) {
            for (j = 0; j < frags[i]->super.n_tasks; j++) {
                ucc_event_manager_subscribe(&frags[i-1]->tasks[j]->em,
                                            UCC_EVENT_COMPLETED, frags[i]->tasks[j]);
                frags[i]->tasks[j]->n_deps_orig = frags[i]->tasks[j]->n_deps + 1;
            }
        } else {
            for (j = 0; j < frags[0]->super.n_tasks; j++) {
                ucc_event_manager_subscribe(&frags[n_frags-1]->tasks[j]->em,
                                            UCC_EVENT_COMPLETED, frags[0]->tasks[j]);
                frags[0]->tasks[j]->n_deps_orig = frags[i]->tasks[j]->n_deps + 1;
            }
        }
        frags[i]->super.super.handlers[UCC_EVENT_SCHEDULE_STARTED] = ucc_frag_start_handler;
        ucc_event_manager_subscribe(&schedule->super.super.em, UCC_EVENT_SCHEDULE_STARTED,
                                    &frags[i]->super.super);
        ucc_event_manager_subscribe(&frags[i]->super.super.em, UCC_EVENT_COMPLETED,
                                    &schedule->super.super);

        /* printf("frag %p = [ %p %p ]\n", frags[i], frags[i]->tasks[0], frags[i]->tasks[1]); */
    }
    schedule->n_frags_in_pipeline = n_frags;
    *schedule_p = schedule;
    return UCC_OK;
err:
    for (i = i - 1; i >= 0; i--) {
        frag_finalize(frags[i]);
    }
    return status;
}

ucc_status_t ucc_dependency_handler(ucc_coll_task_t *parent, /* NOLINT */
                                    ucc_coll_task_t *task)
{
    task->n_deps_satisfied++;
    //  printf("task %p, n_deps %d, satisfied %d\n",
    //         task, task->n_deps, task->n_deps_satisfied);
    if (task->n_deps == task->n_deps_satisfied) {
        return task->post(task);
    }
    return UCC_OK;
}

ucc_status_t ucc_coll_task_init_dependent(ucc_coll_task_t *task, int n_deps)
{
    ucc_coll_task_init(task);
    task->handlers[UCC_EVENT_COMPLETED] = ucc_dependency_handler;
    task->handlers[UCC_EVENT_SCHEDULE_STARTED] = ucc_dependency_handler;
    task->n_deps_satisfied = 0;
    task->n_deps           = n_deps;
    return UCC_OK;
}
