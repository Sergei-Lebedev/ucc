/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */
#ifndef UCC_SCHEDULE_PIPELINED_H_
#define UCC_SCHEDULE_PIPELINED_H_
#include "components/base/ucc_base_iface.h"

#define UCC_SCHEDULE_FRAG_MAX_TASKS 8

typedef struct ucc_schedule_pipelined ucc_schedule_pipelined_t;

#define UCC_SCHEDULE_PIPELINED_MAX_FRAGS 4

typedef ucc_status_t (*ucc_schedule_frag_init_fn_t)(
    ucc_base_coll_args_t *coll_args, ucc_schedule_pipelined_t *schedule_p,
    ucc_base_team_t *team, ucc_schedule_t **frag);

typedef ucc_status_t (*ucc_schedule_frag_setup_fn_t)(
    ucc_schedule_pipelined_t *schedule_p, ucc_schedule_t *frag, int frag_num);

typedef struct ucc_schedule_pipelined {
    ucc_schedule_t               super;
    ucc_schedule_t *             frags[UCC_SCHEDULE_PIPELINED_MAX_FRAGS];
    int                          n_frags;
    int                          n_frags_started;
    int                          n_frags_in_pipeline;
    int                          sequential;
    int                          next_frag_to_post;
    ucc_schedule_frag_setup_fn_t frag_setup;
} ucc_schedule_pipelined_t;

ucc_status_t ucc_schedule_pipelined_init(
    ucc_base_coll_args_t *coll_args, ucc_base_team_t *team,
    ucc_schedule_frag_init_fn_t  frag_init,
    ucc_schedule_frag_setup_fn_t frag_setup, int n_frags, int n_frags_total,
    int sequential, ucc_schedule_pipelined_t *schedule_p);

ucc_status_t ucc_dependency_handler(ucc_coll_task_t *parent, /* NOLINT */
                                    ucc_coll_task_t *task);

ucc_status_t ucc_schedule_pipelined_post(ucc_coll_task_t *task);

ucc_status_t ucc_schedule_pipelined_finalize(ucc_coll_task_t *task);
#endif
