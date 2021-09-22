#include "../tl_cuda_ipc_coll.h"
#include "../tl_cuda_ipc_ring.h"

static inline uint32_t ucc_tl_cuda_ipc_get_send_block(ucc_rank_t trank,
                                                      ucc_rank_t tsize,
                                                      uint32_t step,
                                                      int ring_id)
{
    return dgx_map[ring_id][(dgx_imap[ring_id][trank] + tsize - step + 1) % tsize];
}

static inline uint32_t ucc_tl_cuda_ipc_get_recv_block(ucc_rank_t trank,
                                                      ucc_rank_t tsize,
                                                      uint32_t step,
                                                      int ring_id)
{
    return dgx_map[ring_id][(dgx_imap[ring_id][trank] + tsize - step) % tsize];
}

ucc_status_t ucc_tl_cuda_ipc_allgather_finalize(ucc_coll_task_t *coll_task)
{
    ucc_tl_cuda_ipc_task_t *task = ucc_derived_of(coll_task,
                                                  ucc_tl_cuda_ipc_task_t);
    ucc_tl_cuda_ipc_put_task(task);
    return UCC_OK;
}

ucc_status_t ucc_tl_cuda_ipc_allgather_progress(ucc_coll_task_t *coll_task)
{
    // void             *dbuf     = coll_task->args.dst.info.buffer;
    // size_t block_count, block_offset, frag_count, frag_offset, recv_block;
    // size_t remote_offset, local_offset;

    ucc_tl_cuda_ipc_schedule_t *schedule = ucc_derived_of(coll_task->schedule,
                                                          ucc_tl_cuda_ipc_schedule_t);
    ucc_tl_cuda_ipc_task_t     *task     = ucc_derived_of(coll_task,
                                                          ucc_tl_cuda_ipc_task_t);
    ucc_tl_cuda_ipc_team_t *team      = TASK_TEAM(task);
    size_t                  ccount    = coll_task->args.dst.info.count / team->size;
    ucc_datatype_t          dt        = coll_task->args.dst.info.datatype;
    size_t                  data_size = ccount * ucc_dt_size(dt);
    const int         ring_id  = task->allgather.ring_id;
    const ucc_rank_t  trank    = team->rank;
    const ucc_rank_t  tsize    = team->size;
    const uint32_t    coll_id  = task->allgather.coll_id;
    const ucc_rank_t  sendto   = ucc_ring_send_to(trank, tsize, ring_id);
    const ucc_rank_t  recvfrom = ucc_ring_recv_from(trank, tsize, ring_id);
    mem_info_t *peer_info;
    uint32_t send_step, recv_step, recv_block;
    ucc_ee_executor_task_args_t exec_args;
    size_t block_count, block_offset;
    ucc_status_t st;

    while(task->allgather.step < tsize) {
        if (task->allgather.exec_task != NULL) {
            st = ucc_ee_executor_task_test(task->allgather.exec_task);
            if (st == UCC_OK) {
                // printf("rank %d: step %d exec task done\n",
                //        (int)trank, task->allgather.step);
                UCC_TL_CUDA_IPC_PROFILE_REQUEST_EVENT(coll_task, "allgather_task_complete", 0);
                task->allgather.step++;
                ucc_tl_cuda_ipc_set_rank_step(team, coll_id, trank,
                                              task->allgather.step,
                                              ring_id);
                task->allgather.exec_task = NULL;

                continue;
            } else if (st > 0) {
                task->super.super.status = UCC_INPROGRESS;
                return task->super.super.status;
            } else {
                task->super.super.status = st;
                return st;
            }
        }
        send_step = ucc_tl_cuda_ipc_get_rank_step(team, coll_id, sendto,
                                                  ring_id);
        recv_step = ucc_tl_cuda_ipc_get_rank_step(team, coll_id, recvfrom,
                                                  ring_id);
        if ((recv_step < task->allgather.step) ||
            (send_step < task->allgather.step))
        {
            task->super.super.status = UCC_INPROGRESS;
            return UCC_INPROGRESS;
        }


        recv_block = ucc_tl_cuda_ipc_get_recv_block(trank, tsize, task->allgather.step, ring_id);
        block_count = ucc_ring_block_count(data_size, task->allgather.n_rings, ring_id);
        block_offset = ucc_ring_block_offset(data_size, task->allgather.n_rings, ring_id);

        // printf("rank %d: recv from %d, count %d, block %d\n",
        //        (int)trank, (int)recvfrom, (int)block_count, (int)recv_block);

        peer_info = GET_MEM_INFO(team, coll_id, recvfrom);
        exec_args.task_type     = UCC_MC_EE_EXECUTOR_TASK_TYPE_COPY;
        exec_args.src1.buffer   = PTR_OFFSET(task->allgather.peer_map_addr,
                                             data_size * recv_block +
                                             block_offset + peer_info->offset);
        exec_args.src1.count    = block_count;
        exec_args.dst.buffer    = PTR_OFFSET(coll_task->args.dst.info.buffer,
                                             data_size * recv_block +
                                             block_offset);
        exec_args.dst.count    = block_count;
        UCC_TL_CUDA_IPC_PROFILE_REQUEST_EVENT(coll_task, "cuda_ipc_allgather_task_post", 0);
        st = ucc_ee_executor_task_post(&exec_args,
                                       &task->allgather.exec_task,
                                       schedule->allgather.eee);
        if (ucc_unlikely(st != UCC_OK)) {
            task->super.super.status = st;
            return st;
        }
        return task->super.super.status;
    }
    UCC_TL_CUDA_IPC_PROFILE_REQUEST_EVENT(coll_task, "cuda_ipc_allgather_finish", 0);
    task->super.super.status = UCC_OK;
    return task->super.super.status;
}

ucc_status_t ucc_tl_cuda_ipc_allgather_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_cuda_ipc_schedule_t *schedule = ucc_derived_of(coll_task->schedule,
                                                          ucc_tl_cuda_ipc_schedule_t);
    ucc_tl_cuda_ipc_task_t *task      = ucc_derived_of(coll_task,
                                                       ucc_tl_cuda_ipc_task_t);
    ucc_tl_cuda_ipc_team_t *team      = TASK_TEAM(task);
    ucc_rank_t              trank     = team->rank;
    size_t                  ccount    = coll_task->args.dst.info.count / team->size;
    ucc_datatype_t          dt        = coll_task->args.dst.info.datatype;
    size_t                  data_size = ccount * ucc_dt_size(dt);
    size_t block_count, block_offset;
    ucc_ee_executor_task_args_t exec_args;
    ucc_status_t st;
    const uint32_t    coll_id  = task->allgather.coll_id;
    const int         ring_id  = task->allgather.ring_id;
    task->allgather.exec_task = NULL;

    UCC_TL_CUDA_IPC_PROFILE_REQUEST_EVENT(coll_task, "cuda_ipc_allgather_start", 0);
    if (!UCC_IS_INPLACE(coll_task->args)) {
        task->allgather.step      = 0;
        block_count = ucc_ring_block_count(data_size, task->allgather.n_rings,
                                           task->allgather.ring_id);
        block_offset = ucc_ring_block_offset(data_size, task->allgather.n_rings,
                                             task->allgather.ring_id);
        exec_args.task_type   = UCC_MC_EE_EXECUTOR_TASK_TYPE_COPY;
        exec_args.src1.buffer = PTR_OFFSET(coll_task->args.src.info.buffer,
                                           block_offset);
        exec_args.src1.count  = block_count;
        exec_args.dst.buffer  = PTR_OFFSET(coll_task->args.dst.info.buffer,
                                           data_size * trank + block_offset);
        exec_args.dst.count   = block_count;

        st = ucc_ee_executor_task_post(&exec_args, &task->allgather.exec_task,
                                       schedule->allgather.eee);
        if (ucc_unlikely(st != UCC_OK)) {
            task->super.super.status = st;
            return st;
        }
    } else {
        task->allgather.step      = 1;
    }
    ucc_tl_cuda_ipc_set_rank_step(team, coll_id, trank,
                                  task->allgather.step,
                                  ring_id);

    st = ucc_tl_cuda_ipc_allgather_progress(coll_task);
    if (UCC_INPROGRESS == task->super.super.status) {
        ucc_progress_enqueue(UCC_TL_CORE_CTX(team)->pq, &task->super);
        return UCC_OK;
    }

    return ucc_task_complete(coll_task);
}

ucc_status_t
ucc_tl_cuda_ipc_allgather_ring_sched_post(ucc_coll_task_t *coll_task)
{
    ucc_tl_cuda_ipc_schedule_t *schedule = ucc_derived_of(coll_task, ucc_tl_cuda_ipc_schedule_t);
    ucc_tl_cuda_ipc_team_t *team = ucc_derived_of(schedule->super.super.team, ucc_tl_cuda_ipc_team_t);
    ucc_ee_executor_params_t exec_params;
    ucc_status_t st;
    int i;

    for (i = 0 ; i < schedule->super.n_tasks; i++) {
        schedule->super.tasks[i]->args.src = schedule->super.super.args.src;
        schedule->super.tasks[i]->args.dst = schedule->super.super.args.dst;
    }
    if (!schedule->eee_external) {
        exec_params.ee_type = UCC_EE_CUDA_STREAM;
        exec_params.ee_context = team->stream;
        st = ucc_ee_executor_create_post(&exec_params, &schedule->allgather.eee);
        if (ucc_unlikely(st != UCC_OK)) {
            ucc_error("failed to create ee executor");
            return st;
        }

        //TODO: make nonblocking?
        do {
            st = ucc_ee_executor_create_test(schedule->allgather.eee);
        } while (st == UCC_INPROGRESS);

        if (ucc_unlikely(st != UCC_OK)) {
            ucc_error("failed to create ee executor");
            return st;
        }
    }
    return ucc_schedule_start(&schedule->super);
}

ucc_status_t
ucc_tl_cuda_ipc_allgather_ring_sched_finalize(ucc_coll_task_t *task)
{
    ucc_tl_cuda_ipc_schedule_t *schedule = ucc_derived_of(task, ucc_tl_cuda_ipc_schedule_t);
    ucc_status_t status;

//TODO: move to completed handler
    if (!schedule->eee_external) {
        ucc_ee_executor_destroy((ucc_ee_executor_t*)schedule->allgather.eee);
    }
    status = ucc_schedule_finalize(task);
    ucc_tl_cuda_ipc_put_schedule(schedule);
    return status;
}

ucc_status_t ucc_tl_cuda_ipc_allgather_init(ucc_base_coll_args_t *coll_args,
                                            ucc_base_team_t *tl_team,
                                            ucc_coll_task_t **task_p)
{
    ucc_tl_cuda_ipc_team_t     *team     = ucc_derived_of(tl_team,
                                                          ucc_tl_cuda_ipc_team_t);
    ucc_tl_cuda_ipc_schedule_t *schedule = ucc_tl_cuda_ipc_get_schedule(&coll_args->args,
                                                                        team);
    size_t         ccount    = coll_args->args.dst.info.count;
    ucc_datatype_t dt        = coll_args->args.dst.info.datatype;
    const uint32_t n_rings   = UCC_TL_CUDA_IPC_TEAM_LIB(team)->cfg.num_rings;
    const uint32_t max_colls = UCC_TL_CUDA_IPC_TEAM_LIB(team)->cfg.max_concurrent;
    ucc_tl_cuda_ipc_task_t *task;
    int i;

    if (coll_args->mask & UCC_BASE_COLL_ARGS_FIELD_EEE) {
        schedule->allgather.eee = coll_args->eee;
        schedule->eee_external = 1;
    } else {
        schedule->eee_external = 0;
    }

    for (i = 0; i < n_rings; i++) {
        task = ucc_tl_cuda_ipc_init_task(coll_args, tl_team);
        if (!task) {
            return UCC_ERR_NO_MEMORY;
        }
        task->allgather.coll_id = task->seq_num % max_colls;
        task->allgather.ring_id = i;
        task->allgather.n_rings = n_rings;
        task->super.post        = ucc_tl_cuda_ipc_allgather_start;
        task->super.progress    = ucc_tl_cuda_ipc_allgather_progress;
        task->super.finalize    = ucc_tl_cuda_ipc_allgather_finalize;
        ucc_tl_cuda_ipc_ring_setup(&task->super, task->allgather.ring_id,
                                   task->allgather.coll_id,
                                   coll_args->args.dst.info.buffer,
                                   ccount * ucc_dt_size(dt),
                                   &task->allgather.peer_map_addr);
        ucc_schedule_add_task(&schedule->super, &task->super);
        ucc_event_manager_subscribe(&schedule->super.super.em, UCC_EVENT_SCHEDULE_STARTED,
                                    &task->super, ucc_task_start_handler);
    }
    schedule->super.super.post     = ucc_tl_cuda_ipc_allgather_ring_sched_post;
    schedule->super.super.finalize = ucc_tl_cuda_ipc_allgather_ring_sched_finalize;

    *task_p = &schedule->super.super;
    return UCC_OK;
}
