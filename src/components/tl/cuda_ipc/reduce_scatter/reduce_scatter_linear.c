#include "../tl_cuda_ipc_coll.h"

static inline size_t ucc_reduce_scatter_linear_block_offset(size_t total_count,
                                                            ucc_rank_t n_blocks,
                                                            ucc_rank_t block)
{
    size_t block_count = total_count / n_blocks;
    size_t left        = total_count % n_blocks;
    size_t offset      = block * block_count + left;
    return (block < left) ? offset - (left - block) : offset;
}

static inline size_t ucc_reduce_scatter_linear_block_count(size_t total_count,
                                                           ucc_rank_t n_blocks,
                                                           ucc_rank_t block)
{
    size_t block_count = total_count / n_blocks;
    size_t left        = total_count % n_blocks;
    return (block < left) ? block_count + 1 : block_count;
}

#define NUM_POSTS team->size
ucc_status_t
ucc_tl_cuda_ipc_reduce_scatter_linear_progress(ucc_coll_task_t *coll_task)
{
    ucc_tl_cuda_ipc_schedule_t *schedule = ucc_derived_of(coll_task->schedule,
                                                          ucc_tl_cuda_ipc_schedule_t);
    ucc_tl_cuda_ipc_task_t     *task     = ucc_derived_of(coll_task,
                                                          ucc_tl_cuda_ipc_task_t);
    ucc_tl_cuda_ipc_team_t *team     = TASK_TEAM(task);
    uint32_t                coll_id   = task->reduce_scatter_linear.coll_id;
    size_t                  ccount    = coll_task->args.src.info.count / team->size;
    ucc_datatype_t          dt        = coll_task->args.src.info.datatype;
    size_t                  data_size = ccount * ucc_dt_size(dt);
    ucc_rank_t              num_done = 0;
    ucc_ee_executor_task_args_t exec_args;
    size_t block_offset, block_count;
    ucc_rank_t i, peer;
    mem_info_t *peer_info, *my_info;
    ucc_status_t st;


    if (task->reduce_scatter_linear.exec_task[0] == NULL) {
        for (peer = 0; peer < team->size; peer++) {
            if (GET_MEM_INFO(team, coll_id, peer)->seq_num[1] != task->seq_num) {
                task->super.super.status = UCC_INPROGRESS;
                goto exit;
            }
        }
        for (i = 0 ; i < NUM_POSTS; i++) {
            block_count = ucc_reduce_scatter_linear_block_count(ccount,
                                                                NUM_POSTS, i);
            block_offset = ucc_reduce_scatter_linear_block_offset(ccount,
                                                                NUM_POSTS, i) * ucc_dt_size(dt);
            exec_args.task_type    = UCC_MC_EE_EXECUTOR_TASK_TYPE_REDUCE_MULTI;
            exec_args.dst.buffer   = PTR_OFFSET(coll_task->args.dst.info.buffer, block_offset);
            exec_args.dst.count    = block_count;
            exec_args.dst.datatype = UCC_DT_FLOAT32;
            exec_args.src3_size    = team->size;
            for (peer = 0; peer < team->size; peer++) {
                void *src;
                if (peer == team->rank) {
                    src = PTR_OFFSET(coll_task->args.src.info.buffer, data_size * team->rank);
                } else {
                    peer_info = GET_MEM_INFO(team, coll_id, peer);
                    src = PTR_OFFSET(task->reduce_scatter_linear.peer_map_addr[peer],
                                    peer_info->offset + data_size * team->rank);
                }
                exec_args.src3[peer] = PTR_OFFSET(src, block_offset);
            }
            st = ucc_ee_executor_task_post(&exec_args,
                                           &task->reduce_scatter_linear.exec_task[i],
                                           schedule->eee);
            if (ucc_unlikely(st != UCC_OK)) {
                task->super.super.status = st;
                goto exit;
            }
        }
    }

    for (i = 0; i < NUM_POSTS; i++) {
        st = ucc_ee_executor_task_test(task->reduce_scatter_linear.exec_task[i]);
        if (st != UCC_OK) {
            task->super.super.status = UCC_INPROGRESS;
            goto exit;
        }
    }

    my_info = GET_MEM_INFO(team, coll_id, team->rank);
    __sync_synchronize();
    asm volatile("": : :"memory");
    my_info->seq_num[2] = task->seq_num;

    num_done = 0;
    for (i = 0; i < team->size; i++) {
        mem_info_t *pi = GET_MEM_INFO(team, coll_id, i);
        if (pi->seq_num[2] == task->seq_num) {
            num_done++;
        }
    }

    task->super.super.status = (num_done == team->size) ? UCC_OK: UCC_INPROGRESS;
exit:
    return task->super.super.status;
}

ucc_status_t
ucc_tl_cuda_ipc_reduce_scatter_linear_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_cuda_ipc_task_t *task = ucc_derived_of(coll_task,
                                                  ucc_tl_cuda_ipc_task_t);
    ucc_tl_cuda_ipc_team_t *team = TASK_TEAM(task);
    mem_info_t             *info = GET_MEM_INFO(team, task->reduce_scatter_linear.coll_id,
                                                team->rank);
    ucc_rank_t r;

    coll_task->super.status = UCC_INPROGRESS;

    for (r = 0; r < team->size; r++) {
        task->reduce_scatter_linear.exec_task[r] = NULL;
    }
    __sync_synchronize();
    asm volatile("": : :"memory");
    info->seq_num[1] = task->seq_num;

    ucc_tl_cuda_ipc_reduce_scatter_linear_progress(coll_task);
    if (UCC_INPROGRESS == task->super.super.status) {
        ucc_progress_enqueue(UCC_TL_CORE_CTX(team)->pq, &task->super);
        return UCC_OK;
    }

    return ucc_task_complete(coll_task);
}

ucc_status_t
ucc_tl_cuda_ipc_reduce_scatter_linear_finalize(ucc_coll_task_t *coll_task)
{
    ucc_tl_cuda_ipc_task_t *task = ucc_derived_of(coll_task,
                                                  ucc_tl_cuda_ipc_task_t);
    ucc_tl_cuda_ipc_team_t *team = TASK_TEAM(task);

    if (team->size > MAX_STATIC_SIZE) {
        ucc_free(task->reduce_scatter_linear.peer_map_addr);
    }
    ucc_tl_cuda_ipc_put_task(task);
    return UCC_OK;
}

ucc_status_t
ucc_tl_cuda_ipc_reduce_scatter_linear_setup(ucc_coll_task_t *coll_task)
{
    ucc_tl_cuda_ipc_task_t *task      = ucc_derived_of(coll_task, ucc_tl_cuda_ipc_task_t);
    ucc_tl_cuda_ipc_team_t *team      = TASK_TEAM(task);
    size_t                  ccount    = coll_task->args.src.info.count;
    ucc_datatype_t          dt        = coll_task->args.src.info.datatype;
    size_t                  data_size = ccount * ucc_dt_size(dt);
    void *data_buf, *mapped_addr, *base_address;
    ucc_status_t            status;
    size_t                  alloc_length;
    int                     i, coll_id;
    mem_info_t             *my_info;
    uint32_t                max_concurrent;
    ucc_cuda_ipc_cache_t   *cache;

    data_buf = coll_task->args.src.info.buffer;
    max_concurrent = UCC_TL_CUDA_IPC_TEAM_LIB(team)->cfg.max_concurrent;
    coll_id = (task->seq_num % max_concurrent);
    my_info = GET_MEM_INFO(team, coll_id, team->rank);
    ucc_tl_cuda_ipc_get_alloc_info(data_buf, data_size,  &base_address, &alloc_length);

    if (base_address != NULL) {
        CUDACHECK(cudaIpcGetMemHandle((cudaIpcMemHandle_t *) &my_info->handle, base_address));
    }

    my_info->d_ptr  = base_address;
    my_info->size   = alloc_length;
    my_info->offset = data_buf - base_address;

    __sync_synchronize();
    asm volatile("": : :"memory");
    my_info->seq_num[0] = task->seq_num;
    volatile mem_info_t *pi;
    for (i = 0; i < team->size; i++) {
        pi = GET_MEM_INFO(team, coll_id, i);
        while (pi->seq_num[0] != task->seq_num);
    }

    for (i= 0 ; i < team->size; i++) {
        pi = GET_MEM_INFO(team, coll_id, i);
        if (i != team->rank && pi->d_ptr) {
            cache = ucc_cuda_ipc_get_cache(team, i);
            if (ucc_unlikely(!cache)) {
                return UCC_ERR_NO_MESSAGE;
            }
            status = ucc_cuda_ipc_map_memhandle(pi->d_ptr, pi->size,
                                                pi->handle, &mapped_addr,
                                                cache);
            if (UCC_OK != status) {
                ucc_error("ucc_cuda_ipc_map_memhandle failed");
                return UCC_ERR_INVALID_PARAM;
            }
            task->reduce_scatter_linear.peer_map_addr[i] = mapped_addr;
        }
    }
    task->reduce_scatter_linear.coll_id  = coll_id;
    return UCC_OK;
}

ucc_status_t
ucc_tl_cuda_ipc_reduce_scatter_linear_sched_post(ucc_coll_task_t *coll_task)
{
    ucc_tl_cuda_ipc_schedule_t *schedule = ucc_derived_of(coll_task, ucc_tl_cuda_ipc_schedule_t);
    ucc_tl_cuda_ipc_team_t *team = ucc_derived_of(schedule->super.super.team, ucc_tl_cuda_ipc_team_t);
    ucc_status_t st;

    if (!schedule->eee_external) {
        st = ucc_ee_executor_start(schedule->eee,
                                   (void*)team->stream);
        if (ucc_unlikely(st != UCC_OK)) {
            ucc_error("failed to start ee executor");
            return st;
        }

        do {
            st = ucc_ee_executor_status(schedule->eee);
        } while (st == UCC_INPROGRESS);
        if (ucc_unlikely(st != UCC_OK)) {
            ucc_error("failed to start ee executor");
            return st;
        }
    }
    return ucc_schedule_start(&schedule->super);
}

void ucc_tl_cuda_ipc_reduce_scatter_linear_sched_done(void *data, ucc_status_t status)
{
    ucc_tl_cuda_ipc_schedule_t *schedule = (ucc_tl_cuda_ipc_schedule_t*)data;

    if (!schedule->eee_external) {
        ucc_ee_executor_stop(schedule->eee);
    }
}

ucc_status_t
ucc_tl_cuda_ipc_reduce_scatter_linear_sched_finalize(ucc_coll_task_t *task)
{
    ucc_tl_cuda_ipc_schedule_t *schedule = ucc_derived_of(task, ucc_tl_cuda_ipc_schedule_t);
    ucc_status_t status;

//TODO: move to completed handler
    if (!schedule->eee_external) {
        ucc_ee_executor_free(schedule->eee);
    }
    status = ucc_schedule_finalize(task);
    ucc_tl_cuda_ipc_put_schedule(schedule);
    return status;
}

ucc_status_t
ucc_tl_cuda_ipc_reduce_scatter_linear_init(ucc_base_coll_args_t *coll_args,
                                           ucc_base_team_t *tl_team,
                                           ucc_coll_task_t **task_p)
{
    ucc_tl_cuda_ipc_team_t     *team     = ucc_derived_of(tl_team,
                                                          ucc_tl_cuda_ipc_team_t);
    ucc_tl_cuda_ipc_schedule_t *schedule = ucc_tl_cuda_ipc_get_schedule(&coll_args->args,
                                                                        team);
    ucc_tl_cuda_ipc_task_t *task;
    ucc_ee_executor_params_t exec_params;
    ucc_status_t status;

    if (UCC_IS_INPLACE(coll_args->args)) {
        ucc_tl_cuda_ipc_put_schedule(schedule);
        return UCC_ERR_NOT_SUPPORTED;
    }
    if (coll_args->mask & UCC_BASE_COLL_ARGS_FIELD_EEE) {
        schedule->eee_external = 1;
        schedule->eee = coll_args->eee;
    } else {
        schedule->eee_external = 0;
        exec_params.ee_type = UCC_EE_CUDA_STREAM;
        status = ucc_ee_executor_init(&exec_params,
                                      &schedule->eee);
        if (ucc_unlikely(status != UCC_OK)) {
            tl_error(UCC_TL_TEAM_LIB(team), "failed to init ee executor");
            goto free_schedule;
            return status;
        }
        schedule->super.super.flags |=  UCC_COLL_TASK_FLAG_CB2;
        schedule->super.super.cb2.cb = ucc_tl_cuda_ipc_reduce_scatter_linear_sched_done;
        schedule->super.super.cb2.data = (void*)schedule;
    }
    task = ucc_tl_cuda_ipc_init_task(coll_args, tl_team);
    if (!task) {
        return UCC_ERR_NO_MEMORY;
    }

    if (team->size <= MAX_STATIC_SIZE) {
        task->reduce_scatter_linear.peer_map_addr = task->data;
    } else {
        task->reduce_scatter_linear.peer_map_addr = ucc_malloc(sizeof(void*) *
                                                               team->size);
        if (!task->reduce_scatter_linear.peer_map_addr) {
            tl_error(UCC_TL_TEAM_LIB(team),
                     "failed to allocate %zd bytes for peer_map_addr",
                     sizeof(void*) * team->size);
            ucc_tl_cuda_ipc_put_task(task);
            return UCC_ERR_NO_MEMORY;
        }
    }
    task->super.post     = ucc_tl_cuda_ipc_reduce_scatter_linear_start;
    task->super.progress = ucc_tl_cuda_ipc_reduce_scatter_linear_progress;
    task->super.finalize = ucc_tl_cuda_ipc_reduce_scatter_linear_finalize;

    ucc_tl_cuda_ipc_reduce_scatter_linear_setup(&task->super);
    ucc_schedule_add_task(&schedule->super, &task->super);
    ucc_event_manager_subscribe(&schedule->super.super.em, UCC_EVENT_SCHEDULE_STARTED,
                                &task->super, ucc_task_start_handler);

    schedule->super.super.post     = ucc_tl_cuda_ipc_reduce_scatter_linear_sched_post;
    schedule->super.super.finalize = ucc_tl_cuda_ipc_reduce_scatter_linear_sched_finalize;

    *task_p = &schedule->super.super;
    return UCC_OK;

free_schedule:
    ucc_tl_cuda_ipc_put_schedule(schedule);
    return status;
}
