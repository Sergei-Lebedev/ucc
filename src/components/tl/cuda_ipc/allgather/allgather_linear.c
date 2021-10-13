#include "../tl_cuda_ipc_coll.h"

ucc_status_t ucc_tl_cuda_ipc_allgather_linear_progress(ucc_coll_task_t *coll_task)
{
    ucc_tl_cuda_ipc_task_t     *task     = ucc_derived_of(coll_task,
                                                          ucc_tl_cuda_ipc_task_t);
    ucc_tl_cuda_ipc_team_t *team     = TASK_TEAM(task);
    uint32_t                coll_id  = task->allgather_linear.coll_id;
    size_t                  ccount    = coll_task->args.dst.info.count / team->size;
    ucc_datatype_t          dt        = coll_task->args.dst.info.datatype;
    size_t                  data_size = ccount * ucc_dt_size(dt);
    uint32_t n_linear_tasks = UCC_TL_CUDA_IPC_TEAM_LIB(team)->cfg.linear_n_tasks;    
    ucc_rank_t              num_done = 0;
    ucc_rank_t              start = 0;
    ucc_rank_t i, peer, t;
    mem_info_t *peer_info, *my_info;
    ucc_status_t st;
    size_t offset, left, task_count;
    
    if (UCC_IS_INPLACE(coll_task->args)) {
        start = 1;
        num_done = 1;
    }

    for (i = start; i < team->size; i++) {
        peer = (team->rank + i) % team->size;
        if ((task->allgather_linear.exec_task[peer][0] == NULL) &&
            (GET_MEM_INFO(team, coll_id, peer)->seq_num[1] == task->seq_num)) {
            ucc_ee_executor_task_args_t exec_args;
            void                        *src, *dst;
            peer_info  = GET_MEM_INFO(team, coll_id, peer);
            if (peer == team->rank) {
                src = coll_task->args.src.info.buffer;
            } else {
                src = PTR_OFFSET(task->allgather_linear.peer_map_addr[peer],
                                 peer_info->offset);
            }
            dst = PTR_OFFSET(coll_task->args.dst.info.buffer, peer * data_size);

            for (t = 0; t < n_linear_tasks; t++) {
                task_count = ccount / n_linear_tasks;
                left = ccount % n_linear_tasks;
                offset = t * task_count + left;
                if (t < left) {
                    task_count++;
                    offset -= left - t;
                }
                offset *= ucc_dt_size(dt);

                exec_args.task_type   = UCC_MC_EE_EXECUTOR_TASK_TYPE_COPY;
                exec_args.src1.buffer = PTR_OFFSET(src, offset);
                exec_args.src1.count  = task_count * ucc_dt_size(dt);
                exec_args.dst.buffer  = PTR_OFFSET(dst, offset);
                exec_args.dst.count   = task_count * ucc_dt_size(dt);
                st = ucc_ee_executor_task_post(&exec_args,
                                               &task->allgather_linear.exec_task[peer][t],
                                               task->eee);
                if (ucc_unlikely(st != UCC_OK)) {
                    task->super.super.status = st;
                    goto exit;
                }
            }
        }

        int completed = 0;
        for (t = 0; t < n_linear_tasks; t++) {
            if (NULL != task->allgather_linear.exec_task[peer][t]) {
                if ((void*)0x1 != task->allgather_linear.exec_task[peer][t]) {
                    st = ucc_ee_executor_task_test(task->allgather_linear.exec_task[peer][t]);
                    if (st != UCC_OK) {
                        break;
                    }
                    task->allgather_linear.exec_task[peer][t] = (void*)0x1;
                }
                completed++;
            }
        }
        if (completed == n_linear_tasks) {
            num_done++;
        }
    }
    if (num_done == team->size) {
        my_info = GET_MEM_INFO(team, coll_id, team->rank);
        __sync_synchronize();
        asm volatile("": : :"memory");
        my_info->seq_num[2] = task->seq_num;
    }

    num_done = 0;
    for (i = 0; i < team->size; i++) {
        mem_info_t *pi = GET_MEM_INFO(team, coll_id, i);
        if (pi->seq_num[2] == task->seq_num) {
            num_done++;
        }
    }

    if (num_done == team->size) {
        nvtxRangeEnd(coll_task->id);
        UCC_TL_CUDA_IPC_PROFILE_REQUEST_EVENT(coll_task, "cuda_ipc_ag_linear_done", 0);
        task->super.super.status = UCC_OK;
    }

exit:
    return task->super.super.status;
}

ucc_status_t ucc_tl_cuda_ipc_allgather_linear_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_cuda_ipc_task_t *task = ucc_derived_of(coll_task,
                                                  ucc_tl_cuda_ipc_task_t);
    ucc_tl_cuda_ipc_team_t *team = TASK_TEAM(task);
    mem_info_t             *info = GET_MEM_INFO(team, task->allgather_linear.coll_id,
                                                team->rank);
    ucc_rank_t r, t;

    coll_task->super.status = UCC_INPROGRESS;
    coll_task->id = nvtxRangeStartA("ag_ipc_start");
    UCC_TL_CUDA_IPC_PROFILE_REQUEST_EVENT(coll_task, "cuda_ipc_ag_linear_start", 0);
    for (r = 0; r < team->size; r++) {
        for (t = 0; t < N_LINEAR_TASKS; t++) {
            task->allgather_linear.exec_task[r][t] = NULL;
        }
    }
    __sync_synchronize();
    asm volatile("": : :"memory");
    info->seq_num[1] = task->seq_num;

    ucc_tl_cuda_ipc_allgather_linear_progress(coll_task);
    if (UCC_INPROGRESS == task->super.super.status) {
        ucc_progress_enqueue(UCC_TL_CORE_CTX(team)->pq, &task->super);
        return UCC_OK;
    }

    return ucc_task_complete(coll_task);
}

ucc_status_t ucc_tl_cuda_ipc_allgather_linear_finalize(ucc_coll_task_t *coll_task)
{
    ucc_tl_cuda_ipc_task_t *task = ucc_derived_of(coll_task,
                                                  ucc_tl_cuda_ipc_task_t);
    ucc_tl_cuda_ipc_team_t *team = TASK_TEAM(task);

    if (team->size > MAX_STATIC_SIZE) {
        ucc_free(task->reduce_scatter_linear.peer_map_addr);
    }
//TODO: move to completed handler
    if (!task->eee_external) {
        ucc_ee_executor_free(task->eee);
    }

    ucc_tl_cuda_ipc_put_task(task);
    return UCC_OK;
}

ucc_status_t ucc_tl_cuda_ipc_allgather_linear_setup(ucc_coll_task_t *coll_task)
{
    ucc_tl_cuda_ipc_task_t *task      = ucc_derived_of(coll_task, ucc_tl_cuda_ipc_task_t);
    ucc_tl_cuda_ipc_team_t *team      = TASK_TEAM(task);
    ucc_rank_t              trank     = team->rank;
    size_t                  ccount    = coll_task->args.dst.info.count / team->size;
    ucc_datatype_t          dt        = coll_task->args.dst.info.datatype;
    size_t                  data_size = ccount * ucc_dt_size(dt);
    void *data_buf, *mapped_addr, *base_address;
    ucc_status_t            status;
    size_t                  alloc_length;
    int                     i, coll_id;
    mem_info_t             *my_info;
    uint32_t                max_concurrent;
    ucc_cuda_ipc_cache_t   *cache;

    if (UCC_IS_INPLACE(coll_task->args)) {
        data_buf = PTR_OFFSET(coll_task->args.dst.info.buffer, data_size * trank);
    } else {
        data_buf = coll_task->args.src.info.buffer;
    }

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
            task->allgather_linear.peer_map_addr[i] = mapped_addr;
        }
    }
    task->allgather_linear.coll_id  = coll_id;
    return UCC_OK;
}

ucc_status_t
ucc_tl_cuda_ipc_allgather_linear_post(ucc_coll_task_t *coll_task)
{
    ucc_tl_cuda_ipc_task_t *task = ucc_derived_of(coll_task, ucc_tl_cuda_ipc_task_t);
    ucc_tl_cuda_ipc_team_t *team = ucc_derived_of(task->super.team, ucc_tl_cuda_ipc_team_t);
    ucc_status_t st;

    if (!task->eee_external) {
        st = ucc_ee_executor_start(task->eee, (void*)team->stream);
        if (ucc_unlikely(st != UCC_OK)) {
            ucc_error("failed to start ee executor");
            return st;
        }

        do {
            st = ucc_ee_executor_status(task->eee);
        } while (st == UCC_INPROGRESS);
        if (ucc_unlikely(st != UCC_OK)) {
            ucc_error("failed to start ee executor");
            return st;
        }
    }
    return ucc_tl_cuda_ipc_allgather_linear_start(coll_task);
}

void ucc_tl_cuda_ipc_allgather_linear_done(void *data, ucc_status_t status)
{
    ucc_tl_cuda_ipc_task_t *task = (ucc_tl_cuda_ipc_task_t*)data;

    if (!task->eee_external) {
        ucc_ee_executor_stop(task->eee);
    }
}

ucc_status_t ucc_tl_cuda_ipc_allgather_linear_init(ucc_base_coll_args_t *coll_args,
                                                   ucc_base_team_t *tl_team,
                                                   ucc_coll_task_t **task_p)
{
    ucc_tl_cuda_ipc_team_t     *team     = ucc_derived_of(tl_team,
                                                          ucc_tl_cuda_ipc_team_t);
    ucc_tl_cuda_ipc_task_t *task;
    ucc_ee_executor_params_t exec_params;
    ucc_status_t status;

    task = ucc_tl_cuda_ipc_init_task(coll_args, tl_team);
    if (ucc_unlikely(!task)) {
        return UCC_ERR_NO_MEMORY;
    }

    
    if (coll_args->mask & UCC_BASE_COLL_ARGS_FIELD_EEE) {
        task->eee_external = 1;
        task->eee = coll_args->eee;
    } else {
        exec_params.ee_type = UCC_EE_CUDA_STREAM;
        status = ucc_ee_executor_init(&exec_params,
                                      &task->eee);
        if (ucc_unlikely(status != UCC_OK)) {
            tl_error(UCC_TL_TEAM_LIB(team), "failed to init ee executor");
            goto free_task;
        }
        task->super.flags |=  UCC_COLL_TASK_FLAG_CB2;
        task->super.cb2.cb = ucc_tl_cuda_ipc_allgather_linear_done;
        task->super.cb2.data = (void*)task;
    }

    if (team->size <= MAX_STATIC_SIZE) {
        task->allgather_linear.peer_map_addr = task->data;
    } else {
        task->allgather_linear.peer_map_addr = ucc_malloc(sizeof(void*) * team->size);
        if (!task->allgather_linear.peer_map_addr) {
            tl_error(UCC_TL_TEAM_LIB(team),
                     "failed to allocate %zd bytes for peer_map_addr",
                     sizeof(void*) * team->size);
            status = UCC_ERR_NO_MEMORY;
            goto free_task;
        }
    }

    task->super.progress = ucc_tl_cuda_ipc_allgather_linear_progress;
    task->super.post     = ucc_tl_cuda_ipc_allgather_linear_post;
    task->super.finalize = ucc_tl_cuda_ipc_allgather_linear_finalize;


    ucc_tl_cuda_ipc_allgather_linear_setup(&task->super);



    *task_p = &task->super;
    return UCC_OK;

free_task:
    if (!task->eee_external) {
        ucc_ee_executor_free(task->eee);
    }
    ucc_tl_cuda_ipc_put_task(task);
    
    return status;
}
