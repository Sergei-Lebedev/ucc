#include "../tl_cuda_ipc_coll.h"


#define N_DGX_RINGS 2

static ucc_rank_t dgx_map[N_DGX_RINGS][8] = {
    {0, 3, 2, 1, 5, 6, 7, 4},
    {0, 4, 7, 6, 5, 1, 2, 3}
};

static ucc_rank_t dgx_imap[N_DGX_RINGS][8] = {
    {0, 3, 2, 1, 7, 4, 5, 6},
    {0, 5, 6, 7, 1, 4, 3, 2}
};

// static ucc_rank_t dgx_map[N_DGX_RINGS][8] = {
//     {0, 1, 2, 3, 4, 5, 6, 7},
//     {0, 1, 2, 3, 4, 5, 6, 7}
// };

// static ucc_rank_t dgx_imap[N_DGX_RINGS][8] = {
//     {0, 1, 2, 3, 4, 5, 6, 7},
//     {0, 1, 2, 3, 4, 5, 6, 7}
// };

static inline ucc_rank_t ucc_ring_send_to(ucc_rank_t rank, ucc_rank_t size,
                                          int ring_id) {
    return dgx_map[ring_id][(dgx_imap[ring_id][rank] + 1) % size];
}

static inline ucc_rank_t ucc_ring_recv_from(ucc_rank_t rank, ucc_rank_t size,
                                            int ring_id) {
    return dgx_map[ring_id][(dgx_imap[ring_id][rank] - 1 + size) % size];
}

static inline uint32_t ucc_tl_cuda_ipc_get_send_block(ucc_rank_t trank,
                                                      ucc_rank_t tsize,
                                                      uint32_t step,
                                                      int ring_id)
{
    return dgx_map[ring_id][(dgx_imap[ring_id][trank] + tsize - step) % tsize];
}

static inline uint32_t ucc_tl_cuda_ipc_get_recv_block(ucc_rank_t trank,
                                                      ucc_rank_t tsize,
                                                      uint32_t step,
                                                      int ring_id)
{
    return dgx_map[ring_id][(dgx_imap[ring_id][trank] + tsize - step - 1) % tsize];
}

static inline size_t ucc_ring_block_count(size_t total_count,
                                          ucc_rank_t n_blocks, ucc_rank_t block)
{
    size_t block_count = total_count / n_blocks;
    size_t left        = total_count % n_blocks;
    return (block < left) ? block_count + 1 : block_count;
}

static inline size_t ucc_ring_frag_count(size_t total_count,
                                         ucc_rank_t n_blocks, ucc_rank_t block,
                                         ucc_rank_t n_frags, ucc_rank_t frag)
{
    size_t block_count = ucc_ring_block_count(total_count, n_blocks, block);
    return ucc_ring_block_count(block_count, n_frags, frag);
}

static inline size_t ucc_ring_block_offset(size_t total_count,
                                           ucc_rank_t n_blocks, ucc_rank_t block)
{
    size_t block_count = total_count / n_blocks;
    size_t left        = total_count % n_blocks;
    size_t offset      = block * block_count + left;
    return (block < left) ? offset - (left - block) : offset;
}

static inline size_t ucc_ring_frag_offset(size_t block_count,
                                          ucc_rank_t n_frags, ucc_rank_t frag)
{
    return ucc_ring_block_offset(block_count, n_frags, frag);
}

static inline uint32_t ucc_tl_cuda_ipc_get_rank_step(ucc_tl_cuda_ipc_team_t *team,
                                                     uint32_t coll_id,
                                                     ucc_rank_t rank,
                                                     int ring_id)
{
    mem_info_t *info;
    info = GET_MEM_INFO(team, coll_id, rank);
    return info->seq_num[ring_id + 1];
}

static inline void ucc_tl_cuda_ipc_set_rank_step(ucc_tl_cuda_ipc_team_t *team,
                                                 uint32_t coll_id,
                                                 ucc_rank_t rank,
                                                 uint32_t step,
                                                 int ring_id)
{
    mem_info_t *info;
    info = GET_MEM_INFO(team, coll_id, rank);
    __sync_synchronize();
    asm volatile("": : :"memory");
    info->seq_num[ring_id + 1] = step;
}

ucc_status_t ucc_tl_cuda_ipc_reduce_scatter_finalize(ucc_coll_task_t *coll_task)
{
    ucc_tl_cuda_ipc_task_t *task = ucc_derived_of(coll_task,
                                                  ucc_tl_cuda_ipc_task_t);
    ucc_mc_free(task->reduce_scatter.scratch_mc_header);
    ucc_tl_cuda_ipc_put_task(task);
    return UCC_OK;
}

ucc_status_t ucc_tl_cuda_ipc_reduce_scatter_progress(ucc_coll_task_t *coll_task)
{
    ucc_tl_cuda_ipc_task_t     *task     = ucc_derived_of(coll_task,
                                                          ucc_tl_cuda_ipc_task_t);
    ucc_tl_cuda_ipc_schedule_t *schedule = ucc_derived_of(coll_task->schedule,
                                                          ucc_tl_cuda_ipc_schedule_t);
    ucc_tl_cuda_ipc_team_t *team     = TASK_TEAM(task);
    void             *scratch  = task->reduce_scatter.scratch;
    void             *sbuf     = coll_task->args.src.info.buffer;
    void             *dbuf     = coll_task->args.dst.info.buffer;
    const int         ring_id  = task->reduce_scatter.ring_id;
    const ucc_rank_t  trank    = team->rank;
    const ucc_rank_t  tsize    = team->size;
    const ucc_rank_t  sendto   = ucc_ring_send_to(trank, tsize, ring_id);
    const ucc_rank_t  recvfrom = ucc_ring_recv_from(trank, tsize, ring_id);
    const size_t      ccount   = coll_task->args.src.info.count;
    ucc_datatype_t    dt       = coll_task->args.dst.info.datatype;
    const uint32_t    coll_id  = task->reduce_scatter.coll_id;
    mem_info_t *peer_info;
    uint32_t send_step, recv_step;
    ucc_ee_executor_task_args_t exec_args;
    size_t block_count, block_offset, frag_count, frag_offset, recv_block, max_count;
    size_t remote_offset, local_offset;
    ucc_status_t st;

    while(task->reduce_scatter.step < tsize) {
        if (task->reduce_scatter.exec_task != NULL) {
            st = ucc_ee_executor_task_test(task->reduce_scatter.exec_task);
            if (st == UCC_OK) {
                //printf("rank %d: executor task done\n", (int)trank);
                task->reduce_scatter.step++;
                ucc_tl_cuda_ipc_set_rank_step(team, coll_id, trank,
                                              task->reduce_scatter.step,
                                              ring_id);
                task->reduce_scatter.exec_task = NULL;
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
        if ((recv_step < task->reduce_scatter.step) ||
            (send_step < task->reduce_scatter.step))
        {
            task->super.super.status = UCC_INPROGRESS;
            return UCC_INPROGRESS;
        }


        max_count = ucc_ring_block_count(ccount, tsize, 0);
        // send_block = ucc_tl_cuda_ipc_get_send_block(trank, tsize, task->reduce_scatter.step); //(trank + tsize - task->reduce_scatter.step + 1) % tsize;
        // block_count = ucc_ring_block_count(ccount, tsize, send_block);
        // block_offset = ucc_ring_block_offset(ccount, tsize, send_block);
        // printf("rank %d: block %d send to %d: count %d offset %d\n", (int)trank, (int)send_block, (int)sendto, (int)block_count, (int)block_offset);
        recv_block = ucc_tl_cuda_ipc_get_recv_block(trank, tsize, task->reduce_scatter.step, ring_id);
        // block_count = ucc_ring_block_count(ccount, tsize, recv_block);
        // block_offset = ucc_ring_block_offset(ccount, tsize, recv_block);
        block_count = ucc_ring_block_count(ccount, tsize, recv_block);
        frag_count = ucc_ring_block_count(block_count, task->reduce_scatter.n_rings, ring_id);
        block_offset = ucc_ring_block_offset(ccount, tsize, recv_block);
        frag_offset = ucc_ring_block_offset(block_count, task->reduce_scatter.n_rings, ring_id);

        //printf("rank %d: recv from %d: count %d offset %d send_step %d recv_step %d step %d\n", (int)trank, (int)recvfrom, (int)block_count, (int)block_offset, (int)send_step, (int)recv_step, (int)task->reduce_scatter.step);


        peer_info = GET_MEM_INFO(team, coll_id, recvfrom);
        if (task->reduce_scatter.step % 2) {
            remote_offset = 0;
            local_offset = max_count * ucc_dt_size(dt);
        } else {
            remote_offset = max_count * ucc_dt_size(dt);
            local_offset = 0;
        }
        exec_args.task_type     = UCC_MC_EE_EXECUTOR_TASK_TYPE_REDUCE;
        exec_args.src1.buffer   = PTR_OFFSET(sbuf, (block_offset + frag_offset) * ucc_dt_size(dt));
        //take from remote scratch
        exec_args.src1.count    = frag_count;
        exec_args.src1.datatype = dt;
        exec_args.src2.buffer   = PTR_OFFSET(task->reduce_scatter.peer_map_addr,
                                             peer_info->offset + remote_offset);
        exec_args.src2.count    = frag_count;
        exec_args.src2.datatype = dt;
        exec_args.dst.buffer    = (task->reduce_scatter.step == tsize -1 ) ? PTR_OFFSET(dbuf, frag_offset * ucc_dt_size(dt)): PTR_OFFSET(scratch, local_offset);
        exec_args.dst.count     = frag_count;
        exec_args.dst.datatype  = dt;
        exec_args.op            = UCC_OP_SUM;
        st = ucc_ee_executor_task_post(&exec_args,
                                       &task->reduce_scatter.exec_task,
                                       schedule->reduce_scatter.eee);
        if (ucc_unlikely(st != UCC_OK)) {
            task->super.super.status = st;
            return st;
        }
    }
    task->super.super.status = UCC_OK;
    return task->super.super.status;
}

ucc_status_t ucc_tl_cuda_ipc_reduce_scatter_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_cuda_ipc_task_t *task = ucc_derived_of(coll_task,
                                                  ucc_tl_cuda_ipc_task_t);
    ucc_tl_cuda_ipc_schedule_t *schedule = ucc_derived_of(coll_task->schedule, ucc_tl_cuda_ipc_schedule_t);
    ucc_tl_cuda_ipc_team_t *team    = TASK_TEAM(task);
    ucc_rank_t              trank   = team->rank;
    ucc_rank_t              tsize   = team->size;
    void                   *scratch = task->reduce_scatter.scratch;
    void                   *sbuf    = coll_task->args.src.info.buffer;
    size_t                  ccount  = coll_task->args.src.info.count; //TODO: fix inplace
    ucc_datatype_t          dt      = coll_task->args.dst.info.datatype;
    size_t block_count, block_offset, frag_count, frag_offset, block;//, max_count;
    ucc_ee_executor_task_args_t exec_args;
    // ucc_ee_executor_params_t exec_params;
    ucc_status_t st;
    // ucc_rank_t  sendto   = (trank + 1) % tsize;
    // ucc_rank_t  recvfrom = (trank - 1 + tsize) % tsize;
    // size_t block_count, block_offset, send_block, recv_block;

    // exec_params.ee_type = UCC_EE_CUDA_STREAM;
    // exec_params.ee_context = team->stream;
    // st = ucc_ee_executor_create_post(&exec_params,
    //                                  (ucc_ee_executor_t**)&coll_task->ee_task);
    // if (ucc_unlikely(st != UCC_OK)) {
    //     ucc_error("failed to create ee executor");
    //     coll_task->super.status = st;
    //     return st;
    // }

    // //TODO: make nonblocking
    // do {
    //     st = ucc_ee_executor_create_test(coll_task->ee_task);
    // } while (st == UCC_INPROGRESS);

    // if (ucc_unlikely(st != UCC_OK)) {
    //     ucc_error("failed to create ee executor");
    //     coll_task->super.status = st;
    //     return st;
    // }

    //max_count = ucc_ring_block_count(ccount, tsize, 0);
    block = ucc_tl_cuda_ipc_get_send_block(trank, tsize, 1, task->reduce_scatter.ring_id);
    // block_count = ucc_ring_block_count(ccount, tsize, block);
    // block_offset = ucc_ring_block_offset(ccount, tsize, block);
    block_count = ucc_ring_block_count(ccount, tsize, block);
    frag_count = ucc_ring_block_count(block_count, task->reduce_scatter.n_rings, task->reduce_scatter.ring_id);
    block_offset = ucc_ring_block_offset(ccount, tsize, block);
    frag_offset = ucc_ring_block_offset(block_count, task->reduce_scatter.n_rings, task->reduce_scatter.ring_id);

    exec_args.task_type   = UCC_MC_EE_EXECUTOR_TASK_TYPE_COPY;
    exec_args.src1.buffer = PTR_OFFSET(sbuf, (block_offset + frag_offset) * ucc_dt_size(dt));
    exec_args.src1.count  = frag_count * ucc_dt_size(dt);
    exec_args.dst.buffer  = scratch; //PTR_OFFSET(scratch, 2 * max_count * ucc_dt_size(dt));
    exec_args.dst.count   = frag_count * ucc_dt_size(dt);
    st = ucc_ee_executor_task_post(&exec_args,
                                   &task->reduce_scatter.exec_task,
                                   schedule->reduce_scatter.eee);
    if (ucc_unlikely(st != UCC_OK)) {
        task->super.super.status = st;
        return st;
    }


    // recv_block = (trank + tsize - 1) % tsize;
    // block_count = ucc_ring_block_count(ccount, tsize, recv_block);
    // block_offset = ucc_ring_block_offset(ccount, tsize, recv_block);
    // printf("rank %d: recv from %d: count %d offset %d\n", (int)trank, (int)recvfrom, (int)block_count, (int)block_offset);


    // printf("rank %d: send to %d: count %d offset %d\n", (int)trank, (int)sendto, (int)block_count, (int)block_offset);
    task->reduce_scatter.step = 0;
    st = ucc_tl_cuda_ipc_reduce_scatter_progress(coll_task);
    if (UCC_INPROGRESS == task->super.super.status) {
        ucc_progress_enqueue(UCC_TL_CORE_CTX(team)->pq, &task->super);
        return UCC_OK;
    }

    return ucc_task_complete(coll_task);
}

ucc_status_t ucc_tl_cuda_ipc_reduce_scatter_cuda_ipc_setup(ucc_coll_task_t *coll_task)
{
    ucc_tl_cuda_ipc_task_t *task = ucc_derived_of(coll_task, ucc_tl_cuda_ipc_task_t);
    ucc_tl_cuda_ipc_team_t *team = TASK_TEAM(task);
    const int               ring_id  = task->reduce_scatter.ring_id;
    ucc_rank_t              trank    = team->rank;
    ucc_rank_t              tsize    = team->size;
    size_t                  ccount   = coll_task->args.dst.info.count;
    ucc_datatype_t          dt       = coll_task->args.dst.info.datatype;
    ucc_rank_t              recvfrom = ucc_ring_recv_from(trank, tsize, ring_id);
    ucc_rank_t              sendto   = ucc_ring_send_to(trank, tsize, ring_id);
    ucc_cuda_ipc_cache_t   *cache;
    size_t                  block_count;
    uint32_t                max_concurrent, coll_id;
    mem_info_t             *my_info;
    void                   *base_address, *mapped_addr;
    size_t                  alloc_length;
    ucc_status_t            status;

    max_concurrent = UCC_TL_CUDA_IPC_TEAM_LIB(team)->cfg.max_concurrent;
    coll_id   = (task->seq_num % max_concurrent);
    my_info   = GET_MEM_INFO(team, coll_id, trank);
    block_count = ucc_ring_block_count(ccount, tsize, 0);

    ucc_tl_cuda_ipc_get_alloc_info(task->reduce_scatter.scratch,
                                   2 * block_count * ucc_dt_size(dt),
                                   &base_address, &alloc_length);
    if (base_address != NULL) {
        CUDACHECK(cudaIpcGetMemHandle((cudaIpcMemHandle_t *) &my_info->handle, base_address));
    }
    my_info->d_ptr  = base_address;
    my_info->size   = alloc_length;
    my_info->offset = task->reduce_scatter.scratch - base_address;
    ucc_tl_cuda_ipc_set_rank_step(team, coll_id, trank, 0, ring_id);

    __sync_synchronize();
    asm volatile("": : :"memory");
    my_info->seq_num[0] = task->seq_num;
    volatile mem_info_t *recvfrom_mi = GET_MEM_INFO(team, coll_id, recvfrom);
    volatile mem_info_t *sendto_mi   = GET_MEM_INFO(team, coll_id, sendto);

    while ((recvfrom_mi->seq_num[0] != task->seq_num) ||
           (sendto_mi->seq_num[0] != task->seq_num));

    cache = ucc_cuda_ipc_get_cache(team, recvfrom);
    if (ucc_unlikely(!cache)) {
        return UCC_ERR_NO_MESSAGE;
    }
    status = ucc_cuda_ipc_map_memhandle(recvfrom_mi->d_ptr, recvfrom_mi->size,
                                        recvfrom_mi->handle, &mapped_addr,
                                        cache);
    if (UCC_OK != status) {
        ucc_error("ucc_cuda_ipc_map_memhandle failed");
        return UCC_ERR_INVALID_PARAM;
    }
    task->reduce_scatter.peer_map_addr = mapped_addr;
    task->reduce_scatter.coll_id       = coll_id;

    return UCC_OK;
}

ucc_status_t
ucc_tl_cuda_ipc_reduce_scatter_ring_sched_post(ucc_coll_task_t *coll_task)
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

    exec_params.ee_type = UCC_EE_CUDA_STREAM;
    exec_params.ee_context = team->stream;
    st = ucc_ee_executor_create_post(&exec_params, &schedule->reduce_scatter.eee);
    if (ucc_unlikely(st != UCC_OK)) {
        ucc_error("failed to create ee executor");
        return st;
    }

    //TODO: make nonblocking
    do {
        st = ucc_ee_executor_create_test(schedule->reduce_scatter.eee);
    } while (st == UCC_INPROGRESS);

    if (ucc_unlikely(st != UCC_OK)) {
        ucc_error("failed to create ee executor");
        return st;
    }

    return ucc_schedule_start(&schedule->super);
}

ucc_status_t
ucc_tl_cuda_ipc_reduce_scatter_ring_sched_finalize(ucc_coll_task_t *task)
{
    ucc_tl_cuda_ipc_schedule_t *schedule = ucc_derived_of(task, ucc_tl_cuda_ipc_schedule_t);
    ucc_status_t status;

//TODO: move to completed handler
    ucc_ee_executor_destroy((ucc_ee_executor_t*)schedule->reduce_scatter.eee);

    status = ucc_schedule_finalize(task);
    ucc_tl_cuda_ipc_put_schedule(schedule);
    return status;
}


ucc_status_t ucc_tl_cuda_ipc_reduce_scatter_init(ucc_base_coll_args_t *coll_args,
                                                 ucc_base_team_t *tl_team,
                                                 ucc_coll_task_t **task_p)
{
    ucc_tl_cuda_ipc_team_t *team     = ucc_derived_of(tl_team,
                                                      ucc_tl_cuda_ipc_team_t);
    // ucc_tl_cuda_ipc_task_t *task     = ucc_tl_cuda_ipc_init_task(coll_args,
    //                                                              tl_team);
    ucc_tl_cuda_ipc_schedule_t *schedule = ucc_tl_cuda_ipc_get_schedule(&coll_args->args,
                                                                        team);
    size_t             ccount = coll_args->args.dst.info.count;
    ucc_rank_t         tsize  = team->size;
    ucc_datatype_t     dt     = coll_args->args.dst.info.datatype;
    ucc_tl_cuda_ipc_task_t *task;
    int i, n_rings;
    size_t block_count;
    ucc_status_t status;

    if (!UCC_IS_INPLACE(coll_args->args)) {
        ccount = coll_args->args.src.info.count;
    }

    n_rings = 2;
    block_count = ucc_ring_block_count(ccount, tsize, 0);
    for (i = 0; i < n_rings; i++) {
        task = ucc_tl_cuda_ipc_init_task(coll_args, tl_team);
        if (!task) {
            return UCC_ERR_NO_MEMORY;
        }
        task->reduce_scatter.ring_id = i;
        task->reduce_scatter.n_rings = n_rings;
        status = ucc_mc_alloc(&task->reduce_scatter.scratch_mc_header,
                              2 * block_count * ucc_dt_size(dt),
                              UCC_MEMORY_TYPE_CUDA);
        task->reduce_scatter.scratch = task->reduce_scatter.scratch_mc_header->addr;
        if (ucc_unlikely(status != UCC_OK)) {
            tl_error(UCC_TASK_LIB(task), "failed to allocate scratch buffer");
            return status;
        }
        task->super.post     = ucc_tl_cuda_ipc_reduce_scatter_start;
        task->super.progress = ucc_tl_cuda_ipc_reduce_scatter_progress;
        task->super.finalize = ucc_tl_cuda_ipc_reduce_scatter_finalize;
        ucc_tl_cuda_ipc_reduce_scatter_cuda_ipc_setup(&task->super);
        ucc_schedule_add_task(&schedule->super, &task->super);
        ucc_event_manager_subscribe(&schedule->super.super.em, UCC_EVENT_SCHEDULE_STARTED,
                                    &task->super, ucc_task_start_handler);
    }
    schedule->super.super.post     = ucc_tl_cuda_ipc_reduce_scatter_ring_sched_post;
    schedule->super.super.finalize = ucc_tl_cuda_ipc_reduce_scatter_ring_sched_finalize;

    *task_p = &schedule->super.super;
    return UCC_OK;
}
