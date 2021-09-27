/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "tl_cuda_ipc_coll.h"
#include "core/ucc_mc.h"
#include "core/ucc_ee.h"
#include "utils/ucc_math.h"
#include "utils/ucc_coll_utils.h"

#define UCC_TL_CUDA_IPC_REDUCE_SCATTER_DEFAULT_ALG_SELECT_STR                  \
    "reduce_scatter:0-512k:cuda:@0#reduce_scatter:512k-inf:cuda:@1"

#define UCC_TL_CUDA_IPC_ALLGATHER_DEFAULT_ALG_SELECT_STR                       \
    "allgather:0-512k:cuda:@0#allgather:512k-inf:cuda:@1"

const char
    *ucc_tl_cuda_ipc_default_alg_select_str[UCC_TL_CUDA_IPC_N_DEFAULT_ALG_SELECT_STR] = {
    UCC_TL_CUDA_IPC_REDUCE_SCATTER_DEFAULT_ALG_SELECT_STR,
    UCC_TL_CUDA_IPC_ALLGATHER_DEFAULT_ALG_SELECT_STR};

//todo move to common place (mc_cuda.h)
static inline ucc_status_t cuda_error_to_ucc_status(cudaError_t cu_err)
{
    switch(cu_err) {
    case cudaSuccess:
        return UCC_OK;
    case cudaErrorNotReady:
        return UCC_INPROGRESS;
    default:
        break;
    }
    return UCC_ERR_NO_MESSAGE;
}

ucc_status_t ucc_tl_cuda_ipc_alltoallv_cuda_ipc_setup(ucc_coll_task_t *coll_task);

ucc_status_t ucc_tl_cuda_ipc_coll_init(ucc_base_coll_args_t *coll_args,
                                       ucc_base_team_t *team,
                                       ucc_coll_task_t **task_p)
{
    return UCC_ERR_NOT_IMPLEMENTED;
}

ucc_status_t ucc_tl_cuda_ipc_alltoallv_finalize(ucc_coll_task_t *coll_task)
{
    ucc_tl_cuda_ipc_task_t *task = ucc_derived_of(coll_task,
                                                  ucc_tl_cuda_ipc_task_t);
    ucc_tl_cuda_ipc_team_t *team = TASK_TEAM(task);

    if (team->size > MAX_STATIC_SIZE) {
        ucc_free(task->alltoallv.peer_map_addr);
    }
    ucc_tl_cuda_ipc_put_task(task);
    return UCC_OK;
}

ucc_status_t ucc_tl_cuda_ipc_alltoallv_progress(ucc_coll_task_t *coll_task)
{
    ucc_tl_cuda_ipc_task_t *task     = ucc_derived_of(coll_task,
                                                      ucc_tl_cuda_ipc_task_t);
    ucc_tl_cuda_ipc_team_t *team     = TASK_TEAM(task);
    uint32_t                coll_id  = task->alltoallv.coll_id;
    ucc_rank_t              num_done = 0;
    ucc_rank_t i, peer;
    mem_info_t *peer_info, *my_info;
    ucc_status_t st;

    for (i = 0; i < team->size; i++) {
        peer = (team->rank + i) % team->size;
        if ((task->alltoallv.exec_task[peer] == NULL) &&
            (GET_MEM_INFO(team, coll_id, peer)->seq_num[1] == task->seq_num)) {
            ucc_ee_executor_task_args_t exec_args;
            void                        *src, *dst;
            size_t                      data_size, data_displ, dt_size;

            peer_info  = GET_MEM_INFO(team, coll_id, peer);
            data_size  = ucc_coll_args_get_count(&coll_task->args,
                                coll_task->args.dst.info_v.counts, peer);
            data_displ = ucc_coll_args_get_displacement(&coll_task->args,
                                coll_task->args.dst.info_v.displacements, peer);
            dt_size    = ucc_dt_size(coll_task->args.dst.info_v.datatype);

            if (peer == team->rank) {
                src = PTR_OFFSET(coll_task->args.src.info_v.buffer,
                        peer_info->data[team->rank].displ);
            } else {
                src = PTR_OFFSET(task->alltoallv.peer_map_addr[peer],
                        peer_info->offset + peer_info->data[team->rank].displ);
            }
            dst = PTR_OFFSET(coll_task->args.dst.info_v.buffer,
                    data_displ * dt_size);

            exec_args.task_type   = UCC_MC_EE_EXECUTOR_TASK_TYPE_COPY;
            exec_args.src1.buffer = src;
            exec_args.src1.count  = data_size * dt_size;
            exec_args.dst.buffer  = dst;
            exec_args.dst.count   = data_size * dt_size;
            st = ucc_ee_executor_task_post(&exec_args,
                                           &task->alltoallv.exec_task[peer],
                                           task->super.ee_task);
            if (ucc_unlikely(st != UCC_OK)) {
                task->super.super.status = st;
                goto exit;
            }
        }
        if ((task->alltoallv.exec_task[peer] != NULL) &&
            (ucc_ee_executor_task_test(task->alltoallv.exec_task[peer]) == UCC_OK)) {
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

    task->super.super.status = (num_done == team->size) ? UCC_OK: UCC_INPROGRESS;
exit:
    return task->super.super.status;
}

ucc_status_t ucc_tl_cuda_ipc_alltoallv_early_triggered_post(ucc_coll_task_t *coll_task)
{
    return UCC_OK;
    // ucc_tl_cuda_ipc_task_t *task = ucc_derived_of(coll_task,
    //                                               ucc_tl_cuda_ipc_task_t);

    // ucc_tl_cuda_ipc_task_reset(task);
    // task->early_posted = 1;
    // if (coll_task->ee->ee_context) {
    //     task->stream = (cudaStream_t)coll_task->ee->ee_context;
    // }

    // return ucc_tl_cuda_ipc_alltoallv_cuda_ipc_setup(&task->super);
}


ucc_status_t ucc_tl_cuda_ipc_alltoallv_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_cuda_ipc_task_t *task = ucc_derived_of(coll_task,
                                                  ucc_tl_cuda_ipc_task_t);
    ucc_tl_cuda_ipc_team_t *team = TASK_TEAM(task);
    mem_info_t             *info = GET_MEM_INFO(team, task->alltoallv.coll_id,
                                                team->rank);
    ucc_rank_t rank;

    // ucc_status_t status;
    // if (!task->early_posted) {
    //     ucc_tl_cuda_ipc_task_reset(task);
    //     status = ucc_tl_cuda_ipc_alltoallv_cuda_ipc_setup(&task->super);
    //     if (UCC_OK != status) {
    //         return status;
    //     }
    // }

    for (rank = 0; rank < team->size; rank++) {
        task->alltoallv.exec_task[rank] = NULL;
    }
    __sync_synchronize();
    asm volatile("": : :"memory");
    info->seq_num[1] = task->seq_num;

    ucc_tl_cuda_ipc_alltoallv_progress(&task->super);
    if (UCC_INPROGRESS == task->super.super.status) {
        ucc_progress_enqueue(UCC_TL_CORE_CTX(team)->pq, &task->super);
        return UCC_OK;
    }

    return ucc_task_complete(coll_task);

}

ucc_status_t ucc_tl_cuda_ipc_alltoallv_init(ucc_base_coll_args_t *coll_args,
                                            ucc_base_team_t *tl_team,
                                            ucc_coll_task_t **task_p)
{
    ucc_tl_cuda_ipc_team_t *team = ucc_derived_of(tl_team,
                                                  ucc_tl_cuda_ipc_team_t);
    ucc_tl_cuda_ipc_task_t *task = ucc_tl_cuda_ipc_init_task(coll_args,
                                                             tl_team);

    if (!task) {
        return UCC_ERR_NO_MEMORY;
    }
    if (team->size <= MAX_STATIC_SIZE) {
        task->alltoallv.peer_map_addr = task->data;
    } else {
        task->alltoallv.peer_map_addr = ucc_malloc(sizeof(void*) * team->size);
        if (!task->alltoallv.peer_map_addr) {
            tl_error(UCC_TL_TEAM_LIB(team),
                     "failed to allocate %zd bytes for peer_map_addr",
                     sizeof(void*) * team->size);
            ucc_tl_cuda_ipc_put_task(task);
            return UCC_ERR_NO_MEMORY;
        }
    }
    task->stream         = team->stream;
    task->super.post     = ucc_tl_cuda_ipc_alltoallv_start;
    task->super.progress = ucc_tl_cuda_ipc_alltoallv_progress;
    task->super.finalize = ucc_tl_cuda_ipc_alltoallv_finalize;
    task->super.early_triggered_post =
        ucc_tl_cuda_ipc_alltoallv_early_triggered_post;
    ucc_tl_cuda_ipc_alltoallv_cuda_ipc_setup(&task->super);
    *task_p = &task->super;
    return UCC_OK;
}

//TODO move to utils
void ucc_tl_cuda_ipc_get_alloc_info(void *ptr, size_t length,
                                    void **base_address, size_t *alloc_length)
{
    ucc_mem_attr_t mem_attr;
    ucc_status_t status;

    *base_address  = ptr;
    *alloc_length = length;
    if (length == 0) {
        *base_address = NULL;
        return;
    }

    mem_attr.field_mask   = UCC_MEM_ATTR_FIELD_BASE_ADDRESS |
                            UCC_MEM_ATTR_FIELD_ALLOC_LENGTH;
    mem_attr.alloc_length = length;
    status = ucc_mc_get_mem_attr(ptr, &mem_attr);
    if (ucc_likely(status == UCC_OK)) {
        *base_address = mem_attr.base_address;
        *alloc_length = mem_attr.alloc_length;
    }

}

ucc_status_t ucc_tl_cuda_ipc_alltoallv_cuda_ipc_setup(ucc_coll_task_t *coll_task)
{
    ucc_tl_cuda_ipc_task_t *task = ucc_derived_of(coll_task, ucc_tl_cuda_ipc_task_t);
    ucc_tl_cuda_ipc_team_t *team = TASK_TEAM(task);
    ucc_status_t            status;
    void                   *base_address;
    size_t                  alloc_length, sdt_size, rdt_size, ipc_thresh;
    int                     i, coll_id;
    mem_info_t             *my_info;
    void                   *mapped_addr;
    size_t                  total_counts;
    uint32_t                max_concurrent;
    ucc_cuda_ipc_cache_t   *cache;

    max_concurrent = UCC_TL_CUDA_IPC_TEAM_LIB(team)->cfg.max_concurrent;
    /* ipc_thresh = UCC_TL_CUDA_IPC_TEAM_CTX(team)->cfg.alltoallv_ipc_thresh; */
    ipc_thresh = 0;

    coll_id   = (task->seq_num % max_concurrent);

    my_info   = GET_MEM_INFO(team, coll_id, team->rank);

    rdt_size = ucc_dt_size(coll_task->args.dst.info_v.datatype);
    sdt_size = ucc_dt_size(coll_task->args.src.info_v.datatype);

    total_counts = ucc_coll_args_get_total_count(&coll_task->args, coll_task->args.src.info_v.counts, team->size);
    ucc_tl_cuda_ipc_get_alloc_info(coll_task->args.src.info_v.buffer, total_counts * sdt_size,  &base_address, &alloc_length);

    if (base_address != NULL) {
        CUDACHECK(cudaIpcGetMemHandle((cudaIpcMemHandle_t *) &my_info->handle, base_address));
    }

    my_info->d_ptr  = base_address;
    my_info->size   = alloc_length;
    my_info->offset = coll_task->args.src.info_v.buffer - base_address;

    for (i = 0; i < team->size; i++) {
        my_info->data[i].displ =  ucc_coll_args_get_displacement(&coll_task->args,
                coll_task->args.src.info_v.displacements, i) * sdt_size;
    }

    /* { */
    /*     volatile int __flag = 0; */
    /*     char hostname[256]; */
    /*     gethostname(hostname, sizeof(hostname)); */
    /*     printf("PID %d on %s ready for attach\n", getpid(), hostname); */
    /*     fflush(stdout); */
    /*     while (__flag == 0) */
    /*         sleep(5); */
    /* } */

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
        if (i != team->rank && pi->d_ptr &&
                (ucc_coll_args_get_count(&coll_task->args, coll_task->args.dst.info_v.counts, i) *
                 rdt_size) >= ipc_thresh) {
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
            task->alltoallv.peer_map_addr[i] = mapped_addr;
        }
    }

    task->alltoallv.coll_id  = coll_id;
    return UCC_OK;
}

ucc_status_t ucc_tl_cuda_ipc_alg_id_to_init(int alg_id, const char *alg_id_str,
                                            ucc_coll_type_t   coll_type,
                                            ucc_memory_type_t mem_type, //NOLINT
                                            ucc_base_coll_init_fn_t *init)
{
    ucc_status_t status = UCC_OK;
    switch (coll_type) {
    case UCC_COLL_TYPE_REDUCE_SCATTER:
        switch (alg_id) {
        case UCC_TL_CUDA_IPC_REDUCE_SCATTER_ALG_LINEAR:
            *init = ucc_tl_cuda_ipc_reduce_scatter_linear_init;
            break;
        case UCC_TL_CUDA_IPC_REDUCE_SCATTER_ALG_RING:
            *init = ucc_tl_cuda_ipc_reduce_scatter_ring_init;
            break;
        default:
            status = UCC_ERR_INVALID_PARAM;
            break;

        };
        break;
    case UCC_COLL_TYPE_ALLGATHER:
        switch (alg_id) {
        case UCC_TL_CUDA_IPC_ALLGATHER_ALG_LINEAR:
            *init = ucc_tl_cuda_ipc_allgather_linear_init;
            break;
        case UCC_TL_CUDA_IPC_ALLGATHER_ALG_RING:
            *init = ucc_tl_cuda_ipc_allgather_ring_init;
            break;
        default:
            status = UCC_ERR_INVALID_PARAM;
            break;
        };
        break;
    default:
        status = UCC_ERR_NOT_SUPPORTED;
        break;
    }
    return status;
}
