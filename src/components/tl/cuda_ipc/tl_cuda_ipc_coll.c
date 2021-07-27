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
    ucc_tl_cuda_ipc_task_t *task = ucc_derived_of(coll_task,
                                                  ucc_tl_cuda_ipc_task_t);
    cudaError_t cu_err;

    cu_err = cudaEventQuery(task->event);

    task->super.super.status = cuda_error_to_ucc_status(cu_err);
    return task->super.super.status;
}

static ucc_status_t ucc_tl_cuda_ipc_alltoallv_post_copies(ucc_coll_task_t *coll_task)
{
    ucc_tl_cuda_ipc_task_t *task = ucc_derived_of(coll_task,
                                                  ucc_tl_cuda_ipc_task_t);
    ucc_tl_cuda_ipc_team_t *team  = TASK_TEAM(task);
    ptrdiff_t          rbuf  = (ptrdiff_t)coll_task->args.dst.info_v.buffer;
    size_t   rdt_size, sdt_size, data_size, data_displ, ipc_thresh;
    int i, peer;
    mem_info_t *peer_info, *my_info;
    ptrdiff_t src;
    uint32_t coll_id;
    uint32_t max_concurrent;
    cuda_ipc_sync_t        *s;

    max_concurrent = UCC_TL_CUDA_IPC_TEAM_LIB(team)->cfg.max_concurrent;
    /* ipc_thresh = UCC_TL_CUDA_IPC_TEAM_CTX(team)->cfg.alltoallv_ipc_thresh; */
    ipc_thresh = 0;

    coll_id   = (task->seq_num % max_concurrent);
    s = GET_SYNC(team, coll_id);
    rdt_size = ucc_dt_size(coll_task->args.dst.info_v.datatype);
    sdt_size = ucc_dt_size(coll_task->args.src.info_v.datatype);
    for (i=0; i < team->size; i++) {
        peer = (team->rank + i) % team->size;
        /* peer_info = &((mem_info_t *)task->alltoall_intra.info)[peer]; */
        peer_info = GET_MEM_INFO(team, coll_id, peer);

        if (peer == team->rank) {
            src = (ptrdiff_t)coll_task->args.src.info_v.buffer +
                    + peer_info->data[team->rank].displ;
        } else {
            src = (ptrdiff_t) task->alltoallv.peer_map_addr[peer] +
                    peer_info->offset + peer_info->data[team->rank].displ;
        }

        data_size  = ucc_coll_args_get_count(&coll_task->args,
                            coll_task->args.dst.info_v.counts, peer) * rdt_size;
        if (data_size < ipc_thresh) {
            continue;
        }
        data_displ = ucc_coll_args_get_displacement(&coll_task->args,
                            coll_task->args.dst.info_v.displacements, peer)* rdt_size;

        //printf("SNED [%d: %d] sdispl:%ld rdispl:(%ld:%ld) size:%ld \n", team->rank, rank, data_displ, peer_info->offset, peer_info->displ[intra_rank], data_size);
        if (data_size != 0) {
            if (peer != team->rank) {
                CUDACHECK(cudaStreamWaitEvent(task->stream,
                                              s[peer].ipc_event, 0));
            }

            CUDACHECK(cudaMemcpyAsync((void *)(rbuf + data_displ), (void *)src,
                                      data_size, cudaMemcpyDeviceToDevice,
                                      task->stream));

            if (peer != team->rank) {
                CUDACHECK(cudaEventRecord(s[peer].ipc_event, task->stream));
            }
        }
    }

    my_info = GET_MEM_INFO(team, coll_id, team->rank);

    volatile mem_info_t *pi;
    __sync_synchronize();
    asm volatile("": : :"memory");
    my_info->seq_num[1] = task->seq_num;

    for (i = 0; i < team->size; i++) {
        pi = GET_MEM_INFO(team, coll_id, i);
        while (pi->seq_num[1] != task->seq_num);
    }

    for (i= 0 ; i < team->size; i++) {
        pi = GET_MEM_INFO(team, coll_id, i);
        if (i != team->rank) {
            data_size  = ucc_coll_args_get_count(&coll_task->args,
                            coll_task->args.src.info_v.counts, i) * sdt_size;
            if (data_size != 0) {
                CUDACHECK(cudaStreamWaitEvent(task->stream, s[i].event, 0));
            }
        }
    }
    CUDACHECK(cudaEventRecord(task->event, task->stream));
    return UCC_OK;
}


ucc_status_t ucc_tl_cuda_ipc_alltoallv_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_cuda_ipc_task_t *task = ucc_derived_of(coll_task,
                                                  ucc_tl_cuda_ipc_task_t);
    ucc_tl_cuda_ipc_team_t *team = TASK_TEAM(task);
    ucc_status_t            status;

    ucc_tl_cuda_ipc_task_reset(task);
    status = ucc_tl_cuda_ipc_alltoallv_cuda_ipc_setup(&task->super);
    if (UCC_OK != status) {
        return status;
    }
    status = ucc_tl_cuda_ipc_alltoallv_post_copies(&task->super);
    if (UCC_OK != status) {
        return status;
    }

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
    cuda_ipc_sync_t        *s;
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

    s = GET_SYNC(team, coll_id);
    for (i = 0; i < team->size; i++) {
        my_info->data[i].displ =  ucc_coll_args_get_displacement(&coll_task->args,
                coll_task->args.src.info_v.displacements, i) * sdt_size;
        my_info->data[i].ev_handle = s[i].ipc_event_handle;
        CUDACHECK(cudaEventRecord(s[i].event, task->stream));
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

        if(i != team->rank) {
            if (s[i].ipc_event == (cudaEvent_t) NULL) {
                CUDACHECK(cudaIpcOpenEventHandle(&s[i].ipc_event,
                                                 pi->data[team->rank].ev_handle));
            }
        }
    }

    task->alltoallv.coll_id  = coll_id;
    return UCC_OK;
}
