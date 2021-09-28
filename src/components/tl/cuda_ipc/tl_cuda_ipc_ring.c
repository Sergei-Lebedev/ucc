#include "tl_cuda_ipc_ring.h"

ucc_rank_t dgx_map8[N_DGX_RINGS][8] = {
    {0, 3, 2, 1, 5, 6, 7, 4}, //NV2
    {0, 3, 2, 1, 5, 6, 7, 4}, //NV2
    {0, 4, 7, 6, 5, 1, 2, 3}, //NV2
    {0, 4, 7, 6, 5, 1, 2, 3}, //NV2
    {0, 1, 3, 7, 5, 4, 6, 2}, //NV1
    {0, 2, 6, 4, 5, 7, 3, 1}, //NV1
    {0, 3, 2, 1, 5, 6, 7, 4}, //NV2
    {0, 3, 2, 1, 5, 6, 7, 4}, //NV2
};

ucc_rank_t dgx_map2[N_DGX_RINGS][8] = {
    {0, 1, 2, 3, 0, 0, 0, 0},
    {0, 3, 2, 1, 0, 0, 0, 0},
    {0, 3, 1, 2, 0, 0, 0, 0},
    {0, 2, 1, 3, 0, 0, 0, 0},
    {0, 1, 2, 3, 0, 0, 0, 0},
    {0, 3, 2, 1, 0, 0, 0, 0},
    {0, 3, 1, 2, 0, 0, 0, 0},
    {0, 2, 1, 3, 0, 0, 0, 0},
};

ucc_rank_t dgx_map1[N_DGX_RINGS][8] = {
    {0, 1, 2, 3, 4, 5, 6, 7},
    {0, 1, 2, 3, 4, 5, 6, 7},
    {0, 1, 2, 3, 4, 5, 6, 7},
    {0, 1, 2, 3, 4, 5, 6, 7},
    {0, 1, 2, 3, 4, 5, 6, 7},
    {0, 1, 2, 3, 4, 5, 6, 7},
    {0, 1, 2, 3, 4, 5, 6, 7},
    {0, 1, 2, 3, 4, 5, 6, 7},
};

ucc_rank_t dgx_map[N_DGX_RINGS][8];
ucc_rank_t dgx_imap[N_DGX_RINGS][8];

void ucc_tl_cuda_ipc_rings_init(ucc_rank_t team_size)
{
    int i, j;
    ucc_rank_t (*map)[8];

    if (team_size == 2) {
        map = dgx_map1;
    } else if (team_size == 4) {
        map = dgx_map2;
    } else if (team_size == 8) {
        map = dgx_map8;
    }

    for (i = 0; i < N_DGX_RINGS; i++) {
        for (j = 0; j < team_size; j++) {
            dgx_map[i][j] = map[i][j];
        }
    }

    for (i = 0; i < N_DGX_RINGS; i++) {
        for (j = 0; j < team_size; j++) {
            dgx_imap[i][dgx_map[i][j]] =j;
        }
    }
}

ucc_status_t ucc_tl_cuda_ipc_ring_setup(ucc_coll_task_t *coll_task,
                                        int ring_id,
                                        uint32_t coll_id,
                                        void *scratch,
                                        size_t scratch_size,
                                        void **recv_peer_addr)
{
    ucc_tl_cuda_ipc_task_t *task = ucc_derived_of(coll_task, ucc_tl_cuda_ipc_task_t);
    ucc_tl_cuda_ipc_team_t *team = TASK_TEAM(task);
    ucc_rank_t              trank    = team->rank;
    ucc_rank_t              tsize    = team->size;
    ucc_rank_t              recvfrom = ucc_ring_recv_from(trank, tsize, ring_id);
    ucc_rank_t              sendto   = ucc_ring_send_to(trank, tsize, ring_id);
    ucc_cuda_ipc_cache_t   *cache;
    mem_info_t             *my_info;
    void                   *base_address, *mapped_addr;
    size_t                  alloc_length;
    ucc_status_t            status;

    my_info   = GET_MEM_INFO(team, coll_id, trank);
    ucc_tl_cuda_ipc_get_alloc_info(scratch, scratch_size,
                                   &base_address, &alloc_length);
    CUDACHECK(cudaIpcGetMemHandle((cudaIpcMemHandle_t*) &my_info->handle,
                                   base_address));
    my_info->d_ptr  = base_address;
    my_info->size   = alloc_length;
    my_info->offset = scratch - base_address;
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
    // task->reduce_scatter.peer_map_addr = mapped_addr;
    // task->reduce_scatter.coll_id       = coll_id;
    *recv_peer_addr = mapped_addr;
    return UCC_OK;
}

ucc_status_t ucc_tl_cuda_ipc_ring_free(ucc_coll_task_t *coll_task,
                                       int ring_id, uint32_t coll_id,
                                       void *mapped_addr)
{
    ucc_tl_cuda_ipc_task_t *task = ucc_derived_of(coll_task, ucc_tl_cuda_ipc_task_t);
    ucc_tl_cuda_ipc_team_t *team = TASK_TEAM(task);
    ucc_rank_t              trank    = team->rank;
    ucc_rank_t              tsize    = team->size;
    ucc_rank_t              recvfrom = ucc_ring_recv_from(trank, tsize, ring_id);
    volatile mem_info_t *recvfrom_mi = GET_MEM_INFO(team, coll_id, recvfrom);
    ucc_cuda_ipc_cache_t *cache;
    ucc_status_t status;

    cache = ucc_cuda_ipc_get_cache(team, recvfrom);
    if (ucc_unlikely(!cache)) {
        return UCC_ERR_NO_MESSAGE;
    }
    status = ucc_cuda_ipc_unmap_memhandle((uintptr_t)recvfrom_mi->d_ptr, mapped_addr, cache);
    return status;
}