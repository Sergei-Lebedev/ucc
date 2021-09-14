/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_TL_CUDA_IPC_RING_H_
#define UCC_TL_CUDA_IPC_RING_H_

#include "tl_cuda_ipc_coll.h"

#define N_DGX_RINGS 4

extern ucc_rank_t dgx_map[N_DGX_RINGS][8];
extern ucc_rank_t dgx_imap[N_DGX_RINGS][8];

static inline size_t ucc_ring_block_offset(size_t total_count,
                                           ucc_rank_t n_blocks,
                                           ucc_rank_t block)
{
    size_t block_count = total_count / n_blocks;
    size_t left        = total_count % n_blocks;
    size_t offset      = block * block_count + left;
    return (block < left) ? offset - (left - block) : offset;
}

static inline size_t ucc_ring_block_count(size_t total_count,
                                          ucc_rank_t n_blocks, ucc_rank_t block)
{
    size_t block_count = total_count / n_blocks;
    size_t left        = total_count % n_blocks;
    return (block < left) ? block_count + 1 : block_count;
}

static inline size_t ucc_tl_cuda_ipc_ring_max_frag_size(size_t total_count,
                                                        ucc_rank_t n_blocks,
                                                        ucc_rank_t n_frags)
{
    size_t block_count = ucc_ring_block_count(total_count, n_blocks, 0);
    return ucc_ring_block_count(block_count, n_frags, 0);
}

static inline ucc_rank_t ucc_ring_send_to(ucc_rank_t rank, ucc_rank_t size,
                                          int ring_id) {
    return dgx_map[ring_id][(dgx_imap[ring_id][rank] + 1) % size];
}

static inline ucc_rank_t ucc_ring_recv_from(ucc_rank_t rank, ucc_rank_t size,
                                            int ring_id) {
    return dgx_map[ring_id][(dgx_imap[ring_id][rank] - 1 + size) % size];
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

ucc_status_t ucc_tl_cuda_ipc_ring_setup(ucc_coll_task_t *coll_task,
                                        int ring_id,
                                        uint32_t coll_id,
                                        void *scratch,
                                        size_t scratch_size,
                                        void **recv_peer_addr);

#endif
