/**
 * Copyright (C) Mellanox Technologies Ltd. 2022.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "allgatherv.h"
#include "components/ec/ucc_ec.h"
#include "tl_cuda_cache.h"
#include "tl_cuda_ring.h"
#include "utils/arch/cpu.h"
#include "utils/arch/cuda_def.h"

ucc_status_t ucc_tl_cuda_allgatherv_ring_finalize(ucc_coll_task_t *coll_task)
{
    ucc_tl_cuda_task_t *task = ucc_derived_of(coll_task, ucc_tl_cuda_task_t);

    tl_trace(UCC_TASK_LIB(task), "finalizing task %p", task);
    ucc_tl_cuda_task_put(task);
    return UCC_OK;
}

ucc_status_t ucc_tl_cuda_allgatherv_ring_setup_start(ucc_tl_cuda_task_t *task)
{
    ucc_tl_cuda_team_t *team  = TASK_TEAM(task);
    ucc_rank_t          trank = UCC_TL_TEAM_RANK(team);
    ucc_status_t        status;
    int                 ring;

    for (ring = 0; ring < task->allgatherv_ring.num_rings; ring++) {
        set_rank_step(task, trank, 0, ring);
    }
    ucc_memory_cpu_store_fence();
    status = ucc_tl_cuda_shm_barrier_start(UCC_TL_TEAM_RANK(team), task->bar);
    if (ucc_unlikely(status != UCC_OK)) {
        goto exit_err;
    }

    return UCC_OK;

exit_err:
    return status;
}

ucc_status_t ucc_tl_cuda_allgatherv_ring_setup_test(ucc_tl_cuda_task_t *task)
{
    ucc_tl_cuda_team_t *team  = TASK_TEAM(task);
    return ucc_tl_cuda_shm_barrier_test(UCC_TL_TEAM_RANK(team), task->bar);
}

static inline void
ucc_tl_cuda_allgatherv_ring_size_offset(ucc_tl_cuda_task_t *task, int block, int frag, int ring,
                                        size_t *block_size, size_t *frag_size, size_t *ring_frag_size,
                                        size_t *block_offset, size_t *frag_offset, size_t *ring_frag_offset)
{
    ucc_datatype_t dt_size = ucc_dt_size(task->allgatherv_ring.dt);
    int            nrings  = task->allgatherv_ring.num_rings;
    int            nfrags  = task->allgatherv_ring.num_frags;

    *block_size       = task->allgatherv_ring.get_count(task, block) * dt_size;
    *frag_size        = ucc_buffer_block_count(*block_size, nfrags, frag);
    *ring_frag_size   = ucc_buffer_block_count(*frag_size, nrings, ring);

    *block_offset     = task->allgatherv_ring.get_offset(task, block) * dt_size;
    *frag_offset      = ucc_buffer_block_offset(*block_size, nfrags, frag);
    *ring_frag_offset = ucc_buffer_block_offset(*frag_size, nrings, ring);
}

ucc_status_t ucc_tl_cuda_allgatherv_ring_progress_ring(ucc_tl_cuda_task_t *task,
                                                       uint32_t ring_id)
{
    ucc_tl_cuda_team_t *team     = TASK_TEAM(task);
    ucc_coll_args_t    *args     = &TASK_ARGS(task);
    ucc_rank_t          trank    = UCC_TL_TEAM_RANK(team);
    int                 tsize    = (int)UCC_TL_TEAM_SIZE(team);
    ucc_rank_t          sendto   = get_send_to(team, trank, tsize, ring_id);
    ucc_rank_t          recvfrom = get_recv_from(team, trank, tsize, ring_id);
    void               *rbuf     = task->allgatherv_ring.rbuf;
    size_t              ssize    = UCC_TL_CUDA_TEAM_LIB(team)->cfg.scratch_size;
    int                nsteps    = tsize - 1;
    ucc_ee_executor_t *exec;
    ucc_ee_executor_task_args_t exec_args;
    ucc_ee_executor_task_t *etask;
    void *dbuf1, *dbuf2, *sbuf1, *sbuf2;
    int step, send_step, recv_step, frag, frag_step, i;
    ucc_rank_t peer_block;
    ucc_status_t st;
    size_t remote_offset, local_offset, frag_offset, frag_size, block_size,
           block_offset, ring_frag_offset, ring_frag_size, ring_scratch_offset,
           frag_count1, frag_count2;

    st = ucc_coll_task_get_executor(&task->super, &exec);
    if (ucc_unlikely(st != UCC_OK)) {
        return st;
    }

    step = get_rank_step(task, trank, ring_id);
    while (step < (nsteps * task->allgatherv_ring.num_frags)) {
        if ((task->allgatherv_ring.exec_task[ring_id * 2] != NULL) ||
            (task->allgatherv_ring.exec_task[ring_id * 2 + 1] != NULL)) {
            for (i = 1 ; i >= 0; i--) {
                etask = task->allgatherv_ring.exec_task[i + ring_id * 2];
                if (etask != NULL) {
                    st = ucc_ee_executor_task_test(etask);
                    if (st == UCC_OK) {
                        ucc_ee_executor_task_finalize(etask);
                        task->allgatherv_ring.exec_task[i + ring_id * 2] = NULL;
                    } else {
                        if (ucc_likely(st > 0)) {
                            return UCC_INPROGRESS;
                        }
                        return st;
                    }
                }
            }
            step++;
            set_rank_step(task, trank, step, ring_id);
            continue;
        }
        frag      = step / nsteps;
        frag_step = step % nsteps;

        if (frag_step == nsteps - 1) {
            send_step = get_rank_step(task, sendto, ring_id);
            recv_step = get_rank_step(task, recvfrom, ring_id);
            if ((send_step < step) || (recv_step < step)) {
                return UCC_INPROGRESS;
            }
            recv_step = get_rank_step(task, get_recv_from(team, recvfrom, tsize, ring_id), ring_id);
            if ((recv_step < step)) {
                return UCC_INPROGRESS;
            }
            if (step % 2) {
                remote_offset = ssize / 2;
                local_offset = 0;
            } else {
                remote_offset = 0;
                local_offset = ssize / 2;
            }
            ring_scratch_offset = ssize / 2 / task->allgatherv_ring.num_rings;

            peer_block = get_recv_block(team, trank, tsize, frag_step, ring_id);
            ucc_tl_cuda_allgatherv_ring_size_offset(task, peer_block, frag, ring_id,
                                                    &block_size, &frag_size, &ring_frag_size,
                                                    &block_offset, &frag_offset, &ring_frag_offset);
            sbuf1 = PTR_OFFSET(TASK_SCRATCH(task, recvfrom), local_offset + ring_scratch_offset * ring_id);
            dbuf1 = PTR_OFFSET(rbuf, block_offset + frag_offset + ring_frag_offset);
            frag_count1 = ring_frag_size;
            peer_block = get_send_block(team, trank, tsize, frag_step, ring_id);
            ucc_tl_cuda_allgatherv_ring_size_offset(task, peer_block, frag, ring_id,
                                                    &block_size, &frag_size, &ring_frag_size,
                                                    &block_offset, &frag_offset, &ring_frag_offset);
            sbuf2 = PTR_OFFSET(TASK_SCRATCH(task, trank), local_offset + ring_scratch_offset * ring_id);
            dbuf2 = PTR_OFFSET(rbuf, block_offset + frag_offset + ring_frag_offset);
            frag_count2 = ring_frag_size;
        } else {
            send_step = get_rank_step(task, sendto, ring_id);
            recv_step = get_rank_step(task, recvfrom, ring_id);
            if ((send_step < step) || (recv_step < step)) {
                return UCC_INPROGRESS;
            }
            peer_block = get_send_block(team, trank, tsize, frag_step, ring_id);
            ucc_tl_cuda_allgatherv_ring_size_offset(task, peer_block, frag, ring_id,
                                                    &block_size, &frag_size, &ring_frag_size,
                                                    &block_offset, &frag_offset, &ring_frag_offset);
            if (step % 2) {
                remote_offset = ssize / 2;
                local_offset = 0;
            } else {
                remote_offset = 0;
                local_offset = ssize / 2;
            }
            ring_scratch_offset = ssize / 2 / task->allgatherv_ring.num_rings;

            if (frag_step == 0) {
                send_step = get_rank_step(task, get_send_to(team, sendto, tsize, ring_id), ring_id);
                if ((send_step < step)) {
                    return UCC_INPROGRESS;
                }

                if (UCC_IS_INPLACE(*args)) {
                    sbuf1 = PTR_OFFSET(rbuf, block_offset + frag_offset + ring_frag_offset);
                    sbuf2 = NULL;
                    dbuf2 = NULL;
                } else {
                    sbuf1 = PTR_OFFSET(task->allgatherv_ring.sbuf, frag_offset + ring_frag_offset);
                    sbuf2 = sbuf1;
                    dbuf2 = PTR_OFFSET(rbuf, block_offset + frag_offset + ring_frag_offset);
                }
                dbuf1 = PTR_OFFSET(TASK_SCRATCH(task, sendto),
                                   remote_offset + ring_scratch_offset * ring_id);
            } else {
                sbuf1  = PTR_OFFSET(TASK_SCRATCH(task, trank),
                                    local_offset + ring_scratch_offset * ring_id);
                sbuf2 = sbuf1;
                dbuf1 = PTR_OFFSET(TASK_SCRATCH(task, sendto),
                                remote_offset + ring_scratch_offset * ring_id);
                dbuf2 = PTR_OFFSET(rbuf,
                                block_offset + frag_offset + ring_frag_offset);
            }
            frag_count1 = ring_frag_size;
            frag_count2 = ring_frag_size;
        }
        if (sbuf1 == sbuf2) {
            exec_args.task_type = UCC_EE_EXECUTOR_TASK_TYPE_COPY_MULTI;
            exec_args.bufs[0] = sbuf1;
            exec_args.bufs[1] = dbuf1;
            exec_args.bufs[2] = dbuf2;
            exec_args.count   = frag_count1;

            st = ucc_ee_executor_task_post(exec, &exec_args,
                                           &task->allgatherv_ring.exec_task[ring_id * 2]);
            if (ucc_unlikely(st != UCC_OK)) {
                return st;
            }

        } else {
            exec_args.task_type = UCC_EE_EXECUTOR_TASK_TYPE_COPY;
            exec_args.bufs[0]   = dbuf1;
            exec_args.bufs[1]   = sbuf1;
            exec_args.count     = frag_count1;
            st = ucc_ee_executor_task_post(exec, &exec_args,
                                        &task->allgatherv_ring.exec_task[ring_id * 2]);
            if (ucc_unlikely(st != UCC_OK)) {
                return st;
            }
            if (dbuf2 != NULL) {
                exec_args.task_type = UCC_EE_EXECUTOR_TASK_TYPE_COPY;
                exec_args.bufs[0]   = dbuf2;
                exec_args.bufs[1]   = sbuf2;
                exec_args.count     = frag_count2;
                st = ucc_ee_executor_task_post(exec, &exec_args,
                                            &task->allgatherv_ring.exec_task[ring_id * 2 + 1]);
                if (ucc_unlikely(st != UCC_OK)) {
                    return st;
                }
            }
        }
    }

    return UCC_OK;
}

void ucc_tl_cuda_allgatherv_ring_progress(ucc_coll_task_t *coll_task)
{
    ucc_tl_cuda_task_t *task = ucc_derived_of(coll_task, ucc_tl_cuda_task_t);
    ucc_tl_cuda_team_t *team = TASK_TEAM(task);
    ucc_status_t st;
    int ring, num_done, polls;

    task->super.status = UCC_INPROGRESS;
    switch (task->allgatherv_ring.stage) {
    case RING_STAGE_SYNC:
        if (ucc_tl_cuda_get_sync(task) != UCC_OK) {
            task->super.status = UCC_INPROGRESS;
            return;
        }
        st = ucc_tl_cuda_allgatherv_ring_setup_start(task);
        if (st != UCC_OK) {
            task->super.status = st;
            return;
        }
        task->allgatherv_ring.stage = RING_STAGE_SETUP;
    case RING_STAGE_SETUP:
        st = ucc_tl_cuda_allgatherv_ring_setup_test(task);
        if (st != UCC_OK) {
            task->super.status = st;
            return;
        }
        task->allgatherv_ring.stage = RING_STAGE_RING;
    case RING_STAGE_RING:
        polls = 0;
        while (polls < 1) {
            num_done = 0;
            for (ring = 0; ring < task->allgatherv_ring.num_rings; ring++) {
                st = ucc_tl_cuda_allgatherv_ring_progress_ring(task, ring);
                if (ucc_unlikely(st < 0)) {
                    task->super.status = st;
                    return;
                } else if (st == UCC_OK) {
                    num_done++;
                }
            }
            if (num_done == task->allgatherv_ring.num_rings) {
                break;
            }
        }

        if (num_done != task->allgatherv_ring.num_rings) {
            return;
        }

        st = ucc_tl_cuda_shm_barrier_start(UCC_TL_TEAM_RANK(team), task->bar);
        if (ucc_unlikely(st != UCC_OK)) {
            task->super.status = st;
            return;
        }

        task->allgatherv_ring.stage = RING_STAGE_BARRIER;
    default:
        ucc_assert(task->allgatherv_ring.stage == RING_STAGE_BARRIER);
        break;
    }

    task->super.status = ucc_tl_cuda_shm_barrier_test(UCC_TL_TEAM_RANK(team),
                                                      task->bar);
    if (task->super.status == UCC_OK) {
        ucc_tl_cuda_put_sync(task);
    }
}

ucc_status_t ucc_tl_cuda_allgatherv_ring_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_cuda_task_t *task   = ucc_derived_of(coll_task, ucc_tl_cuda_task_t);
    ucc_tl_cuda_team_t *team   = TASK_TEAM(task);
    ucc_tl_cuda_lib_t  *lib    = UCC_TL_CUDA_TEAM_LIB(team);
    ucc_coll_args_t    *args   = &TASK_ARGS(task);
    ucc_rank_t          tsize  = UCC_TL_TEAM_SIZE(team);
    size_t              ssize  = lib->cfg.scratch_size;
    int                 nrings = lib->cfg.allgather_ring_max_rings;
    size_t              send_size;
    size_t              frag_size;
    ucc_rank_t          i;

    nrings = ucc_min(nrings, team->topo->num_rings);
    task->allgatherv_ring.sbuf         = args->src.info.buffer;
    task->allgatherv_ring.num_rings    = nrings;
    if (args->coll_type == UCC_COLL_TYPE_ALLGATHERV) {
        task->allgatherv_ring.rbuf     = args->dst.info_v.buffer;
    } else {
        task->allgatherv_ring.rbuf     = args->dst.info.buffer;
    }

    send_size = task->allgatherv_ring.get_count(task, 0);
    for (i = 1; i < tsize; i++) {
        send_size = ucc_max(send_size, task->allgatherv_ring.get_count(task, i));
    }

    if (send_size == 0) {
        task->super.status = UCC_OK;
        return ucc_task_complete(&task->super);
    }

    memset(task->allgatherv_ring.exec_task, 0,
           2 * nrings * sizeof(ucc_ee_executor_task_t*));

    send_size = ucc_dt_size(task->allgatherv_ring.dt) * send_size;
    frag_size = ucc_min(ssize /  2, send_size);
    task->allgatherv_ring.num_frags = ucc_div_round_up(send_size, frag_size);
    task->allgatherv_ring.stage     = RING_STAGE_SYNC;

    return ucc_progress_queue_enqueue(UCC_TL_CORE_CTX(team)->pq, &task->super);
}

size_t ucc_tl_cuda_allgatherv_get_count(const ucc_tl_cuda_task_t *task,
                                        ucc_rank_t block)
{
    const ucc_coll_args_t *args  = &TASK_ARGS(task);

    return ucc_coll_args_get_count(args, args->dst.info_v.counts, block);
}

size_t ucc_tl_cuda_allgatherv_get_offset(const ucc_tl_cuda_task_t *task,
                                         ucc_rank_t block)
{
    const ucc_coll_args_t *args  = &TASK_ARGS(task);

    return ucc_coll_args_get_displacement(args, args->dst.info_v.displacements,
                                          block);
}

ucc_status_t ucc_tl_cuda_allgatherv_ring_init(ucc_tl_cuda_task_t *task)
{
    ucc_coll_args_t  *args  = &TASK_ARGS(task);

    task->allgatherv_ring.get_count  = ucc_tl_cuda_allgatherv_get_count;
    task->allgatherv_ring.get_offset = ucc_tl_cuda_allgatherv_get_offset;
    task->allgatherv_ring.dt         = args->dst.info_v.datatype;

    task->super.flags               |= UCC_COLL_TASK_FLAG_EXECUTOR;
    task->super.post                 = ucc_tl_cuda_allgatherv_ring_start;
    task->super.triggered_post       = ucc_triggered_post;
    task->super.progress             = ucc_tl_cuda_allgatherv_ring_progress;
    task->super.finalize             = ucc_tl_cuda_allgatherv_ring_finalize;
    task->bar                        = TASK_BAR(task);

    return UCC_OK;
}
