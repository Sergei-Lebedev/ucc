/**
 * Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "config.h"
#include "tl_ucp.h"
#include "alltoallv.h"
#include "core/ucc_progress_queue.h"
#include "utils/ucc_math.h"
#include "utils/ucc_coll_utils.h"
#include "tl_ucp_sendrecv.h"

enum {
    ALLTOALLV_COMPRESS_STAGE_COMPRESS,
    ALLTOALLV_COMPRESS_STAGE_ALLTOALL,
    ALLTOALLV_COMPRESS_STAGE_DECOMPRESS,
};

static inline ucc_rank_t get_recv_peer(ucc_rank_t rank, ucc_rank_t size,
                                       ucc_rank_t step)
{
    return (rank + step) % size;
}

static inline ucc_rank_t get_send_peer(ucc_rank_t rank, ucc_rank_t size,
                                       ucc_rank_t step)
{
    return (rank - step + size) % size;
}

static void set_displacement(int displ64, const ucc_aint_t *displs, int idx,
                             size_t displ)
{
    if (displ64) {
        ((uint64_t *)displs)[idx] = displ;
    } else {
        ((uint32_t *)displs)[idx] = displ;
    }
}

static void set_count(int cnt64, const ucc_count_t *counts, int idx,
                      size_t count)
{
    if (cnt64) {
        ((uint64_t *)counts)[idx] = count;
    } else {
        ((uint32_t *)counts)[idx] = count;
    }
}

static size_t get_displacement(int displ64, const ucc_aint_t *displs, int idx)
{
    if (displ64) {
        return ((uint64_t *)displs)[idx];
    } else {
        return ((uint32_t *)displs)[idx];
    }
}

static size_t get_count(int cnt64, const ucc_count_t *counts, int idx)
{
    if (cnt64) {
        return ((uint64_t *)counts)[idx];
    } else {
        return ((uint32_t *)counts)[idx];
    }
}

void recv_completion_cb(void *request, ucs_status_t status,
                        const ucp_tag_recv_info_t *info,
                        void *user_data)
{
    ucc_tl_ucp_task_t *task  = (ucc_tl_ucp_task_t *)user_data;
    int                cnt64 = UCC_COLL_ARGS_COUNT64(&TASK_ARGS(task));

    if (ucc_unlikely(UCS_OK != status)) {
        tl_error(UCC_TASK_LIB(task), "failure in recv completion %s",
                 ucs_status_string(status));
        task->super.status = ucs_status_to_ucc_status(status);
    }
    task->tagged.recv_completed++;
    set_count(cnt64, task->alltoallv_compress.comp_dst_counts,
              UCC_TL_UCP_GET_SENDER(info->sender_tag), info->length);
    if (request) {
        ucp_request_free(request);
    }
}

ucc_status_t ucc_tl_ucp_alltoallv_pairwise_compress_alltoall(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t *task    = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_tl_ucp_team_t *team    = TASK_TEAM(task);
    ptrdiff_t          sbuf    = (ptrdiff_t)task->alltoallv_compress.scratch_src->addr;
    ptrdiff_t          rbuf    = (ptrdiff_t)task->alltoallv_compress.scratch_dst->addr;
    ucc_memory_type_t  smem    = TASK_ARGS(task).src.info_v.mem_type;
    ucc_memory_type_t  rmem    = TASK_ARGS(task).dst.info_v.mem_type;
    ucc_rank_t         grank   = UCC_TL_TEAM_RANK(team);
    ucc_rank_t         gsize   = UCC_TL_TEAM_SIZE(team);
    int                displ64 = UCC_COLL_ARGS_DISPL64(&TASK_ARGS(task));
    int                cnt64   = UCC_COLL_ARGS_COUNT64(&TASK_ARGS(task));
    int                polls   = 0;
    ucc_rank_t         peer;
    int                posts, nreqs;
    size_t             data_size, data_displ;

    posts    = UCC_TL_UCP_TEAM_LIB(team)->cfg.alltoallv_pairwise_num_posts;
    nreqs    = (posts > gsize || posts == 0) ? gsize : posts;
    while ((task->tagged.send_posted < gsize ||
            task->tagged.recv_posted < gsize) &&
           (polls++ < task->n_polls)) {
        ucp_worker_progress(UCC_TL_UCP_TEAM_CTX(team)->ucp_worker);
        while ((task->tagged.recv_posted < gsize) &&
               ((task->tagged.recv_posted - task->tagged.recv_completed) <
                nreqs)) {
            peer = get_recv_peer(grank, gsize, task->tagged.recv_posted);
            data_size  = get_count(cnt64, task->alltoallv_compress.comp_dst_counts, peer);
            data_displ = get_displacement(displ64, task->alltoallv_compress.comp_dst_displs, peer);
            UCPCHECK_GOTO(ucc_tl_ucp_recv_cb((void *)(rbuf + data_displ),
                                             data_size, rmem, peer, team, task,
                                             recv_completion_cb),
                          task, out);
            polls = 0;
        }
        while ((task->tagged.send_posted < gsize) &&
               ((task->tagged.send_posted - task->tagged.send_completed) <
                nreqs)) {
            peer = get_send_peer(grank, gsize, task->tagged.send_posted);
            data_size  = get_count(cnt64, task->alltoallv_compress.comp_src_counts, peer);
            data_displ = get_displacement(displ64, task->alltoallv_compress.comp_src_displs, peer);
            UCPCHECK_GOTO(ucc_tl_ucp_send_nb((void *)(sbuf + data_displ),
                                             data_size, smem, peer, team, task),
                          task, out);
            polls = 0;
        }
    }
    if ((task->tagged.send_posted < gsize) ||
        (task->tagged.recv_posted < gsize)) {
        return UCC_INPROGRESS;
    }

    return ucc_tl_ucp_test(task);
out:
    return task->super.status;
}

ucc_status_t ucc_tl_ucp_alltoallv_pairwise_compress_post_compress(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t *task = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_tl_ucp_team_t *team = TASK_TEAM(task);
    ucc_coll_args_t   *args = &TASK_ARGS(task);
    ucc_ee_executor_t *exec;
    ucc_ee_executor_task_args_t exec_args;
    ucc_status_t status;

    status = ucc_coll_task_get_executor(&task->super, &exec);
    if (ucc_unlikely(status != UCC_OK)) {
        return status;
    }

    exec_args.task_type                  = UCC_EE_EXECUTOR_TASK_COMPRESS;
    exec_args.compress.dt                = args->src.info_v.datatype;
    exec_args.compress.src               = args->src.info_v.buffer;
    exec_args.compress.dst               = task->alltoallv_compress.scratch_src->addr;
    exec_args.compress.size              = UCC_TL_TEAM_SIZE(team);
    exec_args.compress.src_counts        = args->src.info_v.counts;
    exec_args.compress.src_displacements = args->src.info_v.displacements;
    exec_args.compress.dst_counts        = task->alltoallv_compress.comp_src_counts;
    exec_args.compress.dst_displacements = task->alltoallv_compress.comp_src_displs;
    exec_args.compress.flags             = 0;
    if (UCC_COLL_ARGS_COUNT64(args)) {
        exec_args.compress.flags |= UCC_EE_TASK_COMPRESS_FLAG_COUNT64;
    }
    if (UCC_COLL_ARGS_DISPL64(args)) {
        exec_args.compress.flags |= UCC_EE_TASK_COMPRESS_FLAG_DISPLACEMENT64;
    }

    status = ucc_ee_executor_task_post(exec, &exec_args,
                                       &task->alltoallv_compress.compress_task);
    return status;
}

ucc_status_t ucc_tl_ucp_alltoallv_pairwise_compress_post_decompress(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t *task = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_tl_ucp_team_t *team = TASK_TEAM(task);
    ucc_coll_args_t   *args = &TASK_ARGS(task);
    ucc_ee_executor_t *exec;
    ucc_ee_executor_task_args_t exec_args;
    ucc_status_t status;

    status = ucc_coll_task_get_executor(&task->super, &exec);
    if (ucc_unlikely(status != UCC_OK)) {
        return status;
    }

    exec_args.task_type                  = UCC_EE_EXECUTOR_TASK_COMPRESS;
    exec_args.compress.flags             = UCC_EE_TASK_COMPRESS_FLAG_DECOMPRESS;
    exec_args.compress.src               = task->alltoallv_compress.scratch_dst->addr;
    exec_args.compress.src_counts        = task->alltoallv_compress.comp_dst_counts;
    exec_args.compress.src_displacements = task->alltoallv_compress.comp_dst_displs;
    exec_args.compress.dst               = args->dst.info_v.buffer;
    exec_args.compress.dst_counts        = args->dst.info_v.counts;
    exec_args.compress.dst_displacements = args->dst.info_v.displacements;
    exec_args.compress.dt                = args->dst.info_v.datatype;
    exec_args.compress.size              = UCC_TL_TEAM_SIZE(team);
    if (UCC_COLL_ARGS_COUNT64(args)) {
        exec_args.compress.flags |= UCC_EE_TASK_COMPRESS_FLAG_COUNT64;
    }
    if (UCC_COLL_ARGS_DISPL64(args)) {
        exec_args.compress.flags |= UCC_EE_TASK_COMPRESS_FLAG_DISPLACEMENT64;
    }
    status = ucc_ee_executor_task_post(exec, &exec_args,
                                       &task->alltoallv_compress.compress_task);
    return status;
}

void ucc_tl_ucp_alltoallv_pairwise_compress_progress(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t *task = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_status_t status;

    switch(task->alltoallv_compress.stage) {
    case ALLTOALLV_COMPRESS_STAGE_COMPRESS:
        status = ucc_ee_executor_task_test(task->alltoallv_compress.compress_task);
        if (status != UCC_OK) {
            if (status == UCC_OPERATION_INITIALIZED) {
                status = UCC_INPROGRESS;
            }
            task->super.status = status;
            return;
        }
        ucc_ee_executor_task_finalize(task->alltoallv_compress.compress_task);
        task->alltoallv_compress.stage = ALLTOALLV_COMPRESS_STAGE_ALLTOALL;
    case ALLTOALLV_COMPRESS_STAGE_ALLTOALL:
        status = ucc_tl_ucp_alltoallv_pairwise_compress_alltoall(coll_task);
        if (status != UCC_OK) {
            task->super.status = status;
            return;
        }
        status = ucc_tl_ucp_alltoallv_pairwise_compress_post_decompress(coll_task);
        if (status != UCC_OK) {
            task->super.status = status;
            return;
        }
        task->alltoallv_compress.stage = ALLTOALLV_COMPRESS_STAGE_DECOMPRESS;
    case ALLTOALLV_COMPRESS_STAGE_DECOMPRESS:
        status = ucc_ee_executor_task_test(task->alltoallv_compress.compress_task);
        if (status != UCC_OK) {
            if (status == UCC_OPERATION_INITIALIZED) {
                status = UCC_INPROGRESS;
            }
            task->super.status = status;
            return;
        }
        ucc_ee_executor_task_finalize(task->alltoallv_compress.compress_task);
    }

    UCC_TL_UCP_PROFILE_REQUEST_EVENT(coll_task,
                                     "ucp_alltoallv_pairwise_compress_done", 0);
    task->super.status = UCC_OK;
}

ucc_status_t ucc_tl_ucp_alltoallv_pairwise_compress_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t *task = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_tl_ucp_team_t *team = TASK_TEAM(task);
    ucc_status_t status;

    UCC_TL_UCP_PROFILE_REQUEST_EVENT(coll_task,
                                     "ucp_alltoallv_pairwise_compress_start",
                                     0);
    ucc_tl_ucp_task_reset(task, UCC_INPROGRESS);
    task->alltoallv_compress.stage = ALLTOALLV_COMPRESS_STAGE_COMPRESS;
    status = ucc_tl_ucp_alltoallv_pairwise_compress_post_compress(coll_task);
    if (status != UCC_OK) {
        return status;
    }

    return ucc_progress_queue_enqueue(UCC_TL_CORE_CTX(team)->pq, &task->super);
}

ucc_status_t ucc_tl_ucp_alltoallv_pairwise_compress_finalize(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t *task = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_status_t st;

    ucc_mc_free(task->alltoallv_compress.scratch_offsets);
    ucc_mc_free(task->alltoallv_compress.scratch_src);
    ucc_mc_free(task->alltoallv_compress.scratch_dst);

    st = ucc_tl_ucp_coll_finalize(&task->super);
    if (ucc_unlikely(st != UCC_OK)) {
        tl_error(UCC_TASK_LIB(task), "failed finalize collective");
    }
    return st;
}

ucc_status_t ucc_tl_ucp_alltoallv_pairwise_compress_init_common(ucc_tl_ucp_task_t *task)
{
    ucc_tl_ucp_team_t *team             = TASK_TEAM(task);
    ucc_rank_t         gsize            = UCC_TL_TEAM_SIZE(team);
    ucc_coll_args_t   *args             = &TASK_ARGS(task);
    int                displ64          = UCC_COLL_ARGS_DISPL64(args);
    int                cnt64            = UCC_COLL_ARGS_COUNT64(args);
    ucc_datatype_t     dt               = args->src.info_v.datatype;
    size_t             count_displ_size = 0;
    ucc_aint_t *comp_src_displ, *comp_dst_displ;
    ucc_count_t *comp_src_counts, *comp_dst_counts;
    size_t compress_size, msg_size, comp_size;;
    ucc_status_t status;
    ucc_rank_t i;

    if (args->src.info_v.datatype != args->dst.info_v.datatype) {
        return UCC_ERR_NOT_SUPPORTED;
    }

    if (args->src.info_v.datatype != UCC_DT_FLOAT16 &&
        args->src.info_v.datatype != UCC_DT_BFLOAT16 &&
        args->src.info_v.datatype != UCC_DT_FLOAT32) {
        return UCC_ERR_NOT_SUPPORTED;
    }

    if (args->src.info_v.mem_type != UCC_MEMORY_TYPE_CUDA ||
        args->dst.info_v.mem_type != UCC_MEMORY_TYPE_CUDA) {
        return UCC_ERR_NOT_SUPPORTED;
    }

    task->super.flags    |= UCC_COLL_TASK_FLAG_EXECUTOR;
    task->super.post     = ucc_tl_ucp_alltoallv_pairwise_compress_start;
    task->super.progress = ucc_tl_ucp_alltoallv_pairwise_compress_progress;
    task->super.finalize = ucc_tl_ucp_alltoallv_pairwise_compress_finalize;

    task->n_polls = ucc_min(1, task->n_polls);
    if (UCC_TL_UCP_TEAM_CTX(team)->cfg.pre_reg_mem) {
        if (args->flags & UCC_COLL_ARGS_FLAG_CONTIG_SRC_BUFFER) {
            ucc_tl_ucp_pre_register_mem(
                team, args->src.info_v.buffer,
                (ucc_coll_args_get_total_count(args, args->src.info_v.counts,
                                               gsize) *
                 ucc_dt_size(args->src.info_v.datatype)),
                args->src.info_v.mem_type);
        }

        if (args->flags & UCC_COLL_ARGS_FLAG_CONTIG_DST_BUFFER) {
            ucc_tl_ucp_pre_register_mem(
                team, args->dst.info_v.buffer,
                (ucc_coll_args_get_total_count(args, args->dst.info_v.counts,
                                               gsize) *
                 ucc_dt_size(args->dst.info_v.datatype)),
                args->dst.info_v.mem_type);
        }
    }

    task->alltoallv_compress.scratch_offsets = NULL;
    task->alltoallv_compress.scratch_src     = NULL;
    task->alltoallv_compress.scratch_dst     = NULL;

    if (cnt64) {
        count_displ_size += 8;
    } else {
        count_displ_size += 4;
    }

    if (displ64) {
        count_displ_size += 8;
    } else {
        count_displ_size += 4;
    }

    status = ucc_mc_alloc(&task->alltoallv_compress.scratch_offsets,
                          2 * gsize * count_displ_size, UCC_MEMORY_TYPE_HOST);
    if (ucc_unlikely(status != UCC_OK)) {
        goto exit;
    }

    comp_src_counts = task->alltoallv_compress.scratch_offsets->addr;

    if (cnt64) {
        comp_dst_counts = PTR_OFFSET(comp_src_counts, gsize * 8);
        comp_src_displ  = PTR_OFFSET(comp_dst_counts, gsize * 8);
    } else {
        comp_dst_counts = PTR_OFFSET(comp_src_counts, gsize * 4);
        comp_src_displ  = PTR_OFFSET(comp_dst_counts, gsize * 4);
    }

    if (displ64) {
        comp_dst_displ = PTR_OFFSET(comp_src_displ, gsize * 8);
    } else {
        comp_dst_displ = PTR_OFFSET(comp_src_displ, gsize * 4);
    }

    compress_size = 0;
    set_displacement(displ64, comp_src_displ, 0, 0);
    for (i = 1; i < gsize; i++) {
        msg_size = ucc_coll_args_get_count(args, args->src.info_v.counts, i - 1);
        status = ucc_ee_get_compress_size(UCC_EE_CUDA_STREAM, msg_size, dt, &comp_size);
        if (ucc_unlikely(status != UCC_OK)) {
            goto exit;
        }
        compress_size += comp_size;
        set_displacement(displ64, comp_src_displ, i,
                         get_displacement(displ64, comp_src_displ, i - 1) + comp_size);
    }
    msg_size = ucc_coll_args_get_count(args, args->src.info_v.counts, gsize - 1);
    ucc_ee_get_compress_size(UCC_EE_CUDA_STREAM, msg_size, dt, &comp_size);
    compress_size += comp_size;
    status = ucc_mc_alloc(&task->alltoallv_compress.scratch_src, compress_size,
                          args->src.info_v.mem_type);
    if (ucc_unlikely(status != UCC_OK)) {
        goto exit;
    }

    compress_size = 0;
    set_displacement(displ64, comp_dst_displ, 0, 0);
    for (i = 1; i < gsize; i++) {
        msg_size = ucc_coll_args_get_count(args, args->dst.info_v.counts, i - 1);
        status = ucc_ee_get_compress_size(UCC_EE_CUDA_STREAM, msg_size, dt, &comp_size);
        if (ucc_unlikely(status != UCC_OK)) {
            goto exit;
        }
        compress_size += comp_size;
        set_displacement(displ64, comp_dst_displ, i,
                         get_displacement(displ64, comp_dst_displ, i - 1) + comp_size);
        set_count(cnt64, comp_dst_counts, i - 1, comp_size);
    }
    msg_size = ucc_coll_args_get_count(args, args->dst.info_v.counts, gsize - 1);
    ucc_ee_get_compress_size(UCC_EE_CUDA_STREAM, msg_size, dt, &comp_size);
    compress_size += comp_size;
    set_count(cnt64, comp_dst_counts, gsize - 1, comp_size);

    status = ucc_mc_alloc(&task->alltoallv_compress.scratch_dst, compress_size,
                          args->dst.info_v.mem_type);
    if (ucc_unlikely(status != UCC_OK)) {
        goto exit;
    }

    task->alltoallv_compress.comp_src_counts = comp_src_counts;
    task->alltoallv_compress.comp_src_displs = comp_src_displ;
    task->alltoallv_compress.comp_dst_counts = comp_dst_counts;
    task->alltoallv_compress.comp_dst_displs = comp_dst_displ;

    return UCC_OK;
exit:
    tl_warn(UCC_TASK_LIB(task), "failed to init alltoall with compression");
    if (task->alltoallv_compress.scratch_offsets) {
        ucc_mc_free(task->alltoallv_compress.scratch_offsets);
    }
    if (task->alltoallv_compress.scratch_src) {
        ucc_mc_free(task->alltoallv_compress.scratch_src);
    }
    if (task->alltoallv_compress.scratch_dst) {
        ucc_mc_free(task->alltoallv_compress.scratch_dst);
    }
    return status;
}
