/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "tl_nccl_coll.h"
#include "core/ucc_mc.h"
#include "core/ucc_ee.h"
#include "utils/ucc_math.h"
#include "utils/ucc_coll_utils.h"
#include "utils/profile/ucc_profile.h"
#include <sys/time.h>

#define ncclOpUnsupported (ncclNumOps + 1)
#define ncclDataTypeUnsupported (ncclNumTypes + 1)

struct timeval start_coll[3];

ncclDataType_t ucc_to_nccl_dtype[] = {
    [UCC_DT_INT8]        = (ncclDataType_t)ncclInt8,
    [UCC_DT_INT16]       = (ncclDataType_t)ncclDataTypeUnsupported,
    [UCC_DT_INT32]       = (ncclDataType_t)ncclInt32,
    [UCC_DT_INT64]       = (ncclDataType_t)ncclInt64,
    [UCC_DT_INT128]      = (ncclDataType_t)ncclDataTypeUnsupported,
    [UCC_DT_UINT8]       = (ncclDataType_t)ncclUint8,
    [UCC_DT_UINT16]      = (ncclDataType_t)ncclDataTypeUnsupported,
    [UCC_DT_UINT32]      = (ncclDataType_t)ncclUint32,
    [UCC_DT_UINT64]      = (ncclDataType_t)ncclUint64,
    [UCC_DT_UINT128]     = (ncclDataType_t)ncclDataTypeUnsupported,
    [UCC_DT_FLOAT16]     = (ncclDataType_t)ncclFloat16,
    [UCC_DT_FLOAT32]     = (ncclDataType_t)ncclFloat32,
    [UCC_DT_FLOAT64]     = (ncclDataType_t)ncclFloat64,
    [UCC_DT_USERDEFINED] = (ncclDataType_t)ncclDataTypeUnsupported,
    [UCC_DT_OPAQUE]      = (ncclDataType_t)ncclDataTypeUnsupported,
};

ncclRedOp_t ucc_to_nccl_reduce_op[] = {
    [UCC_OP_USERDEFINED] = (ncclRedOp_t)ncclOpUnsupported,
    [UCC_OP_SUM]         = (ncclRedOp_t)ncclSum,
    [UCC_OP_PROD]        = (ncclRedOp_t)ncclProd,
    [UCC_OP_MAX]         = (ncclRedOp_t)ncclMax,
    [UCC_OP_MIN]         = (ncclRedOp_t)ncclMin,
    [UCC_OP_LAND]        = (ncclRedOp_t)ncclOpUnsupported,
    [UCC_OP_LOR]         = (ncclRedOp_t)ncclOpUnsupported,
    [UCC_OP_LXOR]        = (ncclRedOp_t)ncclOpUnsupported,
    [UCC_OP_BAND]        = (ncclRedOp_t)ncclOpUnsupported,
    [UCC_OP_BOR]         = (ncclRedOp_t)ncclOpUnsupported,
    [UCC_OP_BXOR]        = (ncclRedOp_t)ncclOpUnsupported,
    [UCC_OP_MAXLOC]      = (ncclRedOp_t)ncclOpUnsupported,
    [UCC_OP_MINLOC]      = (ncclRedOp_t)ncclOpUnsupported,
};

static inline ucc_status_t ucc_nccl_check_dt_supported(ucc_datatype_t dt1,
                                                       ucc_datatype_t dt2)
{
    if (ucc_unlikely((dt1 != dt2) ||
                     (ucc_to_nccl_dtype[dt1] == ncclDataTypeUnsupported))) {
        return UCC_ERR_NOT_SUPPORTED;
    }
    return UCC_OK;
}

ucc_status_t ucc_tl_nccl_collective_progress(ucc_coll_task_t *coll_task)
{
    ucc_tl_nccl_task_t *task = ucc_derived_of(coll_task, ucc_tl_nccl_task_t);
    ucc_status_t status;

    status = ucc_mc_ee_event_test(task->completed, UCC_EE_CUDA_STREAM);
    coll_task->super.status = status;

    if (status == UCC_OK) {
        struct timeval end_coll;
        int elapsed;
        int rank = task->team->rank;
        gettimeofday(&end_coll, NULL);
        switch(task->args.coll_type) {
            case UCC_COLL_TYPE_ALLREDUCE:
                elapsed = ((end_coll.tv_sec - start_coll[1].tv_sec) * 1000000) +
                          (end_coll.tv_usec - start_coll[1].tv_usec);
                if (rank == 0){
                    printf("time for ar: %d\n", elapsed);
                }
                UCC_PROFILE_REQUEST_EVENT(task, "nccl_allreduce_finish", 0);
                break;
            case UCC_COLL_TYPE_REDUCE_SCATTER:
                elapsed = ((end_coll.tv_sec - start_coll[0].tv_sec) * 1000000) +
                          (end_coll.tv_usec - start_coll[0].tv_usec);
                if (rank == 0){
                    printf("time for rs: %d\n", elapsed);
                }
                UCC_PROFILE_REQUEST_EVENT(task, "nccl_reducescatter_finish", 0);
                break;
            case UCC_COLL_TYPE_ALLGATHER:
                elapsed = ((end_coll.tv_sec - start_coll[0].tv_sec) * 1000000) +
                          (end_coll.tv_usec - start_coll[0].tv_usec);
                if (rank == 0){
                    printf("time for ag: %d\n", elapsed);
                }
                UCC_PROFILE_REQUEST_EVENT(task, "nccl_allgather_finish", 0);
                break;
            default:
                break;
        }
    }
    return status;
}

ucc_status_t ucc_tl_nccl_alltoall_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_nccl_task_t *task   = ucc_derived_of(coll_task, ucc_tl_nccl_task_t);
    ucc_tl_nccl_team_t *team   = task->team;
    ucc_ee_h            ee     = coll_task->ee;
    cudaStream_t        stream = (ee) ? (cudaStream_t) ee->ee_context : team->stream;
    ucc_rank_t          gsize  = team->size;
    ucc_status_t        status = UCC_OK;
    ptrdiff_t           sbuf   = (ptrdiff_t)task->args.src.info.buffer;
    ptrdiff_t           rbuf   = (ptrdiff_t)task->args.dst.info.buffer;
    size_t data_size;
    ucc_rank_t peer;

    task->super.super.status = UCC_INPROGRESS;
    data_size = (size_t)task->args.src.info.count *
                ucc_dt_size(task->args.src.info.datatype);
    if (data_size == 0) {
        task->super.super.status = UCC_OK;
        return UCC_OK;
    }
    NCCLCHECK_GOTO(ncclGroupStart(), exit_coll, status, UCC_TL_TEAM_LIB(team));
    for (peer = 0; peer < gsize; peer++) {
        NCCLCHECK_GOTO(ncclSend((void *)(sbuf + peer * data_size), data_size,
                                ncclChar, peer, team->nccl_comm, stream),
                       exit_coll, status, UCC_TL_TEAM_LIB(team));
        NCCLCHECK_GOTO(ncclRecv((void *)(rbuf + peer * data_size), data_size,
                                ncclChar, peer, team->nccl_comm, stream),
                       exit_coll, status, UCC_TL_TEAM_LIB(team));
    }
    NCCLCHECK_GOTO(ncclGroupEnd(), exit_coll, status, UCC_TL_TEAM_LIB(team));
    status = ucc_mc_ee_event_post(stream, task->completed, UCC_EE_CUDA_STREAM);
exit_coll:
    if (status == UCC_OK) {
        ucc_progress_enqueue(UCC_TL_CORE_CTX(team)->pq, &task->super);
    } else if (ucc_unlikely(status < 0)) {
        task->super.super.status = status;
    }
    return status;
}

ucc_status_t ucc_tl_nccl_alltoall_init(ucc_tl_nccl_task_t *task)
{
    if (UCC_IS_INPLACE(task->args)) {
        tl_error(UCC_TL_TEAM_LIB(task->team),
                 "inplace alltoallv is not supported");
        return UCC_ERR_NOT_SUPPORTED;
    }
    if ((task->args.src.info.datatype == UCC_DT_USERDEFINED) ||
        (task->args.dst.info.datatype == UCC_DT_USERDEFINED)) {
        tl_error(UCC_TL_TEAM_LIB(task->team),
                 "user defined datatype is not supported");
        return UCC_ERR_NOT_SUPPORTED;
    }
    task->super.post     = ucc_tl_nccl_alltoall_start;
    task->super.progress = ucc_tl_nccl_collective_progress;
    return UCC_OK;
}

ucc_status_t ucc_tl_nccl_alltoallv_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_nccl_task_t *task   = ucc_derived_of(coll_task, ucc_tl_nccl_task_t);
    ucc_tl_nccl_team_t *team   = task->team;
    ucc_ee_h            ee     = coll_task->ee;
    cudaStream_t        stream = (ee) ? (cudaStream_t) ee->ee_context : team->stream;
    ucc_status_t        status = UCC_OK;
    ptrdiff_t           sbuf   = (ptrdiff_t)task->args.src.info_v.buffer;
    ptrdiff_t           rbuf   = (ptrdiff_t)task->args.dst.info_v.buffer;
    size_t sdt_size, rdt_size, count, displ;
    ucc_rank_t peer;

    task->super.super.status = UCC_INPROGRESS;
    sdt_size = ucc_dt_size(task->args.src.info_v.datatype);
    rdt_size = ucc_dt_size(task->args.dst.info_v.datatype);
    NCCLCHECK_GOTO(ncclGroupStart(), exit_coll, status, UCC_TL_TEAM_LIB(team));
    for (peer = 0; peer < team->size; peer++) {
        count = ucc_coll_args_get_count(&task->args,
                                        task->args.src.info_v.counts, peer);
        if (count != 0) {
            displ = ucc_coll_args_get_displacement(&task->args,
                        task->args.src.info_v.displacements, peer);
            NCCLCHECK_GOTO(ncclSend((void *)(sbuf + displ * sdt_size),
                                    count * sdt_size, ncclChar, peer,
                                    team->nccl_comm, stream),
                        exit_coll, status, UCC_TL_TEAM_LIB(team));
        }
        count = ucc_coll_args_get_count(&task->args,
                                        task->args.dst.info_v.counts, peer);
        if (count != 0) {
            displ = ucc_coll_args_get_displacement(&task->args,
                        task->args.dst.info_v.displacements, peer);
            NCCLCHECK_GOTO(ncclRecv((void *)(rbuf + displ * rdt_size),
                                    count * rdt_size, ncclChar, peer,
                                    team->nccl_comm, stream),
                        exit_coll, status, UCC_TL_TEAM_LIB(team));
        }
    }
    NCCLCHECK_GOTO(ncclGroupEnd(), exit_coll, status, UCC_TL_TEAM_LIB(team));
    status = ucc_mc_ee_event_post(stream, task->completed, UCC_EE_CUDA_STREAM);
exit_coll:
    if (status == UCC_OK) {
        ucc_progress_enqueue(UCC_TL_CORE_CTX(team)->pq, &task->super);
    } else if (ucc_unlikely(status < 0)) {
        task->super.super.status = status;
    }
    return status;
}

ucc_status_t ucc_tl_nccl_alltoallv_init(ucc_tl_nccl_task_t *task)
{
    if (UCC_IS_INPLACE(task->args)) {
        tl_error(UCC_TL_TEAM_LIB(task->team),
                 "inplace alltoall is not supported");
        return UCC_ERR_NOT_SUPPORTED;
    }
    if ((task->args.src.info_v.datatype == UCC_DT_USERDEFINED) ||
        (task->args.dst.info_v.datatype == UCC_DT_USERDEFINED)) {
        tl_error(UCC_TL_TEAM_LIB(task->team),
                 "user defined datatype is not supported");
        return UCC_ERR_NOT_SUPPORTED;
    }
    task->super.post     = ucc_tl_nccl_alltoallv_start;
    task->super.progress = ucc_tl_nccl_collective_progress;
    return UCC_OK;
}

ucc_status_t ucc_tl_nccl_allreduce_start(ucc_coll_task_t *coll_task)
{

    ucc_tl_nccl_task_t *task   = ucc_derived_of(coll_task, ucc_tl_nccl_task_t);
    ucc_tl_nccl_lib_t *nccl_lib = ucc_derived_of(task->team->super.super.context->lib,
                                                 ucc_tl_nccl_lib_t);
    ucc_tl_nccl_team_t *team   = task->team;
//    ucc_ee_h            ee     = coll_task->ee;
//    cudaStream_t        stream = (ee) ? (cudaStream_t) ee->ee_context : team->stream;
    void               *dst    = task->args.dst.info.buffer;
    void               *src    = UCC_IS_INPLACE(task->args) ?
                                    task->args.dst.info.buffer:
                                    task->args.src.info.buffer;
    ucc_status_t        status = UCC_OK;
    ncclDataType_t      dt     = ucc_to_nccl_dtype[
                                    task->args.src.info.datatype];
    ncclRedOp_t         op     = ucc_to_nccl_reduce_op[
                                    task->args.reduce.predefined_op];
    size_t              count  = task->args.src.info.count;

    gettimeofday(&start_coll[1], NULL);
    UCC_PROFILE_REQUEST_EVENT(task, "nccl_allreduce_start", 0);
    task->super.super.status = UCC_INPROGRESS;
    if (nccl_lib->cfg.pp_allreduce == 0) {
        NCCLCHECK_GOTO(ncclAllReduce(src, dst, count, dt, op, team->nccl_comm,
                                     team->stream),
                    exit_coll, status, UCC_TL_TEAM_LIB(team));
        status = ucc_mc_ee_event_post(team->stream, task->completed, UCC_EE_CUDA_STREAM);
    } else {
        NCCLCHECK_GOTO(ncclAllReduce(src, dst, count, dt, op, team->nccl_comms[1],
                                    team->streams[1]),
                    exit_coll, status, UCC_TL_TEAM_LIB(team));
        status = ucc_mc_ee_event_post(team->streams[1], task->completed, UCC_EE_CUDA_STREAM);
    }



exit_coll:
    if (status == UCC_OK) {
        ucc_progress_enqueue(UCC_TL_CORE_CTX(team)->pq, &task->super);
    } else if (ucc_unlikely(status < 0)) {
        task->super.super.status = status;
    }
    return status;
}

ucc_status_t ucc_tl_nccl_allreduce_start_pp(ucc_coll_task_t *coll_task)
{
    ucc_tl_nccl_task_t *task   = ucc_derived_of(coll_task, ucc_tl_nccl_task_t);
    ucc_tl_nccl_team_t *team   = task->team;
    ucc_ee_h            ee     = coll_task->ee;
    cudaStream_t        stream = (ee) ? (cudaStream_t) ee->ee_context : team->stream;
    void               *dst    = task->args.dst.info.buffer;
    void               *src    = UCC_IS_INPLACE(task->args) ?
                                    task->args.dst.info.buffer:
                                    task->args.src.info.buffer;
    ucc_status_t        status = UCC_OK;
    ncclDataType_t      dt     = ucc_to_nccl_dtype[
                                    task->args.src.info.datatype];
    ncclRedOp_t         op     = ucc_to_nccl_reduce_op[
                                    task->args.reduce.predefined_op];
    size_t              count  = task->args.src.info.count;
    size_t              chunk_count  = count / team->local_size;
    size_t              chunk_size = chunk_count * ucc_dt_size(task->args.src.info.datatype);


    task->super.super.status = UCC_INPROGRESS;

    NCCLCHECK_GOTO(ncclReduceScatter(src, PTR_OFFSET(dst, chunk_size*team->local_rank),
                                     chunk_count, dt, op,
                                     team->nccl_comms[0], stream),
                   exit_coll, status, UCC_TL_TEAM_LIB(team));
    NCCLCHECK_GOTO(ncclAllReduce(PTR_OFFSET(dst, chunk_size*team->local_rank),
                                 PTR_OFFSET(dst, chunk_size*team->local_rank),
                                 chunk_count, dt, op,
                                 team->nccl_comms[1], stream),
                   exit_coll, status, UCC_TL_TEAM_LIB(team));

    NCCLCHECK_GOTO(ncclAllGather(PTR_OFFSET(dst, chunk_size*team->local_rank), dst,
                                 chunk_count, dt, team->nccl_comms[2],
                                 stream),
                   exit_coll, status, UCC_TL_TEAM_LIB(team));
    status = ucc_mc_ee_event_post(stream, task->completed, UCC_EE_CUDA_STREAM);
exit_coll:
    if (status == UCC_OK) {
        ucc_progress_enqueue(UCC_TL_CORE_CTX(team)->pq, &task->super);
    } else if (ucc_unlikely(status < 0)) {
        task->super.super.status = status;
    }
    return status;
}

ucc_status_t ucc_tl_nccl_allreduce_init(ucc_tl_nccl_task_t *task)
{
    // ucc_tl_nccl_lib_t *nccl_lib = ucc_derived_of(task->team->super.super.context->lib,
    //                                              ucc_tl_nccl_lib_t);

    if ((task->args.mask & UCC_COLL_ARGS_FIELD_USERDEFINED_REDUCTIONS) ||
        (ucc_to_nccl_reduce_op[task->args.reduce.predefined_op] ==
         ncclOpUnsupported)) {
        tl_error(UCC_TL_TEAM_LIB(task->team),
                 "reduction operation is not supported");
        return UCC_ERR_NOT_SUPPORTED;
    }
    if (UCC_OK != ucc_nccl_check_dt_supported(task->args.src.info.datatype,
                                              task->args.src.info.datatype)) {
        tl_error(UCC_TL_TEAM_LIB(task->team),
                 "dataype is not supported");
        return UCC_ERR_NOT_SUPPORTED;
    }
    UCC_PROFILE_REQUEST_NEW(task, "tl_nccl_task", 0);

    // if (nccl_lib->cfg.pp_allreduce == 0) {
    //     task->super.post     = ucc_tl_nccl_allreduce_start;
    // } else if (nccl_lib->cfg.pp_allreduce == 1) {
    //     task->super.post     = ucc_tl_nccl_allreduce_start_pp;
    // }
    task->super.post     = ucc_tl_nccl_allreduce_start;
    task->super.progress = ucc_tl_nccl_collective_progress;
    return UCC_OK;
}

ucc_status_t ucc_tl_nccl_reduce_scatter_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_nccl_task_t *task   = ucc_derived_of(coll_task, ucc_tl_nccl_task_t);
    ucc_tl_nccl_team_t *team   = task->team;
//    ucc_ee_h            ee     = coll_task->ee;
//    cudaStream_t        stream = (ee) ? (cudaStream_t) ee->ee_context : team->stream;
    void               *dst    = task->args.dst.info.buffer;
    void               *src    = UCC_IS_INPLACE(task->args) ?
                                    task->args.dst.info.buffer:
                                    task->args.src.info.buffer;
    ucc_status_t        status = UCC_OK;
    ncclDataType_t      dt     = ucc_to_nccl_dtype[
                                    task->args.src.info.datatype];
    ncclRedOp_t         op     = ucc_to_nccl_reduce_op[
                                    task->args.reduce.predefined_op];
    size_t              count  = task->args.src.info.count;

    task->super.super.status = UCC_INPROGRESS;
    gettimeofday(&start_coll[0], NULL);
    UCC_PROFILE_REQUEST_EVENT(task, "nccl_reducescatter_start", 0);
    // fprintf(stdout, "rank %d starting reduce scatter task %p\n", team->rank, task);
    NCCLCHECK_GOTO(ncclReduceScatter(src, dst, count, dt, op,
                                     team->nccl_comms[0], team->streams[0]),
                   exit_coll, status, UCC_TL_TEAM_LIB(team));
    status = ucc_mc_ee_event_post(team->streams[0], task->completed, UCC_EE_CUDA_STREAM);
exit_coll:
    if (status == UCC_OK) {
        ucc_progress_enqueue(UCC_TL_CORE_CTX(team)->pq, &task->super);
    } else if (ucc_unlikely(status < 0)) {
        task->super.super.status = status;
    }
    return status;
}

ucc_status_t ucc_tl_nccl_reduce_scatter_init(ucc_tl_nccl_task_t *task)
{
    if ((task->args.mask & UCC_COLL_ARGS_FIELD_USERDEFINED_REDUCTIONS) ||
        (ucc_to_nccl_reduce_op[task->args.reduce.predefined_op] ==
         ncclOpUnsupported)) {
        tl_error(UCC_TL_TEAM_LIB(task->team),
                 "reduction operation is not supported");
        return UCC_ERR_NOT_SUPPORTED;
    }
    if (UCC_OK != ucc_nccl_check_dt_supported(task->args.src.info.datatype,
                                              task->args.src.info.datatype)) {
        tl_error(UCC_TL_TEAM_LIB(task->team),
                 "dataype is not supported");
        return UCC_ERR_NOT_SUPPORTED;
    }
    UCC_PROFILE_REQUEST_NEW(task, "tl_nccl_task", 0);

    task->super.post     = ucc_tl_nccl_reduce_scatter_start;
    task->super.progress = ucc_tl_nccl_collective_progress;
    return UCC_OK;
}

ucc_status_t ucc_tl_nccl_allgather_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_nccl_task_t *task   = ucc_derived_of(coll_task, ucc_tl_nccl_task_t);
    ucc_tl_nccl_team_t *team   = task->team;
//    ucc_ee_h            ee     = coll_task->ee;
//    cudaStream_t        stream = (ee) ? (cudaStream_t) ee->ee_context : team->stream;
    void               *dst    = task->args.dst.info.buffer;
    void               *src    = task->args.src.info.buffer;
    ncclDataType_t      dt     = ucc_to_nccl_dtype[
                                    task->args.dst.info.datatype];
    ucc_status_t        status = UCC_OK;
    size_t              count  = task->args.dst.info.count;

    if (UCC_IS_INPLACE(task->args)) {
        src = (void*)((ptrdiff_t)task->args.dst.info.buffer +
               count * ucc_dt_size(task->args.dst.info.datatype) * team->local_rank);
    }
    task->super.super.status = UCC_INPROGRESS;
    gettimeofday(&start_coll[2], NULL);
    UCC_PROFILE_REQUEST_EVENT(task, "nccl_allgather_start", 0);
    NCCLCHECK_GOTO(ncclAllGather(src, dst, count, dt, team->nccl_comms[2], team->streams[2]),
                   exit_coll, status, UCC_TL_TEAM_LIB(team));
    status = ucc_mc_ee_event_post(team->streams[2], task->completed, UCC_EE_CUDA_STREAM);
exit_coll:
    if (status == UCC_OK) {
        ucc_progress_enqueue(UCC_TL_CORE_CTX(team)->pq, &task->super);
    } else if (ucc_unlikely(status < 0)) {
        task->super.super.status = status;
    }
    return status;
}

ucc_status_t ucc_tl_nccl_allgather_init(ucc_tl_nccl_task_t *task)
{
    ucc_datatype_t dt1 = UCC_IS_INPLACE(task->args) ?
                            task->args.dst.info.datatype :
                            task->args.src.info.datatype;
    ucc_datatype_t dt2 = task->args.dst.info.datatype;

    if (UCC_OK != ucc_nccl_check_dt_supported(dt1, dt2)) {
        /* TODO: can we use ncclChar if datatype is not supported? */
        tl_error(UCC_TL_TEAM_LIB(task->team),
                 "dataype is not supported");
        return UCC_ERR_NOT_SUPPORTED;
    }
    UCC_PROFILE_REQUEST_NEW(task, "tl_nccl_task", 0);

    task->super.post     = ucc_tl_nccl_allgather_start;
    task->super.progress = ucc_tl_nccl_collective_progress;
    return UCC_OK;
}

ucc_status_t ucc_tl_nccl_coll_finalize(ucc_coll_task_t *coll_task)
{
    ucc_tl_nccl_task_t *task  = ucc_derived_of(coll_task, ucc_tl_nccl_task_t);
    ucc_status_t       status = UCC_OK ;

    tl_info(UCC_TL_TEAM_LIB(task->team), "finalizing coll task %p", task);
    UCC_PROFILE_REQUEST_FREE(task);
    ucc_mc_ee_destroy_event(task->completed, UCC_EE_CUDA_STREAM);
    ucc_mpool_put(task);
    return status;
}

ucc_status_t ucc_tl_nccl_allreduce_schedule_init_frag(ucc_base_coll_args_t *coll_args,
                                                      ucc_base_team_t *team,
                                                      ucc_schedule_frag_t **frag_p)
{
    ucc_tl_nccl_team_t   *nccl_team = ucc_derived_of(team, ucc_tl_nccl_team_t);
    ucc_tl_nccl_context_t *nccl_ctx  = ucc_derived_of(team->context,
                                                      ucc_tl_nccl_context_t);
    ucc_schedule_frag_t *frag = ucc_malloc(sizeof(*frag), "nccl ar frag");
    ucc_status_t       status = UCC_OK ;
    ucc_tl_nccl_task_t  *task, *rs_task, *ar_task;

    ucc_schedule_frag_init(frag, team->context->ucc_context);

    task = ucc_mpool_get(&nccl_ctx->req_mp);
    ucc_coll_task_init(&task->super);
    memcpy(&task->args, &coll_args->args, sizeof(ucc_coll_args_t));
    task->args.coll_type = UCC_COLL_TYPE_REDUCE_SCATTER;
    task->team = nccl_team;
    task->super.finalize = ucc_tl_nccl_coll_finalize;
    status = ucc_mc_ee_create_event((void **)&task->completed, UCC_EE_CUDA_STREAM);
    if (ucc_unlikely(status != UCC_OK)) {
        tl_error(UCC_TL_TEAM_LIB(nccl_team), "failed to get ee event");
    }
    status = ucc_tl_nccl_reduce_scatter_init(task);
    if (UCC_OK != status) {
        tl_error(UCC_TL_TEAM_LIB(nccl_team), "failed to init reduce_scatter task");
        return status;
    }
    ucc_coll_task_init_dependent(&task->super, 1);
    ucc_schedule_frag_add_task(frag, &task->super);
    ucc_event_manager_subscribe(&frag->super.super.em, UCC_EVENT_SCHEDULE_STARTED,
                                &task->super);
    rs_task = task;

    task = ucc_mpool_get(&nccl_ctx->req_mp);
    ucc_coll_task_init(&task->super);
    memcpy(&task->args, &coll_args->args, sizeof(ucc_coll_args_t));
    task->team = nccl_team;
    task->super.finalize = ucc_tl_nccl_coll_finalize;
    status = ucc_mc_ee_create_event((void **)&task->completed, UCC_EE_CUDA_STREAM);
    if (ucc_unlikely(status != UCC_OK)) {
        tl_error(UCC_TL_TEAM_LIB(nccl_team), "failed to get ee event");
    }
    task->args.coll_type = UCC_COLL_TYPE_ALLREDUCE;
    task->args.mask |= UCC_COLL_ARGS_FIELD_FLAGS;
    task->args.flags |= UCC_COLL_ARGS_FLAG_IN_PLACE;
    status = ucc_tl_nccl_allreduce_init(task);
    if (UCC_OK != status) {
        tl_error(UCC_TL_TEAM_LIB(nccl_team), "failed to init allreduce task");
        return status;
    }
    ucc_coll_task_init_dependent(&task->super, 1);
    ucc_schedule_frag_add_task(frag, &task->super);
    ucc_event_manager_subscribe(&rs_task->super.em, UCC_EVENT_COMPLETED, &task->super);
    ar_task = task;

    task = ucc_mpool_get(&nccl_ctx->req_mp);
    ucc_coll_task_init(&task->super);
    memcpy(&task->args, &coll_args->args, sizeof(ucc_coll_args_t));
    task->team = nccl_team;
    task->super.finalize = ucc_tl_nccl_coll_finalize;
    status = ucc_mc_ee_create_event((void **)&task->completed, UCC_EE_CUDA_STREAM);
    if (ucc_unlikely(status != UCC_OK)) {
        tl_error(UCC_TL_TEAM_LIB(nccl_team), "failed to get ee event");
    }
    task->args.coll_type = UCC_COLL_TYPE_ALLGATHER;
    task->args.mask |= UCC_COLL_ARGS_FIELD_FLAGS;
    task->args.flags |= UCC_COLL_ARGS_FLAG_IN_PLACE;
    status = ucc_tl_nccl_allgather_init(task);
    if (UCC_OK != status) {
        tl_error(UCC_TL_TEAM_LIB(nccl_team), "failed to init allgather task");
        return status;
    }
    ucc_coll_task_init_dependent(&task->super, 1);
    ucc_schedule_frag_add_task(frag, &task->super);
    ucc_event_manager_subscribe(&ar_task->super.em, UCC_EVENT_COMPLETED, &task->super);
    *frag_p = frag;
    return UCC_OK;
}

ucc_status_t ucc_tl_nccl_allreduce_schedule_finalize(ucc_schedule_frag_t *frag)
{
    ucc_free(frag);
    return UCC_OK;
}

ucc_status_t ucc_tl_nccl_allreduce_schedule_setup_frag(ucc_schedule_pipelined_t *schedule_p,
                                                       ucc_schedule_frag_t *frag, int frag_num)
{

    ucc_coll_args_t *args = &schedule_p->args.args;
    ucc_datatype_t     dt        = args->src.info.datatype;
    size_t             dt_size   = ucc_dt_size(dt);
    ucc_coll_args_t *targs_rs, *targs_ar, *targs_ag;
    int n_frags = schedule_p->super.n_tasks;

    size_t frag_count = args->src.info.count / n_frags;
    size_t left = args->src.info.count % n_frags;
    size_t offset = frag_num * frag_count + left;

    if (frag_num < left) {
        frag_count++;
        offset -= left - frag_num;
    }

    ucc_tl_nccl_team_t *team = ucc_derived_of(frag->tasks[0], ucc_tl_nccl_task_t)->team;
    size_t chunk_count  = frag_count / team->local_size;
    size_t chunk_size = chunk_count * dt_size;

//    fprintf(stdout, "setup frag %d frag count %d offset %d n_frags %d\n", frag_num, (int)frag_count, (int)offset, n_frags);
    targs_rs = &ucc_derived_of(frag->tasks[0], ucc_tl_nccl_task_t)->args;
    targs_rs->src.info.buffer = PTR_OFFSET(args->src.info.buffer, offset * dt_size);
    targs_rs->dst.info.buffer = PTR_OFFSET(args->dst.info.buffer, offset * dt_size + chunk_size * team->local_rank);
    targs_rs->src.info.count = chunk_count;
    targs_rs->dst.info.count = chunk_count;

    targs_ar = &ucc_derived_of(frag->tasks[1], ucc_tl_nccl_task_t)->args;
    targs_ar->dst.info.buffer = targs_rs->dst.info.buffer;
    targs_ar->src.info.count = chunk_count;
    targs_ar->dst.info.count = chunk_count;

    targs_ag = &ucc_derived_of(frag->tasks[2], ucc_tl_nccl_task_t)->args;
    targs_ag->dst.info.buffer = PTR_OFFSET(args->dst.info.buffer, offset * dt_size);
    targs_ag->src.info.count = chunk_count;
    targs_ag->dst.info.count = chunk_count;

    return UCC_OK;
}
ucc_status_t ucc_tl_nccl_allreduce_schedule_init(ucc_base_coll_args_t *coll_args,
                                                 ucc_tl_nccl_team_t *team,
                                                 ucc_coll_task_t **task_h)
{
    ucc_tl_nccl_lib_t *nccl_lib = ucc_derived_of(team->super.super.context->lib,
                                                 ucc_tl_nccl_lib_t);
    ucc_schedule_pipelined_t *schedule_p;

    size_t msgsize = coll_args->args.src.info.count *
                     ucc_dt_size(coll_args->args.src.info.datatype);

    int n_frags = 1;
    int pipeline_depth;
    if (msgsize > nccl_lib->cfg.frag_thresh) {
        int min_num_frags = msgsize / nccl_lib->cfg.frag_size;
        n_frags = ucc_max(min_num_frags, nccl_lib->cfg.n_frags);
    }

    pipeline_depth = ucc_min(n_frags, nccl_lib->cfg.pipeline_depth);
    // fprintf(stdout, "src %p dst %p nfrags %d pp_depth %d\n", coll_args->args.src.info.buffer,
    //                                            coll_args->args.dst.info.buffer,
    //                                            n_frags, pipeline_depth);
    coll_args->args.dst.info.datatype = coll_args->args.src.info.datatype;
    ucc_schedule_pipelined_init(coll_args, &team->super.super,
                                ucc_tl_nccl_allreduce_schedule_init_frag,
                                ucc_tl_nccl_allreduce_schedule_finalize,
                                ucc_tl_nccl_allreduce_schedule_setup_frag,
                                pipeline_depth, n_frags, &schedule_p);
    *task_h = &schedule_p->super.super;
    return UCC_OK;
}

ucc_status_t ucc_tl_nccl_allgatherv_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_nccl_task_t *task   = ucc_derived_of(coll_task, ucc_tl_nccl_task_t);
    ucc_tl_nccl_team_t *team   = task->team;
    ucc_ee_h            ee     = coll_task->ee;
    cudaStream_t        stream = (ee) ? (cudaStream_t) ee->ee_context : team->stream;
    ucc_status_t        status = UCC_OK;
    void               *sbuf   = task->args.src.info.buffer;
    ptrdiff_t           rbuf   = (ptrdiff_t)task->args.dst.info_v.buffer;
    size_t sdt_size, rdt_size, count, displ;
    ucc_rank_t peer;

    task->super.super.status = UCC_INPROGRESS;
    sdt_size = ucc_dt_size(task->args.src.info.datatype);
    rdt_size = ucc_dt_size(task->args.dst.info_v.datatype);
    NCCLCHECK_GOTO(ncclGroupStart(), exit_coll, status, UCC_TL_TEAM_LIB(team));
    count = task->args.src.info.count;
    if (count != 0) {
        for (peer = 0; peer < team->size; peer++) {
            NCCLCHECK_GOTO(ncclSend(sbuf, count * sdt_size, ncclChar, peer,
                                    team->nccl_comm, stream),
                        exit_coll, status, UCC_TL_TEAM_LIB(team));
        }
    }
    for (peer = 0; peer < team->size; peer++) {
        count = ucc_coll_args_get_count(&task->args,
                                        task->args.dst.info_v.counts, peer);
        if (count != 0) {
            displ = ucc_coll_args_get_displacement(&task->args,
                        task->args.dst.info_v.displacements, peer);
            NCCLCHECK_GOTO(ncclRecv((void *)(rbuf + displ * rdt_size),
                                    count * rdt_size, ncclChar, peer,
                                    team->nccl_comm, stream),
                        exit_coll, status, UCC_TL_TEAM_LIB(team));
        }
    }
    NCCLCHECK_GOTO(ncclGroupEnd(), exit_coll, status, UCC_TL_TEAM_LIB(team));
    status = ucc_mc_ee_event_post(stream, task->completed, UCC_EE_CUDA_STREAM);
exit_coll:
    if (status == UCC_OK) {
        ucc_progress_enqueue(UCC_TL_CORE_CTX(team)->pq, &task->super);
    } else if (ucc_unlikely(status < 0)) {
        task->super.super.status = status;
    }
    return status;
}

ucc_status_t ucc_tl_nccl_allgatherv_init(ucc_tl_nccl_task_t *task)
{
    if (UCC_IS_INPLACE(task->args)) {
        tl_error(UCC_TL_TEAM_LIB(task->team),
                 "inplace allgatherv is not supported");
        return UCC_ERR_NOT_SUPPORTED;
    }
    if ((task->args.src.info_v.datatype == UCC_DT_USERDEFINED) ||
        (task->args.dst.info_v.datatype == UCC_DT_USERDEFINED)) {
        tl_error(UCC_TL_TEAM_LIB(task->team),
                 "user defined datatype is not supported");
        return UCC_ERR_NOT_SUPPORTED;
    }
    task->super.post     = ucc_tl_nccl_allgatherv_start;
    task->super.progress = ucc_tl_nccl_collective_progress;
    return UCC_OK;
}

ucc_status_t ucc_tl_nccl_bcast_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_nccl_task_t *task   = ucc_derived_of(coll_task, ucc_tl_nccl_task_t);
    ucc_tl_nccl_team_t *team   = task->team;
    ucc_ee_h            ee     = coll_task->ee;
    cudaStream_t        stream = (ee) ? (cudaStream_t) ee->ee_context : team->stream;
    void               *src    = task->args.src.info.buffer;
    ncclDataType_t      dt     = ucc_to_nccl_dtype[
                                    task->args.src.info.datatype];
    ucc_status_t        status = UCC_OK;
    size_t              count  = task->args.src.info.count;
    ucc_rank_t          root   = task->args.root;

    task->super.super.status = UCC_INPROGRESS;
    NCCLCHECK_GOTO(ncclBroadcast(src, src, count, dt, root, team->nccl_comm,
                                 stream),
                   exit_coll, status, UCC_TL_TEAM_LIB(team));
    status = ucc_mc_ee_event_post(stream, task->completed, UCC_EE_CUDA_STREAM);
exit_coll:
    if (status == UCC_OK) {
        ucc_progress_enqueue(UCC_TL_CORE_CTX(team)->pq, &task->super);
    } else if (ucc_unlikely(status < 0)) {
        task->super.super.status = status;
    }
    return status;
}

ucc_status_t ucc_tl_nccl_bcast_init(ucc_tl_nccl_task_t *task)
{
    if (UCC_OK != ucc_nccl_check_dt_supported(task->args.src.info.datatype,
                                              task->args.src.info.datatype)) {
        /* TODO: can we use ncclChar if datatype is not supported? */
        tl_error(UCC_TL_TEAM_LIB(task->team),
                 "dataype is not supported");
        return UCC_ERR_NOT_SUPPORTED;
    }
    task->super.post     = ucc_tl_nccl_bcast_start;
    task->super.progress = ucc_tl_nccl_collective_progress;
    return UCC_OK;
}
