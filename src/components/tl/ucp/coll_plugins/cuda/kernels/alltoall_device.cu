#include <ucp/api/device/ucp_device_impl.h>
#include <stdio.h>
extern "C" {
#include "components/tl/ucp/tl_ucp.h"
#include "components/tl/ucp/tl_ucp_coll.h"
}

__global__ void alltoall_device(void *send_buf, void *recv_buf, size_t count)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int nthreads = blockDim.x;
    int nblocks = gridDim.x;
    // ucs_status_t status;


    // status = ucp_device_put_single<UCS_DEVICE_LEVEL_THREAD>(params.mem_list,
    //     params.single.mem_list_index,
    //     params.single.address,
    //     params.single.remote_address,
    //     params.single.length, flags,
    //     req_ptr);


    if (tid == 0) {
        printf("alltoall_device: tid = %d, bid = %d, nthreads = %d, nblocks = %d\n send_buf = %p, recv_buf = %p, count = %d\n", tid, bid, nthreads, nblocks, send_buf, recv_buf, (int)count);
    }
}

#ifdef __cplusplus
extern "C" {
#endif

ucc_status_t tl_ucp_cuda_alltoall_pairwise(ucc_coll_args_t *args)
{
    alltoall_device<<<1, 1>>>(args->src.info.buffer, args->dst.info.buffer, args->src.info.count);
    return UCC_OK;
}

#ifdef __cplusplus
}
#endif