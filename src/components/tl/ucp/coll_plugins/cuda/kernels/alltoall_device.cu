#include <ucp/api/device/ucp_device_impl.h>
#include <stdio.h>
extern "C" {
#include "components/tl/ucp/tl_ucp.h"
#include "components/tl/ucp/tl_ucp_coll.h"
#include "components/tl/ucp/coll_plugins/cuda/ucp_cuda.h"
}

__global__ void alltoall_device(size_t count,
                                ucp_device_mem_list_handle_h mem0,
                                ucp_device_mem_list_handle_h mem1)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int nthreads = blockDim.x;
    int nblocks = gridDim.x;
    int tsize = 2;
    ucp_device_mem_list_handle_h mem_list_h[2] = {mem0, mem1};

    if (tid == 0) {
        for (int i = 0; i < tsize; i++) {
            ucp_device_put_single<UCS_DEVICE_LEVEL_THREAD>(
                mem_list_h[i], 0, 0, 0, count, 0, 0, NULL);
        }
    }
}

#ifdef __cplusplus
extern "C" {
#endif

ucc_status_t tl_ucp_cuda_alltoall_pairwise(ucc_tl_ucp_task_t *task)
{
    ucc_coll_args_t *args = &task->super.bargs.args;
    size_t count = args->src.info.count * ucc_dt_size(args->src.info.datatype);
    alltoall_device_task_t *a2a_task_data = (alltoall_device_task_t *)(task->plugin_data);
    cudaError_t cuda_error;

    alltoall_device<<<1, 1>>>(count,
                              a2a_task_data->mem_list_h[0],
                              a2a_task_data->mem_list_h[1]);
    cuda_error = cudaGetLastError();
    if (cuda_error != cudaSuccess) {
        tl_error(UCC_TASK_LIB(task), "cudaGetLastError() failed: %s", cudaGetErrorString(cuda_error));
        return UCC_ERR_NO_MESSAGE;
    }
    return UCC_OK;
}

#ifdef __cplusplus
}
#endif