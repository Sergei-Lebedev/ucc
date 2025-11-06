/**
 * Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_TL_UCP_CUDA_H_
#define UCC_TL_UCP_CUDA_H_

#include "components/tl/ucp/tl_ucp.h"
#include "components/tl/ucp/tl_ucp_coll.h"

typedef struct alltoll_device_task {
    ucp_device_mem_list_handle_h mem_list_h[2];
} alltoall_device_task_t;


ucc_status_t tl_ucp_cuda_alltoall_pairwise(ucc_tl_ucp_task_t *task);

#endif
