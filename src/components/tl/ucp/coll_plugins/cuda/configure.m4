# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# $COPYRIGHT$
# Additional copyrights may follow

CHECK_TLCP_REQUIRED("ucp_cuda")

AS_IF([test "$CHECKED_TLCP_REQUIRED" = "y"],
[
    tlcp_modules="${tlcp_modules}:ucp_cuda"
    tlcp_ucp_cuda_enabled=y
], [])

AM_CONDITIONAL([TLCP_UCP_CUDA_ENABLED], [test "$tlcp_ucp_cuda_enabled" = "y"])
AC_CONFIG_FILES([src/components/tl/ucp/coll_plugins/cuda/Makefile
                 src/components/tl/ucp/coll_plugins/cuda/kernels/Makefile])

