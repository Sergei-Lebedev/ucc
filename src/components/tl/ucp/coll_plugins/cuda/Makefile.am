#
# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#

if TLCP_UCP_CUDA_ENABLED

SUBDIRS = kernels

sources = ucp_cuda.c

module_LTLIBRARIES = libucc_tlcp_ucp_cuda.la
libucc_tlcp_ucp_cuda_la_SOURCES  = $(sources)
libucc_tlcp_ucp_cuda_la_CPPFLAGS = $(AM_CPPFLAGS) $(BASE_CPPFLAGS) $(UCX_CPPFLAGS) $(CUDA_CPPFLAGS)
libucc_tlcp_ucp_cuda_la_CFLAGS   = $(BASE_CFLAGS) $(CUDA_CFLAGS)
libucc_tlcp_ucp_cuda_la_LDFLAGS  = -version-info $(SOVERSION) --as-needed $(UCX_LDFLAGS) $(CUDA_LDFLAGS)
libucc_tlcp_ucp_cuda_la_LIBADD   = $(UCX_LIBADD) $(UCC_TOP_BUILDDIR)/src/libucc.la \
                                   $(UCC_TOP_BUILDDIR)/src/components/tl/ucp/libucc_tl_ucp.la kernels/libucc_tl_ucp_cuda_kernels.la \
                                   $(CUDA_LIBS)

include $(top_srcdir)/config/module.am

endif
