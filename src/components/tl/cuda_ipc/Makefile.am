#
# Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
#

sources =                                  \
	tl_cuda_ipc.h                          \
	tl_cuda_ipc.c                          \
	tl_cuda_ipc_lib.c                      \
	tl_cuda_ipc_context.c                  \
	tl_cuda_ipc_team.c                     \
	tl_cuda_ipc_coll.h                     \
	tl_cuda_ipc_coll.c                     \
	tl_cuda_ipc_cache.h                    \
	tl_cuda_ipc_cache.c                    \
	tl_cuda_ipc_ring.h                     \
	tl_cuda_ipc_ring.c                     \
	tl_cuda_ipc_linear.h                   \
	tl_cuda_ipc_linear.c                   \
	reduce_scatter/reduce_scatter_linear.c \
	reduce_scatter/reduce_scatter_ring.c   \
	allgather/allgather_ring.c             \
	allgather/allgather_linear.c           \
	allgather/allgather_linear_multi.c     \
	alltoallv/alltoallv.c

module_LTLIBRARIES = libucc_tl_cuda_ipc.la
libucc_tl_cuda_ipc_la_SOURCES  = $(sources)
libucc_tl_cuda_ipc_la_CPPFLAGS = $(AM_CPPFLAGS) $(BASE_CPPFLAGS) $(CUDA_CPPFLAGS)
libucc_tl_cuda_ipc_la_CFLAGS   = $(BASE_CFLAGS)
libucc_tl_cuda_ipc_la_LDFLAGS  = -version-info $(SOVERSION) --as-needed $(CUDA_LDFLAGS)
libucc_tl_cuda_ipc_la_LIBADD   = $(CUDA_LIBS) $(UCC_TOP_BUILDDIR)/src/libucc.la

include $(top_srcdir)/config/module.am
