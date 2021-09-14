/**
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "cl_hier.h"
#include "utils/ucc_malloc.h"
ucc_status_t ucc_cl_hier_get_lib_attr(const ucc_base_lib_t *lib,
                                       ucc_base_lib_attr_t  *base_attr);
ucc_status_t ucc_cl_hier_get_context_attr(const ucc_base_context_t *context,
                                           ucc_base_ctx_attr_t      *base_attr);

static ucc_config_field_t ucc_cl_hier_lib_config_table[] = {
    {"", "", NULL, ucc_offsetof(ucc_cl_hier_lib_config_t, super),
     UCC_CONFIG_TYPE_TABLE(ucc_cl_lib_config_table)},

    {"NODE_SBGP_TLS", "ucp,nccl",
     "TLS to be used for NODE subgroup",
     ucc_offsetof(ucc_cl_hier_lib_config_t, sbgp_tls[UCC_HIER_SBGP_NODE]),
     UCC_CONFIG_TYPE_STRING_ARRAY},

    {"NET_SBGP_TLS", "ucp,nccl",
     "TLS to be used for NET subgroup",
     ucc_offsetof(ucc_cl_hier_lib_config_t, sbgp_tls[UCC_HIER_SBGP_NET]),
     UCC_CONFIG_TYPE_STRING_ARRAY},

    {"FULL_SBGP_TLS", "ucp",
     "TLS to be used for FULL subgroup",
     ucc_offsetof(ucc_cl_hier_lib_config_t, sbgp_tls[UCC_HIER_SBGP_FULL]),
     UCC_CONFIG_TYPE_STRING_ARRAY},

    {"A2AV_NODE_THRESH", "0",
     "Messages larger than that threshold will be sent via node sbgp tl",
     ucc_offsetof(ucc_cl_hier_lib_config_t, a2av_node_thresh),
     UCC_CONFIG_TYPE_MEMUNITS},

    {"ALLREDUCE_HYBRID_FRAG_THRESH", "inf",
     "Threshold to enable fragmentation and pipelining of Hybrid allreduce alg",
     ucc_offsetof(ucc_cl_hier_lib_config_t, allreduce_hybrid_frag_thresh),
     UCC_CONFIG_TYPE_MEMUNITS},

    {"ALLREDUCE_HYBRID_FRAG_SIZE", "inf",
     "Maximum allowed fragment size of hybrid allreduce algorithm",
     ucc_offsetof(ucc_cl_hier_lib_config_t, allreduce_hybrid_frag_size),
     UCC_CONFIG_TYPE_MEMUNITS},

    {"ALLREDUCE_HYBRID_N_FRAGS", "2",
     "Number of fragments each allreduce is split into when hybrid alg is used\n"
     "The actual number of fragments can be larger if fragment size exceeds\n"
     "ALLREDUCE_HYBRID_FRAG_SIZE",
     ucc_offsetof(ucc_cl_hier_lib_config_t, allreduce_hybrid_n_frags),
     UCC_CONFIG_TYPE_UINT},

    {"ALLREDUCE_HYBRID_PIPELINE_DEPTH", "2",
     "Number of fragments simultaneously progressed by the hybrid alg",
     ucc_offsetof(ucc_cl_hier_lib_config_t, allreduce_hybrid_pipeline_depth),
     UCC_CONFIG_TYPE_UINT},

    {"ALLREDUCE_HYBRID_SEQUENTIAL", "0",
     "Type of pipelined schedule for hybrid AR (sequential/parallel)",
     ucc_offsetof(ucc_cl_hier_lib_config_t, allreduce_hybrid_seq),
     UCC_CONFIG_TYPE_UINT},

    {NULL}
};

static ucs_config_field_t ucc_cl_hier_context_config_table[] = {
    {"", "", NULL, ucc_offsetof(ucc_cl_hier_context_config_t, super),
     UCC_CONFIG_TYPE_TABLE(ucc_cl_context_config_table)},

    {NULL}
};

UCC_CLASS_DEFINE_NEW_FUNC(ucc_cl_hier_lib_t, ucc_base_lib_t,
                          const ucc_base_lib_params_t *,
                          const ucc_base_config_t *);

UCC_CLASS_DEFINE_DELETE_FUNC(ucc_cl_hier_lib_t, ucc_base_lib_t);

UCC_CLASS_DEFINE_NEW_FUNC(ucc_cl_hier_context_t, ucc_base_context_t,
                          const ucc_base_context_params_t *,
                          const ucc_base_config_t *);

UCC_CLASS_DEFINE_DELETE_FUNC(ucc_cl_hier_context_t, ucc_base_context_t);

UCC_CLASS_DEFINE_NEW_FUNC(ucc_cl_hier_team_t, ucc_base_team_t,
                          ucc_base_context_t *, const ucc_base_team_params_t *);

ucc_status_t ucc_cl_hier_team_create_test(ucc_base_team_t *cl_team);

ucc_status_t ucc_cl_hier_team_destroy(ucc_base_team_t *cl_team);

ucc_status_t ucc_cl_hier_team_get_scores(ucc_base_team_t   *cl_team,
                                          ucc_coll_score_t **score);
UCC_CL_IFACE_DECLARE(hier, HIER);
