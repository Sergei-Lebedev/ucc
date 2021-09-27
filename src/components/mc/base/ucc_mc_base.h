/**
 * Copyright (C) Mellanox Technologies Ltd. 2020-2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_MC_BASE_H_
#define UCC_MC_BASE_H_

#include "config.h"
#include "utils/ucc_component.h"
#include "utils/ucc_class.h"
#include "utils/ucc_parser.h"
#include "utils/ucc_mpool.h"
#include "core/ucc_global_opts.h"

typedef struct ucc_mc_params {
    ucc_thread_mode_t thread_mode;
} ucc_mc_params_t;

typedef struct ucc_mc_buffer_header {
    ucc_memory_type_t mt;
    int               from_pool;
    void             *addr;
} ucc_mc_buffer_header_t;

/**
 * UCC memory attributes field mask
 */
typedef enum ucc_mem_attr_field {
    UCC_MEM_ATTR_FIELD_MEM_TYPE     = UCC_BIT(0), /**< Indicate if memory type
                                                       is populated. E.g. CPU/GPU */
    UCC_MEM_ATTR_FIELD_BASE_ADDRESS = UCC_BIT(2), /**< Request base address of the
                                                       allocation */
    UCC_MEM_ATTR_FIELD_ALLOC_LENGTH = UCC_BIT(3)  /**< Request the whole length of the
                                                       allocation */
} ucc_mem_attr_field_t;


typedef struct ucc_mem_attr {
    /**
     * Mask of valid fields in this structure, using bits from
     * @ref ucc_mem_attr_field_t.
     */
    uint64_t          field_mask;

    /**
     * The type of memory. E.g. CPU/GPU memory or some other valid type
     */
    ucc_memory_type_t mem_type;

    /**
     * Base address of the allocation to which the provided buffer belongs to.
     * If the mc not support base address query, then the pointer passed to
     * ucc_mc_cuda_mem_query is returned as is.
     */
    void              *base_address;

    /**
     * If UCC_MEM_ATTR_FIELD_ALLOC_LENGTH is set this parameter should be equal
     * to known allocation length. On output length of the whole allocation
     * to which the provided buffer belongs to is returned.
     * If the md not support querying allocation length, then the length passed
     * to ucc_mc_cuda_mem_query is returned as is.
     */
    size_t            alloc_length;

} ucc_mem_attr_t;

/**
 * UCC memory component attributes field mask
 */
typedef enum ucc_mc_attr_field {
    UCC_MC_ATTR_FIELD_THREAD_MODE = UCC_BIT(0)
}  ucc_mc_attr_field_t;

typedef struct ucc_mc_attr {
    /**
     * Mask of valid fields in this structure, using bits from
     * @ref ucc_mc_attr_field_t.
     */
    uint64_t          field_mask;
    ucc_thread_mode_t thread_mode;
} ucc_mc_attr_t;

/**
 * Array of string names for each memory type
 */
extern const char *ucc_memory_type_names[];

/**
 * Array of string names for each execution engine type
 */
extern const char *ucc_ee_type_names[];

typedef struct ucc_mc_config {
    ucc_log_component_config_t log_component;
} ucc_mc_config_t;
extern ucc_config_field_t ucc_mc_config_table[];

typedef struct ucc_mc_ops {
    ucc_status_t (*mem_query)(const void *ptr, ucc_mem_attr_t *mem_attr);
    ucc_status_t (*mem_alloc)(ucc_mc_buffer_header_t **h_ptr, size_t size);
    ucc_status_t (*mem_free)(ucc_mc_buffer_header_t *h_ptr);
    ucc_status_t (*reduce)(const void *src1, const void *src2, void *dst,
                           size_t count, ucc_datatype_t dt,
                           ucc_reduction_op_t op);
    ucc_status_t (*reduce_multi)(const void *src1, const void *src2, void *dst,
                                 size_t n_vectors, size_t count, size_t stride,
                                 ucc_datatype_t dt, ucc_reduction_op_t op);
    ucc_status_t (*memcpy)(void *dst, const void *src, size_t len,
                           ucc_memory_type_t dst_mem,
                           ucc_memory_type_t src_mem);
 } ucc_mc_ops_t;

typedef struct ucc_ee_ops {
    ucc_status_t (*ee_task_post)(void *ee_context, void **ee_req);
    ucc_status_t (*ee_task_enqueue)(void *ee_context, void **ee_req);
    ucc_status_t (*ee_task_sync)(void *ee_req);
    ucc_status_t (*ee_task_query)(void *ee_req);
    ucc_status_t (*ee_task_end)(void *ee_req);
    ucc_status_t (*ee_create_event)(void **event);
    ucc_status_t (*ee_destroy_event)(void *event);
    ucc_status_t (*ee_event_post)(void *ee_context, void *event);
    ucc_status_t (*ee_event_test)(void *event);
} ucc_ee_ops_t;

typedef struct ucc_ee_executor {
    ucc_ee_type_t  ee_type;
    void          *ee_context;
} ucc_ee_executor_t;

typedef enum ucc_ee_executor_task_type {
    UCC_MC_EE_EXECUTOR_TASK_TYPE_WAIT         = UCC_BIT(0),
    UCC_MC_EE_EXECUTOR_TASK_TYPE_REDUCE       = UCC_BIT(1),
    UCC_MC_EE_EXECUTOR_TASK_TYPE_REDUCE_MULTI = UCC_BIT(2),
    UCC_MC_EE_EXECUTOR_TASK_TYPE_COPY         = UCC_BIT(3),
} ucc_ee_executor_task_type_t;

typedef struct ucc_ee_executor_params {
    uint64_t        mask;
    ucc_ee_type_t   ee_type;
    void           *ee_context;
    uint64_t        task_types;
} ucc_ee_executor_params_t;

typedef struct ucc_ee_executor_task_args {
    ucc_ee_executor_task_type_t task_type;
    ucc_coll_buffer_info_t      src1;
    ucc_coll_buffer_info_t      src2;
    void                       *src3[8];
    ucc_count_t                 src3_size;
    ucc_coll_buffer_info_t      dst;
    ucc_reduction_op_t          op;
} ucc_ee_executor_task_args_t;

/*
 *  buffers[0] - destination
 *  buffers[1] .. buffers[UCC_EE_EXECUTOR_NUM_BUFS - 1] - source
 *  count - number of elements in destination
 *  size - number of operands
 *  stride - step between source buffers
 *  dt - datatype
 *  op - reduction operation
 */
// typedef struct ucc_ee_executor_task_task {
//     void               *buffers[UCC_EE_EXECUTOR_NUM_BUFS];
//     ucc_count_t         count;
//     ucc_count_t         size;
//     ucc_count_t         stride;
//     ucc_datatype_t      dt;
//     ucc_reduction_op_t  op;
// } ucc_ee_executor_task_args_t;

typedef struct ucc_ee_executor_task {
    ucc_ee_executor_t           *eee;
    ucc_ee_executor_task_args_t  args;
    ucc_status_t                 status;
} ucc_ee_executor_task_t;

typedef struct ucc_ee_executor_ops {
    ucc_status_t (*executor_create_post)(const ucc_ee_executor_params_t *params,
                                            ucc_ee_executor_t **executor);
    ucc_status_t (*executor_create_test)(ucc_ee_executor_t *executor);
    ucc_status_t (*executor_task_post)(ucc_ee_executor_task_args_t *task_args,
                                       ucc_ee_executor_task_t **task,
                                       ucc_ee_executor_t *executor);
    ucc_status_t (*executor_task_test)(ucc_ee_executor_task_t *task);
    ucc_status_t (*executor_destroy)(ucc_ee_executor_t *executor);
} ucc_ee_executor_ops_t;

typedef struct ucc_mc_base {
    ucc_component_iface_t           super;
    uint32_t                        ref_cnt;
    ucc_ee_type_t                   ee_type;
    ucc_memory_type_t               type;
    ucc_mc_config_t                *config;
    ucc_config_global_list_entry_t  config_table;
    ucc_status_t                   (*init)(const ucc_mc_params_t *mc_params);
    ucc_status_t                   (*get_attr)(ucc_mc_attr_t *mc_attr);
    ucc_status_t                   (*finalize)();
    ucc_mc_ops_t                    ops;
    const ucc_ee_ops_t              ee_ops;
    const ucc_ee_executor_ops_t     executor_ops;
} ucc_mc_base_t;

#endif
