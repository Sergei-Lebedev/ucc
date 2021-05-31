/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include "ucc_parser.h"
#include "ucc_malloc.h"
#include "ucc_log.h"

ucc_status_t ucc_config_names_array_dup(ucc_config_names_array_t *dst,
                                        const ucc_config_names_array_t *src)
{
    int i;
    dst->names = ucc_malloc(sizeof(char*) * src->count, "ucc_config_names_array");
    if (!dst->names) {
        ucc_error("failed to allocate %zd bytes for ucc_config_names_array",
                  sizeof(char *) * src->count);
        return UCC_ERR_NO_MEMORY;
    }
    dst->count = src->count;
    for (i = 0; i < src->count; i++) {
        dst->names[i] = strdup(src->names[i]);
        if (!dst->names[i]) {
            ucc_error("failed to dup config_names_array entry");
            goto err;
        }
    }
    return UCC_OK;
err:
    for (i = i - 1; i >= 0; i--) {
        free(dst->names[i]);
    }
    return UCC_ERR_NO_MEMORY;
}

void ucc_config_names_array_free(ucc_config_names_array_t *array)
{
    int i;
    for (i = 0; i < array->count; i++) {
        free(array->names[i]);
    }
    ucc_free(array->names);
}

ucc_status_t ucc_log_component_config_init(ucc_log_component_config_t *log_comp,
                                           const char *name,
                                           ucc_log_level_t log_level)
{
    log_comp->log_level = log_level;
    ucc_strncpy_safe(log_comp->name, name, sizeof(log_comp->name));
#ifdef HAVE_UCS_LOG_COMPONENT_CONFIG_FILE_FILTER
    log_comp->file_filter = strdup("*");
    if (!log_comp->file_filter) {
        ucc_error("failed to dup ucc_log_component_config_t.file_filter");
        return UCC_ERR_NO_MEMORY;
    }
#endif
    return UCC_OK;
}

void ucc_log_component_config_free(ucc_log_component_config_t *log_comp)
{
#ifdef HAVE_UCS_LOG_COMPONENT_CONFIG_FILE_FILTER
    free((void*)log_comp->file_filter);
#endif
    return;
}
