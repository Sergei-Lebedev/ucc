#include "ucc_base_iface.h"
#include "utils/ucc_malloc.h"
#include "utils/ucc_log.h"

ucc_config_field_t ucc_base_config_table[] = {
    {"LOG_LEVEL", "warn",
     "UCC logging level. Messages with a level higher or equal to the "
     "selected will be printed.\n"
     "Possible values are: fatal, error, warn, info, debug, trace, data, func, "
     "poll.",
     ucc_offsetof(ucc_base_config_t, log_component), UCC_CONFIG_TYPE_LOG_COMP},

    {NULL}};

ucc_status_t ucc_base_config_read(const char *full_prefix,
                                  ucc_config_global_list_entry_t *cfg_entry,
                                  ucc_base_config_t **config)
{
    ucc_base_config_t *cfg;
    ucc_status_t       status;
    cfg = ucc_malloc(cfg_entry->size, "cl_ctx_cfg");
    if (!cfg) {
        ucc_error("failed to allocate %zd bytes for cl context config",
                  cfg_entry->size);
    }
    cfg->cfg_entry = cfg_entry;
    status = ucc_config_parser_fill_opts(cfg, cfg_entry->table, full_prefix,
                                         cfg_entry->prefix, 0);
    if (UCC_OK != status) {
        ucc_free(cfg);
        *config = NULL;
    } else {
        *config = cfg;
    }
    return status;
}
